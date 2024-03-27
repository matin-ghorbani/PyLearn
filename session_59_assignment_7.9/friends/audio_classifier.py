import os
import shutil

import pydub
import numpy as np
import librosa as lr
from tqdm import tqdm

from keras.models import Sequential, load_model
from keras import layers
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.activations import relu, softmax
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras.utils import audio_dataset_from_directory


class AudioProcessor:
    @staticmethod
    def merge_audios(*audios):
        audio: pydub.audio_segment.AudioSegment
        result: pydub.audio_segment.AudioSegment = 0

        for audio in audios[0]:
            result += audio

        return result

    @staticmethod
    def remove_silences(audio, min_silence_length=2000, silence_threshold=-45):
        chunks = pydub.silence.split_on_silence(
            audio, min_silence_len=min_silence_length, silence_thresh=silence_threshold
        )
        return sum(chunks)

    @staticmethod
    def split_audio(audio, duration=2000):
        chunks = pydub.utils.make_chunks(audio, duration)
        return [chunk for chunk in chunks if len(chunk) >= duration]

    def last_config(self, audio: pydub.audio_segment.AudioSegment):
        audio = audio.set_sample_width(2)
        audio = audio.set_frame_rate(48000)
        audio = audio.set_channels(1)
        audio = self.remove_silences(audio)

        return audio

    # @staticmethod
    # def convert_audio_to_model_input(audio: pydub.audio_segment.AudioSegment):
    #     # Convert Pydub audio segment to NumPy array
    #     audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)

    #     # Reshape to expected input format (48000 samples, 1 channel)
    #     audio_array = np.reshape(audio_array, (48000, 1))

    #     return audio_array

    @staticmethod
    def convert_audio_to_model_input(audio_path: str) -> np.ndarray:
        audio = lr.load(audio_path, sr=None)[0]
        audio = lr.util.fix_length(audio, size=48000)

        audio = np.expand_dims(audio, axis=-1)
        return np.expand_dims(audio, axis=0)


    @staticmethod
    def find_duplicates(file_path: str, directory_path: str):
        file_name = file_path.split('/')[-1].split('.')[0]
        audios = []

        for file in os.listdir(directory_path):
            parts = file.split('.')[0].split('_')
            if parts[0] == file_name.split('_')[0]:
                audios.append(file)

        return audios

    @staticmethod
    def create_dataset_for_training(raw_data_path: str, new_dataset_path):
        new_voice: pydub.audio_segment.AudioSegment

        for voice_name in tqdm(os.listdir(raw_data_path)):
            voice_speaker = voice_name.split('.')[0]

            voice = pydub.AudioSegment.from_file(
                os.path.join(raw_data_path, voice_name)
            )

            founded_duplicates = AudioProcessor.find_duplicates(
                voice_name, raw_data_path)
            if len(founded_duplicates):
                voice = AudioProcessor.merge_audios([
                    pydub.AudioSegment.from_file(os.path.join(raw_data_path, audio)) for audio in founded_duplicates
                ])

                speaker_path = os.path.join(
                    new_dataset_path, voice_speaker.split('_')[0])
            else:
                speaker_path = os.path.join(new_dataset_path, voice_speaker)

            os.makedirs(speaker_path, exist_ok=True)
            voice = AudioProcessor.last_config(voice)
            voices = AudioProcessor.split_audio(voice)

            for i, new_voice in enumerate(voices, 1):
                new_voice.export(
                    os.path.join(speaker_path, f'{voice_speaker}_{i}.wav'), format='wav'
                )


class AudioClassifier:
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path
        self.train_data = None
        self.validation_data = None

    def make_train_validation_datasets(self, validation_split=.2, batch_size=2, output_sequence_length=48000) -> None:
        self.train_data = audio_dataset_from_directory(
            self.dataset_path,
            batch_size=batch_size,
            validation_split=validation_split,
            output_sequence_length=output_sequence_length,
            label_mode='categorical',
            sampling_rate=None,
            seed=59,
            subset='training'
        )

        self.validation_data = audio_dataset_from_directory(
            self.dataset_path,
            batch_size=batch_size,
            validation_split=validation_split,
            output_sequence_length=output_sequence_length,
            label_mode='categorical',
            sampling_rate=None,
            seed=59,
            subset='validation'
        )

    def build_model(self, optimizer=Adam, loss=categorical_crossentropy):
        number_of_persons = 0
        for person in os.listdir(self.dataset_path):
            if os.path.isdir(os.path.join(self.dataset_path, person)):
                number_of_persons += 1

        self.model = Sequential([
            layers.Conv1D(32, 40, 16, activation=relu,
                          input_shape=(48000, 1), padding='same'),
            layers.MaxPool1D(4),

            layers.Conv1D(256, 40, activation=relu,
                          input_shape=(48000, 1), padding='same'),
            layers.MaxPool1D(4),

            layers.Conv1D(128, 40, activation=relu,
                          input_shape=(48000, 1), padding='same'),
            layers.MaxPool1D(4),

            layers.Conv1D(64, 40, activation=relu,
                          input_shape=(48000, 1), padding='same'),
            layers.MaxPool1D(4),

            layers.Conv1D(32, 40, activation=relu,
                          input_shape=(48000, 1), padding='same'),
            layers.MaxPool1D(4),

            layers.Flatten(),
            layers.Dense(32, activation=relu),
            layers.Dense(number_of_persons, activation=softmax)
        ])

        self.model.compile(optimizer=optimizer(),
                           loss=loss, metrics=['accuracy'])

    def train_model(self, model_save_path, epochs=30):
        stop = EarlyStopping(patience=15, monitor='val_accuracy')
        check = ModelCheckpoint(
            model_save_path, save_best_only=True, monitor='val_accuracy')

        self.history: History = self.model.fit(
            self.train_data,
            validation_data=self.validation_data,
            epochs=epochs,
            callbacks=[stop, check]
        )

    def evaluate(self, model_path):
        model: Sequential = load_model(model_path)
        model.evaluate(self.validation_data)

    @staticmethod
    def predict(model_path, audio: np.ndarray):
        model: Sequential = load_model(model_path)
        predictions: np.ndarray = model.predict(audio)

        return np.argmax(predictions)


if __name__ == '__main__':
    AudioProcessor.create_dataset_for_training('./raw_data', './dataset')

    classifier = AudioClassifier('./dataset')
    classifier.make_train_validation_datasets()
    classifier.build_model()
    classifier.train_model(
        './weights/best_audio_classifier_50ep.h5', epochs=50)
    classifier.evaluate('./weights/best_audio_classifier_50ep.h5')
