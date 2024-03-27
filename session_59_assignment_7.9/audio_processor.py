import os
import shutil

import pydub
import librosa as lr
from spleeter.separator import Separator
import numpy as np
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

    @staticmethod
    def last_config(audio: pydub.audio_segment.AudioSegment):
        audio = audio.set_sample_width(2)
        audio = audio.set_frame_rate(48000)
        audio = audio.set_channels(1)
        audio = AudioProcessor.remove_silences(audio)

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
    def extract_speaker_voice(audio_path: str, save_path: str) -> None:
        separator = Separator("spleeter:2stems")
        separator.separate_to_file(audio_path, save_path)

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
    
    @staticmethod
    def create_dataset_for_singers(singers, dataset_path, new_dataset_path):
        os.makedirs(new_dataset_path, exist_ok=True)
        new_audio: pydub.audio_segment.AudioSegment
        for singer in tqdm(singers):
            singer_dir = os.path.join(new_dataset_path, singer)
            os.makedirs(singer_dir, exist_ok=True)

            first = os.path.join(dataset_path, singer + '_1', 'vocals.wav')
            second = os.path.join(dataset_path, singer + '_2', 'vocals.wav')

            audio = pydub.AudioSegment.from_file(first) + pydub.AudioSegment.from_file(second)
            audio = AudioProcessor.last_config(audio)
            audio = AudioProcessor.split_audio(audio)

            for i, new_audio in enumerate(audio, 1):
                new_audio.export(
                    os.path.join(singer_dir, f'{singer}_{i}.wav'), format='wav'
                )
