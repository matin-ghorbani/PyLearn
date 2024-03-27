import os

from audio_classifier import AudioClassifier, AudioProcessor

test_audio = AudioProcessor.convert_audio_to_model_input('./tests/Yas.wav')
prediction = AudioClassifier.predict('./weights/best_singer_classifier_50ep.h5', test_audio)

names = os.listdir('./dataset')
print(names[prediction])
