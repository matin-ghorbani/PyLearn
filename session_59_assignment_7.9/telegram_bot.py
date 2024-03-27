import os

import telebot
from telebot.types import Message

import pydub
import numpy as np
from keras.models import Sequential, load_model

from audio_processor import AudioProcessor


bot = telebot.TeleBot('token', parse_mode=None)
friend: bool = False
singer: bool = False
friends_model: Sequential = load_model('./weights/best_audio_classifier_50ep.h5')
singers_model: Sequential = load_model('./weights/best_singer_classifier_50ep.h5')


@bot.message_handler(commands=['start'])
def send_welcome(message: Message):
    bot.reply_to(
        message, f'سلام {message.from_user.first_name} خوبی؟')
    bot.send_message(
        message.chat.id, 'من ربات متینم یک /help بزن ببینیم چه خبره')


@bot.message_handler(commands=['help'])
def send_help(message: Message):
    bot.send_message(
        message.chat.id, 'دو تا انتخاب داری اولیش /friends هست و اون یکی دیگه /singers هست')


@bot.message_handler(commands=['friend'])
def get_img(message: Message):
    global friend, singer
    friend = True
    singer = False
    bot.send_message(message.chat.id, 'یک ویس برام بفرست')


@bot.message_handler(commands=['singer'])
def get_img(message: Message):
    global friend, singer
    singer = True
    friend = False
    bot.send_message(message.chat.id, 'یک ویس برام بفرست')


@bot.message_handler(content_types=["voice"])
def voice(message: Message):
    global friend, singer, friends_model, singers_model
    
    voice = bot.get_file(message.voice.file_id)
    voice = bot.download_file(voice.file_path)

    with open('temp.wav', 'wb') as file:
        file.write(voice)
    
    voice: pydub.audio_segment.AudioSegment = pydub.AudioSegment.from_file('temp.wav')
    result = AudioProcessor.last_config(voice)
    result.export("ready_for_predict.wav", format="wav")

    voice_to_predict = AudioProcessor.convert_audio_to_model_input('ready_for_predict.wav')

    friends_list = os.listdir('./friends/dataset')
    singers_list = os.listdir('./Singers/dataset')

    
    if friend:
        prediction = np.argmax(
            friends_model.predict(voice_to_predict)
        )
        prediction = friends_list[prediction]
        bot.reply_to(message, f'سلام {prediction}')

    elif singer:
        prediction = np.argmax(
            singers_model.predict(voice_to_predict)
        )
        prediction = singers_list[prediction]
        bot.reply_to(message, f'نام خواننده: {prediction}')

bot.infinity_polling()
