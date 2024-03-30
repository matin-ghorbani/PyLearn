from keras.models import Sequential, load_model
from keras import layers
from keras.activations import softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import History, ModelCheckpoint

import numpy as np
import pandas as pd


class EmojiTextClassifier:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.model = None
        self.words_vectors: dict[str, np.ndarray] = {}

    @staticmethod
    def load_dataset(dataset_path: str) -> list[np.ndarray, np.ndarray]:
        df = pd.read_csv(dataset_path)
        x = np.array(df['sentence'])
        y = np.array(df['label'], dtype=int)

        return x, y

    def load_feature_vectors(self, file_path: str) -> None:
        line: str

        file = open(file_path, encoding='utf-8')
        for line in file:
            line = line.strip().split(' ')
            word = line[0]
            vector = np.array(line[1:], dtype=np.float64)
            self.words_vectors[word] = vector

    def sentence_to_feature_vectors_avg(self, sentence: str) -> np.ndarray[np.floating] | None:
        sentence = sentence.lower()
        words = sentence.strip().split(' ')
        sum_vectors = np.zeros((self.dimension, ))

        try:
            for word in words:
                vector = self.words_vectors[word]
                sum_vectors += vector
            return sum_vectors / len(words)
        except KeyError:
            print(f'There is an unknown word in this sentence: "{sentence}"')
    
    def convert_sentences_to_vectors(self, sentences: np.ndarray) -> np.ndarray:
        sentences_avg = []
        for sentence in sentences:
            sentences_avg.append(
                self.sentence_to_feature_vectors_avg(sentence)
            )
        
        return np.array(sentences_avg)

    def build_model(self, with_dropout: bool = False) -> None:
        if with_dropout:
            self.model = Sequential([
                layers.Dropout(.5, name='DropoutLayer'),
                layers.Dense(5, activation=softmax,
                             input_shape=(self.dimension, ), name='OutputLayer')
            ])

        else:
            self.model = Sequential([
                layers.Dense(5, activation=softmax,
                             input_shape=(self.dimension, ), name='OutputLayer')
            ])

    def compile_model(self, optimizer=Adam(), loss=categorical_crossentropy) -> None:
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

    def train_model(self, x_train, y_train, epochs: int, model_path_to_save: str = 'best_emojis_classifier.keras') -> History:
        check = ModelCheckpoint(model_path_to_save,
                                monitor='accuracy', save_best_only=True)

        history: History = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            callbacks=[check]
        )

        return history

    @staticmethod
    def evaluate(model_path, x_test, y_test) -> None:
        model: Sequential = load_model(model_path)
        print('\nEvaluating model...')
        model.evaluate(x_test, y_test)
    
    def load_model(self, model_path: str) -> None:
        self.model: Sequential = load_model(model_path)

    def predict(self, sentence: str) -> str:
        sentence_avg = self.sentence_to_feature_vectors_avg(sentence)
        sentence_avg = np.array([sentence_avg])

        prediction = self.model.predict(sentence_avg)
        y_hat = np.argmax(prediction)   

        return EmojiTextClassifier.covert_label_to_emoji(y_hat)

    @staticmethod
    def covert_label_to_emoji(label: int) -> str:
        emojis = ['🧡', '⚾', '😃', '😔', '🍴']
        return emojis[label]
