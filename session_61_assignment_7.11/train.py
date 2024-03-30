from argparse import ArgumentParser, BooleanOptionalAction

from keras.callbacks import History
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from text_classifier import EmojiTextClassifier

args = ArgumentParser()
args.add_argument('--train-dataset', type=str,
                  default='./dataset/train.csv', help='The train dataset path')
args.add_argument('--test-dataset', type=str,
                  default='./dataset/test.csv', help='The test dataset path')
args.add_argument('--dimension', type=int,
                  default=50, help='The dimension of feature vectors')
args.add_argument('--vectors-file', type=str,
                  default='./glove_6B/glove.6B.200d.txt', help='The feature vectors file path')
args.add_argument('--dropout', type=bool,
                  default=False, action=BooleanOptionalAction, help='Add dropout layer to network')
args.add_argument('--model-save', type=str,
                  default='best_emojis_classifier.keras', help='The best model path to save')
args.add_argument('--epochs', type=int,
                  default=200, help='The number of epochs to train the model')
args.add_argument('--save-plots', type=bool,
                  default=True, action=BooleanOptionalAction, help='Save the training information plots')

opt = args.parse_args()

classifier = EmojiTextClassifier(opt.dimension)
x_train, y_train = EmojiTextClassifier.load_dataset(opt.train_dataset)
x_test, y_test = EmojiTextClassifier.load_dataset(opt.test_dataset)

classifier.load_feature_vectors(opt.vectors_file)
x_train = classifier.convert_sentences_to_vectors(x_train)
x_test = classifier.convert_sentences_to_vectors(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

classifier.build_model(opt.dropout)
classifier.compile_model()
history: History = classifier.train_model(
    x_train, y_train, opt.epochs, opt.model_save)

EmojiTextClassifier.evaluate(opt.model_save, x_test, y_test)

if opt.save_plots:
    fig, (ax1, ax2) = plt.subplots(1, 2)[1]
    ax1.plot(history.history['accuracy'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')

    ax2.plot(history.history['loss'])
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    fig.suptitle(f'Emojis Classification With Dropout: {False}')
    plt.savefig(f'Emojis_Classification_{opt.epochs}ep_dropout_{opt.dropout}.png')
