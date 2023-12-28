# TODO: Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, softmax
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.src.callbacks import History


# TODO: Load the dataset
# data = pd.read_csv('dataset/dataset.csv')
data = pd.read_csv('dataset/dataset_version2.csv')
# print(data.head(10), '\n\n\n')
# print(data.tail(10))

# TODO: Define x train, x test, y train, y test
X = data[['wall_up', 'wall_right', 'wall_down', 'wall_left', 'apple_up',
          'apple_right', 'apple_down', 'apple_left', 'distance_x', 'distance_y']]
Y = data['direction']
# print(X.head(10), '\n\n\n\n')
# print(Y.head(10))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
# y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# TODO: Implement the network
model = Sequential([
    Dense(len(x_train.columns), activation=relu),
    Dense(64, activation=relu),
    Dense(32, activation=relu),
    Dense(4, activation=softmax)
])

model.compile(optimizer=Adam(),
              loss=sparse_categorical_crossentropy, metrics=['accuracy'])

# TODO: Train the model
output_history: History = model.fit(x_train, y_train, epochs=60)

# TODO: Save the model
# model.save('Snake_weight_60ep.h5')
model.save('Snake_weight_60ep_version2.h5')

# TODO: Plot results
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

plt.plot(output_history.history['loss'], label='Losses')
plt.plot(output_history.history['accuracy'], label='Accuracies')
plt.title('Model Train Losses And Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
