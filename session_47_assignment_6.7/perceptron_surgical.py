import numpy as np
from tqdm import tqdm


class Perceptron:
    def __init__(self, input_length, learning_rate):
        self.weights = np.random.rand(input_length)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate

    def activation(self, x, function='sigmoid'):
        if function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif function == 'relu':
            return np.maximum(0, x)
        elif function == 'tanh':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif function == 'sign':
            if x > 0:
                return 1
            elif x == 0:
                return 0
            elif x < 0:
                return -1
        elif function == 'unitstep':
            if x > 0:
                return 1
            elif x == 0:
                return 0.5
            elif x < 0:
                return 0
        else:
            raise Exception("Not supported activation function")

    def fit(self, X_train, Y_train, epochs, func='sigmoid'):
        for _ in tqdm(range(epochs)):
            for x_train, y_train in zip(X_train, Y_train):
                y_pred = self.activation(x_train @ self.weights + self.bias, func)
                error = y_pred - y_train

                self.weights = self.weights - self.learning_rate * error * x_train
                self.bias = self.bias - self.learning_rate * error

    def predict(self, X_test, func='sigmoid'):
        Y_pred = []
        for x_test in X_test:
            y_pred = self.activation(x_test @ self.weights + self.bias, func)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def calculate_loss(self, X_test, Y_test, metric='mse'):
        y_pred = self.predict(X_test)
        if metric == 'mse':
            loss = np.mean((y_pred - Y_test) ** 2)
        elif metric == 'mae':
            loss = np.mean(np.abs(y_pred - Y_test))
        else:
            raise Exception('Not supported metric')
        return loss

    def calculate_accuracy(self, X_test, Y_test, func):
        Y_pred = self.predict(X_test, func)
        Y_pred = Y_pred.reshape(-1)
        if func == 'sigmoid':
            Y_pred = np.where(Y_pred > 0.5, 1, 0)
        elif func == 'relu':
            Y_pred = np.where(Y_pred > 0, 1, 0)
        elif func == 'tanh':
            Y_pred = np.where(Y_pred > 0, 1, 0)
        elif func == 'unitstep':
            Y_pred = np.where(Y_pred > 0.5, 1, 0)
        elif func == 'sign':
            Y_pred = np.where(Y_pred > 0, 1, 0)
        else:
            raise Exception('Not supported activation function')

        return np.mean(Y_pred == Y_test)

    def evaluate(self, X_test, Y_test, func):
        loss = self.calculate_loss(X_test, Y_test)
        accuracy = self.calculate_accuracy(X_test, Y_test, func)
        return loss, accuracy
