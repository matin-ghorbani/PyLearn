import numpy as np
from tqdm import tqdm


class Perceptron:
    def __init__(self, input_length, learning_rate, function="sigmoid"):
        self.weights = np.random.rand(input_length)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
        self.function = function

    def activation(self, x):
        if self.function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.function == "relu":
            return np.maximum(0, x)
        elif self.function == "tanh":
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif self.function == "linear":
            return x
        elif self.function == "Leaky ReLU":
            if x >= 0:
                return x
            else:
                return 0.2 * x
        elif self.function == 'sign':
            if x > 0:
                return 1
            elif x == 0:
                return 0
            elif x < 0:
                return -1
        elif self.function == 'unitstep':
            if x > 0:
                return 1
            elif x == 0:
                return 0.5
            elif x < 0:
                return 0
        else:
            raise Exception("Not supported activation function")

    def forward(self, x):
        return self.activation(x * self.weights + self.bias)

    def back_propagation(self, x_train, y_train, y_pred):
        d_w = (y_pred - y_train) * x_train
        d_b = (y_pred - y_train)
        return d_w, d_b

    def update(self, d_w, d_b):
        self.weights = self.weights - self.learning_rate * d_w
        self.bias = self.bias - self.learning_rate * d_b

    def fit(self, X_train, Y_train, X_test, Y_test, epochs):
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        for epoch in tqdm(range(epochs)):
            for x_train, y_train in zip(X_train, Y_train):
                y_pred = self.forward(x_train)
                d_w, d_b = self.back_propagation(x_train, y_train, y_pred)
                self.update(d_w, d_b)
            train_loss, train_accuracy = self.evaluate(X_train, Y_train)
            test_loss, test_accuracy = self.evaluate(X_test, Y_test)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        return train_losses, train_accuracies, test_losses, test_accuracies

    def predict(self, X_test):
        Y_pred = []
        for x_test in X_test:
            y_pred = self.forward(x_test)
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

    def calculate_accuracy(self, X_test, Y_test, metric='r2'):
        Y_pred = self.predict(X_test)
        if metric == 'r2':
            RSS = np.sum((Y_test - Y_pred) ** 2)
            TSS = np.sum((Y_test - np.mean(Y_test)) ** 2)
            accuracy = 1 - RSS / TSS
        else:
            raise Exception("Not supported accuracy function")
        return accuracy

    def evaluate(self, X_test, Y_test):
        loss = self.calculate_loss(X_test, Y_test)
        accuracy = self.calculate_accuracy(X_test, Y_test)
        return loss, accuracy
