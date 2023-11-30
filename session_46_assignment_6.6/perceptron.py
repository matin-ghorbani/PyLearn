import numpy as np


class Perceptron:
    def __init__(self, lr_w, lr_b, epochs):
        self.w = np.random.rand(1, 1)
        self.b = np.random.rand(1, 1)
        self.learning_rate_w = lr_w
        self.learning_rate_b = lr_b
        self.epochs = epochs

        self.w = np.random.rand(1, 1)
        self.b = np.random.rand(1, 1)
        self.weights = []
        self.biases = []
        self.losses = []

    def fit(self, X_train, Y_train):
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = Y_train[i]

                y_pred = x * self.w + self.b
                error = y - y_pred

                self.w = self.w + (error * x * self.learning_rate_w)
                self.b = self.b + (error * self.learning_rate_b)
                self.weights.append(self.w)
                self.biases.append(self.b)
                self.losses.append(self.evaluate(X_train, Y_train, 'mae'))
            print(f"Epoch {epoch + 1} done.")

    def predict(self, X_test):
        return X_test * self.w + self.b

    def evaluate(self, X_test, Y_test, metric: str):
        y_pred = X_test * self.w + self.b
        error = Y_test - y_pred
        loss = 0

        if metric == 'mae':
            loss = np.sum(np.abs(error)) / len(Y_test)
        elif metric == 'mse':
            loss = np.mean(error ** 2)
        elif metric == 'rmse':
            loss = np.sqrt(np.mean(error ** 2))

        return loss
