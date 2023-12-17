import numpy as np
import random
from tqdm import tqdm


class Perceptron:
    def __init__(self, learning_rate_w, learning_rate_b, epochs):
        self.weights = None
        self.bias = None
        self.learning_rate_w = learning_rate_w
        self.learning_rate_b = learning_rate_b
        self.epochs = epochs

    def fit(self, x_train, y_train):

        self.weights = random.random()*5
        self.bias = random.random()*5
        self.losses = []
        self.w = []
        self.b = []
        for _ in tqdm(range(self.epochs)):
            for i in range(x_train.shape[0]):
                x = x_train[i]
                y = y_train[i]

                y_pred = x*self.weights+self.bias
                error = y - y_pred

                # SGD
                self.weights = self.weights+(error*x*self.learning_rate_w)
                self.bias = self.bias+(error*self.learning_rate_b)
                self.w.append(self.weights)
                self.b.append(self.bias)
                self.losses.append(self.evaluate(x_train, y_train))
        return self.weights, self.bias, self.w, self.b, self.losses

    def predict(self, X_test):
        y_pred = X_test*self.weights+self.bias
        return y_pred

    def evaluate(self, X_test, Y_test, metric="mae"):
        y_pred = X_test*self.weights+self.bias
        error = Y_test - y_pred
        if metric == 'mae':
            loss = np.sum(np.abs(error))/len(Y_test)
        elif metric == "mse":
            loss = np.mean(error**2)
        elif metric == "rmse":
            loss = np.sqrt(np.mean(error**2))
        return loss
