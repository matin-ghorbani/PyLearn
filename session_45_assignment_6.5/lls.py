import numpy as np


class LinearLeastSquare:
    def __init__(self) -> None:
        self.w = None

    def fit(self, x_train, y_train) -> None:
        self.w = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

    def predict(self, x_test: list) -> list[float]:
        y_pred = []
        for x in x_test:
            y_pred.append(x @ self.w)
        return y_pred

    def evaluate(self, x_test, y_test, metric) -> float | None:
        y_pred = self.predict(x_test)
        error = y_test - y_pred

        if metric == 'mae':
            loss = np.sum(np.abs(error)) / len(y_test)
        elif metric == 'mse':
            loss = np.sum(error ** 2) / len(y_test)
        elif metric == "rmse":
            loss = np.sqrt(np.sum(error ** 2) / len(y_test))
        else:
            loss = None

        return loss

