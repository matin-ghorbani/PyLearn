import numpy as np

class LLS:
    def __init__(self) -> None:
        self.w = None
    
    def fit(self, x_train, y_train):
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.T, x_train)), x_train.T), y_train)
    
    def predict(self, x_test):
        y_pred = []
        for x in x_test:
            y_pred.append(np.matmul(x, self.w))
            # y_pred.append(x * self.w)
        return np.array(y_pred)
