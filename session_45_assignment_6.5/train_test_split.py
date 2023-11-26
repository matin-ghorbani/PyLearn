import numpy as np
import pandas as pd


def train_test_split(x, y, test_size=.2):
    # Shuffle the indices
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    # Calculate the index to split the data
    split_index = int((1 - test_size) * len(x))

    # Split the data
    x_train, x_test = x[indices[:split_index]], x[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    data = pd.read_csv('HousePrice.csv')
    data['Parking'].replace([True, False], [1, 0], inplace=True)
    data['Warehouse'].replace([True, False], [1, 0], inplace=True)
    data['Elevator'].replace([True, False], [1, 0], inplace=True)
    data.drop(['Address'], inplace=True, axis=1)
    X = np.array(data[['Area']])
    Y = np.array(data[['Price']])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

    print(f'The shape of x_train is {x_train.shape}')
    print(f'The shape of x_test is {x_test.shape}')

    print(f'The shape of y_train is {y_train.shape}')
    print(f'The shape of y_test is {y_test.shape}')
