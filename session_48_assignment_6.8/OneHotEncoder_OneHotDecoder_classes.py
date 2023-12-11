from sklearn.preprocessing import OneHotEncoder

class OneHotEncoder:
    def __init__(self):
        self.categories = []

    def fit(self, data):
        unique_values = set(data)
        self.categories = list(unique_values)

    def transform(self, data):
        encoded_data = []
        for value in data:
            encoded_row = [1 if value == category else 0 for category in self.categories]
            encoded_data.append(encoded_row)
        return encoded_data

class OneHotDecoder:
    def __init__(self, categories):
        self.categories = categories

    def transform(self, encoded_data):
        decoded_data = [self.categories[row.index(1)] for row in encoded_data]
        return decoded_data


if __name__ == '__main__':
    data = ['Python', 'C++', 'C#', 'PHP']

    custom_encoder = OneHotEncoder()
    custom_encoder.fit(data)
    custom_encoded_data = custom_encoder.transform(data)

    sklearn_encoder = OneHotDecoder(sparse=False)
    sklearn_encoded_data = sklearn_encoder.fit_transform([[value] for value in data])

    print(f'Custom Encoder: {custom_encoded_data}')

    print(f'\nScikit-Learn Encoder:\n{sklearn_encoded_data}')
