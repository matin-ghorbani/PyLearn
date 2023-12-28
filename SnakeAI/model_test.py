from keras.models import Sequential, load_model
import numpy as np

# model: Sequential = load_model('./Snake_weight_60ep.h5')
model: Sequential = load_model('./Snake_weight_60ep_version2.h5')

print(model.summary())
y_hat = model.predict(np.array([[
    484, 328, 28, 184, 0, 0, 1, 0, 0, 20,
]]))
print('\n\n', y_hat)
print('\n\n', np.argmax(y_hat))
