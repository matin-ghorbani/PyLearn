from keras.models import Sequential, load_model
import numpy as np

model: Sequential = load_model('./Snake_weight_60ep.h5')

print(model.summary())
print('\n\n', model.predict(np.array([[
    256, 256, 256, 256, 1, 0, 0, 0, -144, -56
]])))
