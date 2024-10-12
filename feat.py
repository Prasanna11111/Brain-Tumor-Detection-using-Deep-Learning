from keras.models import load_model, Model
import numpy as np
x_data=[]
# Load the Sequential model from the h5 file
model = load_model('BrainTumorModel.keras')
print(model.summary())
for layer in model.layers:
    print(layer.name)