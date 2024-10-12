import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumorModel.keras')

image=cv2.imread('C:\\Users\\VIVEK\\Downloads\\BrainTumor Classification DL\\datasets\\pred\\pred6.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img)

print(input_img)
print(result)




