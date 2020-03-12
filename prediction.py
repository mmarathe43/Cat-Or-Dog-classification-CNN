import cv2
import numpy as np 
from keras.models import load_model
import matplotlib.pyplot as plt

category=["Dog","Cat"]
img_size=50
def prepare(image):
    img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    new_array=np.array(new_array).reshape(-1,img_size,img_size,1)
    new_array=new_array/255.0
    return new_array

model= load_model("dogorcat-con3-d1.h5")

img_path="./tests/dog/2.jpg"
plt.imread(img_path)
plt.show()
print(category[int(round(model.predict([prepare(img_path)])[0][0]))])