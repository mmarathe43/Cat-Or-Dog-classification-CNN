import data_preprocess
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from keras.utils import normalize
import numpy as np
import matplotlib.pyplot as plt
import cv2


X,y=data_preprocess.get_data()

model=Sequential()


model.add( Conv2D(64, (3,3),input_shape=X.shape[1:]) )
model.add( Activation("relu") )
model.add( MaxPooling2D(pool_size=(2,2)))
#24x24x64

model.add( Conv2D(64, (3,3)))
model.add( Activation("relu") )
model.add( MaxPooling2D(pool_size=(2,2)))

model.add( Conv2D(64, (3,3)))
model.add( Activation("relu") )
model.add( MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())          
# model.add(Dense(60))
# model.add( Activation("relu") )


model.add(Dense(1))
model.add(Activation("sigmoid"))
          
model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])
          
model.fit(X,y,batch_size=32,epochs=6,validation_split=0.1)


model.save("dogorcat-con3-d1.h5")


