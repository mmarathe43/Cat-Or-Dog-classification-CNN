import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

def main():

    DATADIR="C:/kagglecatsanddogs/PetImages"
    categories=["Dog","Cat"]

    training_data=[]
    img_size=50
    def create_training_data():
        for category in categories:
            images=os.path.join(DATADIR,category)
            for img in os.listdir(images):
                try:
                    img_array=cv2.imread(os.path.join(images,img),cv2.IMREAD_GRAYSCALE)
                    new_array=cv2.resize(img_array,(img_size,img_size))
                    catordog = categories.index(category)
                    training_data.append([new_array,catordog])
                except Exception as e:
                    pass


    create_training_data()

    random.shuffle(training_data)

    X=[]
    y=[]

    for a,b in training_data:
        X.append(a)
        y.append(b)

    X=np.array(X).reshape(-1,img_size,img_size,1)
    # X=X/255.0
    f=open("X_pickle.pkl","wb")
    pickle.dump(X,f)
    f.close()

    w=open("y_pickle.pkl","wb")
    pickle.dump(y,w)
    w.close()

def get_data():
    m=open("X_pickle.pkl","rb")
    X_new=pickle.load(m)
    m.close()

    n=open("y_pickle.pkl","rb")
    y_new=pickle.load(n)
    n.close()
    X_new=X_new/255.0
    return (X_new,y_new)

if __name__=="__main__":
    main()