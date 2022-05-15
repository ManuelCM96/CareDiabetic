import os
import sys
import random
import numpy as np
import cv2 as cv
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from time import sleep
import h5py
from keras.preprocessing.image import load_img,img_to_array
#cap=cv.VideoCapture(0)
respuesta2=0
longitud,altura=224,224
modelo='modelo/modelo.h5'
img=['Prueba.jpeg','Pie_bueno.jpeg','Pie_malo.jpg']
img2=random.choice(img)
#pesos_modelo='modelo/pesos.h5'
#custom_object={'crf_loss':crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
modelo=load_model(modelo,custom_objects={'KerasLayer':hub.KerasLayer})
modelo.summary()
def predict(file):
    x=load_img(file,target_size=(longitud,altura,3))
    x=img_to_array(x)
    x=np.expand_dims(x,axis=0)
    arreglo=modelo.predict(x)
    print(arreglo)
    resultado=arreglo[0]
    print(resultado)
    respuesta=np.argmax(resultado)
    respuesta2=respuesta
    if respuesta == 0:
        print('Dañado')
    elif respuesta == 1:
        print('Sano')
    return respuesta
#def captura():
    #leido,frame=cap.read()
    #if leido== True:
    #    cv.imwrite("foto.jpg",frame)
    #    print("Foto tomada correctamente")
    #else:
    #    print("Error al acceder a la camara")
#image=cv.imread(img2)
#image_resize=cv.resize(image,(300,200))
#window_name='Image'
#cv.imshow(window_name,image_resize) 
#predict('Pie_bueno.jpeg')

#cv.waitKey(0)
#cv.destroyAllWindows()