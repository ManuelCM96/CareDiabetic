import sys
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

#from tensorflow import keras
#from tensorflow.keras import layers
K.clear_session()

#data_entrenamiento='/train'
#data_validacion='/validation'
#parametros
epocas=4
altura,longitud=100,100
batch_size=32
pasos=500
pasos_validacion=200
filtrosConv1=32
filtrosConv2=64
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
tamano_pool=(2,2)
clases=3
lr=0.0005
#pre_Procesamiento de imagenes
entrenamiento_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
validacion_datagen=ImageDataGenerator(
    rescale=1./255
)
imagen_entrenamiento=entrenamiento_datagen.flow_from_directory(
    'train',
    target_size=(altura,longitud),
    batch_size=(batch_size),
    class_mode='categorical'
)
imagen_validacion=validacion_datagen.flow_from_directory(
    'validation',
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'
)
#Estructura Red CNN
cnn=Sequential()
cnn.add(Convolution2D(filtrosConv1,tamano_filtro1,padding='same',input_shape=(altura,longitud,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2,tamano_filtro2,padding='same',input_shape=(altura,longitud,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases,activation='softmax'))
#Compilar la Estructura de la Red
#opt=optimizers.Adam(learning_rate=0.01)
cnn.compile(loss='categorical_crossentropy',optimizer='adam')
#Entrenar Nuestra Red
cnn.fit(imagen_entrenamiento,epochs=epocas,validation_data=imagen_validacion, validation_steps=pasos_validacion)
#Guardar el modelo entrenado
target_dir='./modelo'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')


