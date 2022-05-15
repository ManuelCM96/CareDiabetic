import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
epocas=20
datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=15,
    zoom_range=[0.5,1.5],
    validation_split=0.5
)
data_gen_entrenamiento=datagen.flow_from_directory('train',target_size=(224,224),
                                                   batch_size=32,shuffle=True,subset='training')
data_gen_pruebas=datagen.flow_from_directory('train',target_size=(224,224),
                                                   batch_size=32,shuffle=True,subset='validation')
mobilenetv2=hub.KerasLayer(url,input_shape=(224,224,3))
mobilenetv2.trainable=False
modelo=tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(2,activation='softmax')
])
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
modelo.fit(
    data_gen_entrenamiento,epochs=epocas,batch_size=32,
    validation_data=data_gen_pruebas
)
target_dir='./modelo'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
modelo.save('modelo/modelo.h5')
del modelo
