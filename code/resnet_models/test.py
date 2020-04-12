import os, sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]

#export CUDA_VISIBLE_DEVICES=0
import matplotlib.pyplot as plt

import keras
import tensorflow 
from clr_callback import *
import numpy as np
np.random.seed(2016)
import scipy
import os
import glob
import math
import pickle
import datetime
#import pandas as pd
from keras.layers.noise import GaussianDropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,AveragePooling2D, Conv2D,MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import RMSprop,SGD
from keras.optimizers import Adam,Adagrad,Adadelta
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import RepeatVector
from keras.layers.normalization import BatchNormalization
#from keras.layers.core import  ActivityRegularization
#from keras.regularizers import WeightRegularizer
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import UpSampling2D
from keras.layers.noise import GaussianNoise
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM,GRU
#from keras.layers import Merge
#from keras.regularizers import l2
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
#from keras.applications.densenet import DenseNet121
#from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications import vgg16 as keras_vgg16
from keras.layers import BatchNormalization
from keras.models import Model
# dimensions of our images.
img_width, img_height = 224,224

#from keras.layers import Flatten, Dense, Input
#from keras_vggface.vggface import VGGFace

hidden_dim=512

train_data_dir = '/home/rahulp2/covid/all/train2'
validation_data_dir = '/home/rahulp2/covid/all/validation2'
test_data_dir = '/home/rahulp2/cov/new/test3'
nb_train_samples = 336
nb_validation_samples = 72
nb_test_samples = 20
nb_epoch = 100
img_rows, img_cols, img_channel = 224, 224, 3
input_tensor_shape=(img_rows, img_cols, img_channel)

#input_layer = Input(shape=(224, 224,3),
 #             name='image_input')
base_model = ResNet50(weights='imagenet', include_top=False, input_shape= input_tensor_shape)

for layer in base_model.layers:
        layer.trainable=False
    
x_model = base_model.output
x_model = Flatten()(x_model)
#x_model = GlobalAveragePooling2D(name='globalaveragepooling2d')(x_model)
    
#x_model = Dense(1024, activation='relu',name='fc1_Dense')(x_model)
x_model = Dropout(0.5, name='dropout_1')(x_model)
    
#x_model = Dense(256, activation='relu',name='fc2_Dense')(x_model)
#x_model = Dropout(0.5, name='dropout_2')(x_model)
predictions = Dense(1, activation='sigmoid',name='output_layer')(x_model)
    
model = Model(input=base_model.input, output=predictions)


print(model.summary())

model.load_weights("weights2-improvement-54-0.84.hdf5")


clr_triangular = CyclicLR(base_lr=0.0001, max_lr=0.001,
                        step_size=15.,mode='triangular')

rms =RMSprop( lr=0.0001)
 
#rms =Adagrad( lr=0.001,epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])

'''
filepath="weights2-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,clr_triangular]

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255)
'''
# this is the augmentation configuration we will use for testing:
# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)
#vali_datagen = ImageDataGenerator(rescale=1./255)

'''

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')


validation_generator = vali_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='binary')
'''

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='binary')

np.savetxt("name.txt",test_generator.filenames, delimiter=" ", fmt="%s")

'''
history= model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,callbacks=callbacks_list)

print(model.summary())
'''
scores = model.evaluate_generator(test_generator, 30)
print("Accuracy = ", scores[1])

intermediate_output = model.predict_generator(test_generator,30,nb_worker=1, pickle_safe=True )

np.savetxt("SP1.csv",intermediate_output, delimiter=",")
