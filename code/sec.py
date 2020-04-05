import os, sys



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

train_data_dir = '/home/rahulp2/covid/fold1/train2'

# you can put your images in a test folder: for us we have two folder: covid and pnuemonia. If the prediction of our model is >=0.5 then its a pnuemonia (other than covid), and if its <0.5 then its a covid case.
test_data_dir = '/Users/rahulpaul/Desktop/covid_dataset/X-ray/final_test/test'

nb_test_samples = 20
img_rows, img_cols, img_channel = 224, 224, 3
input_tensor_shape=(img_rows, img_cols, img_channel)

#input_layer = Input(shape=(224, 224,3),
 #             name='image_input')
base_model = keras_vgg16.VGG16(input_shape=(224,224,3),weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
#x= Conv2D(512, (3, 3), activation='relu', padding='same',)(x)
#x= BatchNormalization()(x)
#x= MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
#x= Conv2D(512,(1,1),activation='relu',padding='same')(x)
#x= BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
#x = ActivityRegularization(l2=0.01)(x)
#x=Dropout(0.2)(x)
#x = Flatten(name='flatten')(x)
x = Dense(64, activation='relu', name='fc6')(x)
x= Dropout(0.5, name='dropout_1')(x)
#x = Dense(512, activation='relu', name='fc7')(x)
#x=Dropout(0.5, name='dropout_2')(x)

predictions = Dense(1, activation='sigmoid', name='predictions')(x)


#input_layer = Input(shape=(224, 224,3),
 #             name='image_input')
#base_model = DenseNet121(input_shape=(224,224,3), weights='imagenet', include_top=False)

#for layer in base_model.layers:
#        layer.trainable=False
    
#x_model = base_model.output
#x_model = Flatten()(x_model)    
#x_model = Dense(128, activation='relu',name='fc1_Dense')(x_model)
#x_model = Dropout(0.5, name='dropout_1')(x_model)
    
#predictions = Dense(1, activation='sigmoid',name='output_layer')(x_model)
    
model = Model(input=base_model.input, output=predictions)

#vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
#last_layer = vgg_model.get_layer('pool5').output
#x = Flatten(name='flatten')(last_layer)
#x = Dense(hidden_dim, activation='relu', name='fc6')(x)
#x= Dropout(0.5, name='dropout_1')(x)
#x = Dense(hidden_dim, activation='relu', name='fc7')(x)
#x=Dropout(0.5, name='dropout_2')(x)
#out = Dense(1, activation='sigmoid', name='fc8')(x)
#model = Model(vgg_model.input, out


print(model.summary())

model.load_weights("weights2-improvement-15-0.98.hdf5")


clr_triangular = CyclicLR(base_lr=0.0001, max_lr=0.001,
                        step_size=15.,mode='triangular')

rms =RMSprop( lr=0.0001)
 
#rms =Adagrad( lr=0.001,epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])

# if you want to train then you have to uncomment the following section

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

np.savetxt("name2.txt",test_generator.filenames, delimiter=" ", fmt="%s")

'''
history= model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,callbacks=callbacks_list)

print(model.summary())
'''
scores = model.evaluate_generator(test_generator, 20)
print("Accuracy = ", scores[1])

#predicted values will be saved in a csv file.
intermediate_output = model.predict_generator(test_generator,20,nb_worker=1, pickle_safe=True )

np.savetxt("C5.csv",intermediate_output, delimiter=",")


