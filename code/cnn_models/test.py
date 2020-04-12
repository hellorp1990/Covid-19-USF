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
from keras.layers import Merge
from keras.layers import merge
from keras.models import Model
# dimensions of our images.
img_width, img_height = 100,100

#from keras.layers import Flatten, Dense, Input
#from keras_vggface.vggface import VGGFace

hidden_dim=512

train_data_dir = '/home/rahulp2/covid/all/train2'
validation_data_dir = '/home/rahulp2/covid/all/validation2'
test_data_dir = '/home/rahulp2/cov/new/test3'
nb_train_samples = 336
nb_validation_samples = 72
nb_test_samples = 4191
nb_epoch = 100

nb_classes =2

input_layer = Reshape((100,100,3), input_shape=(img_width, img_height,3))
input_layer2 = Reshape((10,10,3), input_shape=(img_width, img_height,3))



def fusionNet(nb_classes, inputs=(100, 100,3)):
    input_img = Input(shape=inputs, name='RGB_input')
    conv1 = Conv2D(64, (5, 5), name="conv1")(input_img)
    leaky1= LeakyReLU(alpha=.01) (conv1)
    pool1 = MaxPooling2D(pool_size=(3, 3)) (leaky1)
    conv2 = Conv2D(64, (2, 2),  name="conv2") (pool1)
    leaky2= LeakyReLU(alpha=.01) (conv2)
    pool2= MaxPooling2D(pool_size=(3, 3)) (leaky2)
    drop1= Dropout(0.1)(pool2)
    #flat= Flatten() (drop1)
    #fc1= Dense(128, activation= "relu")(flat)
    #fc2= Dense(8, activation= "relu")(fc1)
    #drop2= Dropout(0.25)(fc2)
    leaky11= LeakyReLU(alpha=.01) (input_img)
    pool11 = MaxPooling2D(pool_size=(10, 10)) (leaky11) 
    drop11= Dropout(0.1)(pool11)
    merge1 = merge([drop11, drop1], mode='concat')
    conv4 = Conv2D(64, (2, 2), activation='relu', name="conv4") (merge1)
    pool41= MaxPooling2D(pool_size=(3, 3)) (conv4)
    flat= Flatten() (pool41)
    drop2= Dropout(0.25)(flat)
    fc1= Dense(1)(drop2)
    sig1 = Activation("sigmoid", name='sigmoid1')(fc1)

    return Model(inputs=[input_img], output=sig1)

model = fusionNet(nb_classes=2,inputs=(100, 100,3))


print(model.summary())

model.load_weights("weights2-improvement-135-0.82.hdf5")


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

np.savetxt("name3.txt",test_generator.filenames, delimiter=" ", fmt="%s")

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

np.savetxt("SP7.csv",intermediate_output, delimiter=",")


