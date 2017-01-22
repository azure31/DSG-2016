"""
Get predictions from trained VGGNet
We apply data augmentation to the test images and take average prediction to get the final label
@author: Team Tabs
"""

from __future__ import print_function
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import keras
from keras.regularizers import l2, activity_l2
import h5py
from keras.layers.normalization import BatchNormalization


#train_dir=''
img_rows = 224			#Change to 256 or 192 as appropriate
img_cols = 224			#Change to 256 or 192 as appropriate
####################################################################################################
#Making VGG model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,img_rows,img_cols)))
model.add(Convolution2D(64, 3, 3, activation='relu', init = 'he_normal', name='conv1_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu', init = 'he_normal', name='conv1_2'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu', init = 'he_normal', name='conv2_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu', init = 'he_normal', name='conv2_2'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu', init = 'he_normal', name='conv3_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu', init = 'he_normal', name='conv3_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu', init = 'he_normal', name='conv3_3'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu', init = 'he_normal', name='conv4_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu', init = 'he_normal', name='conv4_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu', init = 'he_normal', name='conv4_3'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu', init = 'he_normal', name='conv5_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu', init = 'he_normal', name='conv5_2'))
model.add(ZeroPadding2D((1,1)))																#Change to (2,2) for image size 192*192
model.add(Convolution2D(512, 3, 3, activation='relu', init = 'he_normal', name='conv5_3'))	
model.add(MaxPooling2D((2,2), strides=(2,2)))												#Change window size to (3,3) for image-size 256*256	and to (1,1) for image-size 192*192

model.add(Flatten())
model.add(Dense(4096, activation='relu', init = 'he_normal'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', init = 'he_normal'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.load_weights('vgg16_conv3_33_submission_4_1_7.h5')
sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])




###################################################
#For Prediction------------------------------------
###################################################
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1, 
    height_shift_range=0.1,# randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True, zoom_range=0.2, channel_shift_range = 0.2)  # randoml

#Load Test Data
X_test = np.load('X_test_224.npy')
X_test = X_test.astype('float16')
mean_pixel = [103.939, 116.799, 123.68]
for c in range(3):
    X_test[:,c, :, :] = X_test[:,c, :, :] - mean_pixel[c]
	

y_test = np.zeros(shape = (X_test.shape[0]), dtype = 'uint8')

#Define a generator function to create random sets of test datasets
def get_aug_predictions(X,y):
	test_gen=datagen.flow(X, y, batch_size = 32, shuffle = False)
	pred=model.predict_generator(test_gen, val_samples = X.shape[0])
	return(pred)

#Make predictions on slight variants of test dataset created using the ImagedataGenerator module of keras 	
print("starting the first prediction")	
predictions1=get_aug_predictions(X_test, y_test)
print("done with first")
predictions2=get_aug_predictions(X_test, y_test)
print("done with second")
predictions3=get_aug_predictions(X_test, y_test)
print("done with third")

#make predictions on the original test dataset
pred_no_aug=model.predict(X_test)
print("done with forth")


#Create average probability ensemble of the four predictions obtained above 
prob_avg=np.mean([predictions1,predictions2,predictions3, pred_no_aug], axis=0)
pred_all_with_original=np.argmax(prob_avg, axis=1)+1

import pandas as pd
col1 = pd.DataFrame(pred_all_with_original)
col2=pd.DataFrame(prob_avg)
test_pred_df = pd.concat([col1,col2], axis = 1)
test_pred_df.to_csv('vgg_224_tdg_fulldata.csv')