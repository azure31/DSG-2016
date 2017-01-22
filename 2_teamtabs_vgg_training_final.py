from __future__ import print_function
import os
import numpy as np
np.random.seed(2016)

###model_name
model_name='vgg1'


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


img_rows = 224
img_cols = 224
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
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu', init = 'he_normal', name='conv5_3'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu', init = 'he_normal'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', init = 'he_normal'))
model.add(Dropout(0.5))

###########################################################################################################
#Loading VGG weights
weights_path= 'vgg16_weights.h5'
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
####

f.close()

########################################################################################################
#Adding the output-dense layer of size 4
model.add(Dense(4, activation='softmax'))
print('Model loaded.')

####################################################################################################
#Freeze weights of VGG model (first 20 layers)
for layer in model.layers[:20]:
    layer.trainable=False

#Define object for image augmentation	
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1, 
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True, zoom_range=0.2)  # randomly flip images
################################################################################################
#Load data
X_train=np.load('X_train_224.npy')
X_valid=np.load('X_valid_224.npy')

y_train=np.load('y_train_150.npy')
y_valid=np.load('y_valid_150.npy')

mean_pixel = [103.939, 116.799, 123.68]
for c in range(3):
    X_train[:,c, :, :] = X_train[:,c, :, :] - mean_pixel[c]
    X_valid[:,c, :, :] = X_valid[:,c, :, :] - mean_pixel[c]


################################################	
#Compile and train the model
#First Phase of training
sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train,
                        batch_size=16, shuffle = True),
                        samples_per_epoch=6400,
                        validation_data=(X_valid, y_valid),
                        nb_epoch=4,
                        verbose = 1)

#Second Phase of training
sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train,
                        batch_size=16, shuffle = True),
                        samples_per_epoch=6400,
                        validation_data=(X_valid, y_valid),
                        nb_epoch=1,
                        verbose = 1)

#Third Phase of training
sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train,
                        batch_size=16, shuffle = True),
                        samples_per_epoch=6400,
                        validation_data=(X_valid, y_valid),
                        nb_epoch=7,
                        verbose = 1)

model.save_weights('vgg16_conv3_33_submission_4_1_7.h5')		



