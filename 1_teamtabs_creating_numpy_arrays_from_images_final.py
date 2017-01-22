"""
Create numpy arrays from roof images

i/p: roof images, id_train.csv, img_rows, img_cols, sample_submission4.csv
o/p: train, test, validation numpy arrays

@author: Teamtabs 
"""

import os
SET_PATH="G:\\PGDBA\\Kaggle\\DataScienceGame\\roof_images"
os.chdir(SET_PATH)

import numpy as np
import pandas as pd
np.random.seed(2016)
import glob
import cv2
from keras.utils import np_utils

#Define utility functions to display image 

def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
color_type = 3 #- rgb

def get_im_cv2(path, img_rows, img_cols, color_type=3):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path,1)
    # Resize image
    resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
    return resized

    
# Change img_rows and img_cols for  genrating numpy data of images of different sizes
img_rows = 224
img_cols = 224

#Create train and validation data
X_train = []
X_valid = []
X_train_id = []
X_valid_id = []
y_train = []
y_valid = []

f = open('id_train.csv', 'r')
line = f.readline()
k = 0
while (1):
    line = f.readline()
    if line == '':
        break
    arr = line.strip().split(',')
    path = os.path.join("G:\\PGDBA\\Kaggle\\DataScienceGame\\roof_images",arr[0] + '.jpg')
    img = get_im_cv2(path, img_rows, img_cols, color_type = color_type)
    
    if (k)%5==0:
        X_valid.append(img)
        X_valid_id.append(int(arr[0]))
        y_valid.append(int(arr[1])-1)
    else:
        X_train.append(img)
        X_train_id.append(int(arr[0]))
        y_train.append(int(arr[1])-1)
    k = k+1
    if(k%500==0):print(k)

	
#Create test data

X_test = []
X_test_id = []
f = open('sample_submission4.csv', 'r')
line = f.readline()
k = 0
while (1):
    line = f.readline()
    if line == '':
        break
    arr = line.strip().split(',')
    path = os.path.join("G:\\PGDBA\\Kaggle\\DataScienceGame\\roof_images",arr[0] + '.jpg')
    img = get_im_cv2(path, img_rows, img_cols, color_type = color_type)
    X_test_id.append(int(arr[0]))
    X_test.append(img)
    k = k+1
    if(k%500==0):print(k)

f.close()

X_test_train_id = np.append(X_train_id , X_test_id)
total_X_train = []
total_X_train_id = []

#Create unlabelled data 
k = 0
for file in glob.glob("*.jpg"):
    line = file
    if line == '':
        break
    arr = line.strip().split('.')
    if int(arr[0]) not in X_test_train_id:
        img = get_im_cv2(file, img_rows, img_cols)        
        total_X_train.append(img)
        total_X_train_id.append(int(arr[0]))
    
    k = k+1
    if(k%500==0):print(k)
    


X_unlabelled = np.array(total_X_train, dtype=np.uint8)
y_train = np.array(y_train)
X_valid = np.array(X_valid, dtype='float16')
y_valid = np.array(y_valid, dtype=np.uint8)
X_train=np.array(X_train, dtype='float16')
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_train = X_train.transpose((0, 3, 1, 2))
y_train = np_utils.to_categorical(y_train, 4)
y_valid = np_utils.to_categorical(y_valid, 4)
X_test = np.array(X_test , dtype = np.units8)

##Save as .npy files
np.save('X_test_224.npy', X_test)
np.save('X_valid_224.npy', X_valid)
np.save('y_train_224.npy', y_train)
np.save('y_valid_224.npy', y_valid)
np.save('X_unlabelled_224.npy' , X_unlabelled)
np.save('X_train_224.npy', X_train)
