
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import cv2

# In[2]:

columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
data = pd.read_csv('./data/driving_log.csv', names=columns)

print("Data loaded...")




# In[4]:

import numpy as np
import csv
import cv2
import sys

#loading images
car_images = []
steering_angles = []
path = './data/IMG/'
csv_file = './data/driving_log.csv'
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        steering_center = float(row[3])

        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        img_center = path+row[0].split('/')[-1]
        img_left   = path+row[1].split('/')[-1]
        img_right  = path+row[2].split('/')[-1]

        car_images = car_images + [img_center, img_left, img_right]
        steering_angles = steering_angles + [steering_center, steering_left, steering_right] 

X_train = car_images
y_train = np.array(steering_angles)
samples = list(zip(X_train, y_train))

#train test split
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[5]:

import matplotlib.image as mpimg



# flip images horizontally
def horizontal_flip(img, steering_angle):
    flipped_image = cv2.flip(img, 1)
    steering_angle = -1 * steering_angle
    return flipped_image, steering_angle


# In[8]:

#horizontally shift image by some pixels
WIDTH_SHIFT_RANGE = 100
def horiz_shift(img, steering_angle):
    rows, cols, channels = img.shape
    
    # Translation
    tx = WIDTH_SHIFT_RANGE * np.random.uniform() - WIDTH_SHIFT_RANGE / 2
    ty = 0
    steering_angle = steering_angle + tx / WIDTH_SHIFT_RANGE * 2 * .2
    
    transform_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])
    
    translated_image = cv2.warpAffine(img, transform_matrix, (cols, rows))
    return translated_image, steering_angle


#shift the brightness of the images
def brightness_shift(img, bright_value=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if bright_value:
        img[:,:,2] += bright_value
    else:
        random_bright = .25 + np.random.uniform()
        img[:,:,2] = img[:,:,2] * random_bright
    
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img



def randomly_apply_transformation(img, angle):
    i = np.random.randint(3)
    if (i == 0):
        a_img = brightness_shift(img)
        a_angle = angle
    elif (i == 1):
        a_img, a_angle = horiz_shift(img,angle)
    elif (i == 2):
        a_img, a_angle = horizontal_flip(img,angle)
    return a_img, a_angle


#training set generator
# In[11]:

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                image = mpimg.imread(batch_sample[0])
                angle = batch_sample[1]
            
                keep = 0
                while keep == 0:
                    x, y = randomly_apply_transformation(image, angle)
                    if abs(y) < .1:
                        val = np.random.uniform()
                        # 1 is threshold here (val>threshold)
                        if val > 0.4: 
                            keep = 1
                    else:
                        keep = 1
                
                
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


#validation set generator

## Image generator
import numpy as np
import cv2
import sklearn
import random
import matplotlib.image as mpimg

def generator_default(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = mpimg.imread(batch_sample[0])
                angle = batch_sample[1]
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#train_generator = generator(train_samples, batch_size=32)
train_generator = generator(train_samples, batch_size=64)
#validation_generator = generator(validation_samples, batch_size=32)
validation_generator = generator_default(validation_samples, batch_size=64)

print((next(train_generator)[0]).shape)


# In[ ]:
#model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Convolution2D, MaxPooling2D, Cropping2D, SpatialDropout2D

import tensorflow as tf

def resize_inpt(img):
    import tensorflow as tf
    #return tf.image.resize_images(img, (120, 200))
    return tf.image.resize_images(img, (132, 200))


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Lambda(resize_inpt, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((33,17),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(SpatialDropout2D(0.5))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(SpatialDropout2D(0.4))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(SpatialDropout2D(0.3))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(SpatialDropout2D(0.1))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dropout(0.2))
model.add(Dense(50))
#model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
number_of_epochs=12
samples_per_epoch=24000
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=number_of_epochs)
model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=number_of_epochs)


model.save('model_new.h5')


# In[ ]:
