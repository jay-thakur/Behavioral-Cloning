import os
import csv
import cv2
import numpy as np

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) 
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size = 32
epochs = 6

def generator(samples, batch_size):
    num_samples = len(samples)
    print(num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]                
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])                
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Create the Sequential model
model = Sequential()

# Step 1 : Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))

# Step 2 : Set up cropping2D layer
model.add(Cropping2D(cropping=((50,20), (0,0))))

# Step 3 : Add 5 Convolutional Layer with max pooling layer & dropout layer
model.add(Conv2D(24, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(36, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(48, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.1))

# Step 4 : Add a flatten layer
model.add(Flatten())

# Step 5 : Add fully-connected layers with rely activation function
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit_generator(train_generator,samples_per_epoch=len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=epochs)
# model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
# validation_data=validation_generator, validation_steps=len(validation_samples), epochs=epochs, verbose=1)

model.fit_generator(train_generator, samples_per_epoch=int(len(train_samples)/batch_size), validation_data=validation_generator,         nb_val_samples=int(len(validation_samples)/batch_size), nb_epoch=epochs)

model.save('model.h5')