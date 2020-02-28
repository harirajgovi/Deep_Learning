# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:27:33 2019

@author: hari4
"""

#Initializing the CNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

classifier = Sequential()

#First Convolution layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))

#First Convolution layer Max Pooling
classifier.add(MaxPool2D(pool_size=(2, 2)))

#Second Convolution layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

#Second Convolution layer Max Pooling
classifier.add(MaxPool2D(pool_size=(2, 2)))

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Image Augmentation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   rescale=1./255)
test_datagen =ImageDataGenerator(rescale=1./255)

#Importing dataset
training_dataset = train_datagen.flow_from_directory(directory='dataset\\training_set', 
                                                     target_size=(64, 64), 
                                                     class_mode='binary', 
                                                     batch_size=32)
test_dataset = train_datagen.flow_from_directory(directory='dataset\\test_set', 
                                                 target_size=(64, 64), 
                                                 class_mode='binary', 
                                                 batch_size=32)

#Fitting the dataset to CNN
classifier.fit_generator(generator=training_dataset, 
                         steps_per_epoch=8000, 
                         epochs=25,
                         validation_data=test_dataset,
                         validation_steps=2000)

#Predicting the test image
import numpy as np
results = []

for i in range(1, 3):
    test_image = image.load_img(path='dataset\\single_prediction\\cat_or_dog_{num}.jpg'.format(num = i), target_size=(64, 64))
    test_image = image.img_to_array(img=test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        results.append('dog')
    else:
        results.append('cat')