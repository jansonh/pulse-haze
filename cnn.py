import os
import shutil
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as image_utils
from keras.callbacks import EarlyStopping

model = Sequential()

model.add(Convolution2D(128, 3, 3, input_shape=(3, 64, 64), name='conv2_1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, name='conv2_2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, name='conv2_3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    '/path/to/train/images',
    target_size=(64, 64),
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    '/path/to/validation/images',
    target_size=(64, 64),
    class_mode='binary'
)

early_stopping = EarlyStopping(monitor='val_acc', patience=5, min_delta=1e-3)
model.fit_generator(
    train_generator,
    samples_per_epoch=32,
    nb_epoch=1000,
    validation_data=validation_generator,
    nb_val_samples=32,
    callbacks=[early_stopping]
)
model.save_weights('cnn_pulse.h5')
