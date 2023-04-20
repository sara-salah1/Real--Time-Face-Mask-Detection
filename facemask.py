# PYTHON LIBRARIES

import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime
from keras.preprocessing.image import ImageDataGenerator

# BUILD DCNN MODEL
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# CUSTOM COMPILE
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# DATA PREPROCESSING

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

#  TRAIN MODEL
model_saved = model.fit_generator(
    training_set,
    epochs=10,
    validation_data=test_set,

)

#  SAVE MODEL
model.save('mymodel.h5', model_saved)

# ( with mask , without mask) ----->  ( 0,1 )
#  0 means predict image with mask
#  1 means predict image without mask

#  TEST USING SPECIFIC IMAGE
mymodel = load_model('mymodel.h5')
test_image = image.load_img(r'D:\FaceMaskDetector\test\with_mask\1-with-mask.jpg',
                            target_size=(150, 150, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
print(mymodel.predict(test_image))

test_image = image.load_img(r'D:\FaceMaskDetector\test\without_mask\11.jpg',
                            target_size=(150, 150, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
print(mymodel.predict(test_image))