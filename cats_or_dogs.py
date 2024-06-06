# Libs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras.api
import tensorflow as tf
import numpy as np
import seaborn as sns
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras._tf_keras.keras.preprocessing.image import load_img, ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix

# Training DB
training_gen = ImageDataGenerator(rescale=1. / 255,
                                  rotation_range=7,
                                  horizontal_flip=True,
                                  zoom_range=0.2)
training_db = training_gen.flow_from_directory('./dados/training_set',
                                               target_size=(64, 64),
                                               batch_size=4,
                                               class_mode='binary')

# Test DB
test_gen = ImageDataGenerator(rescale=1. /255)
test_db = test_gen.flow_from_directory('./dados/test_set',
                                        target_size=(64, 64),
                                        batch_size=4,
                                        class_mode='binary',
                                        shuffle=False)

# Convolucional Neural Network
cnn = Sequential()
cnn.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(units=40, activation='relu'))
cnn.add(Dense(units=40, activation='relu'))
cnn.add(Dense(units=40, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(training_db, epochs=5, validation_data=test_db)

# Check Outputs
outputs = np.argmax(cnn.predict(test_db), axis=1)
print(accuracy_score(outputs, test_db.classes))
cm = confusion_matrix(outputs, test_db.classes)
print(cm)
sns.heatmap(cm, annot=True)