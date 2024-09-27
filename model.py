import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

def create_data_generator(directory, batch_size=32, target_size=(24, 24)):
    data_gen = ImageDataGenerator(rescale=1./255)
    return data_gen.flow_from_directory(directory, batch_size=batch_size, target_size=target_size, class_mode='categorical', color_mode='grayscale')

BATCH_SIZE = 32
TARGET_SIZE = (24, 24)
train_data = create_data_generator('data/train', batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
valid_data = create_data_generator('data/valid', batch_size=BATCH_SIZE, target_size=TARGET_SIZE)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
STEPS_PER_EPOCH = len(train_data.classes) // BATCH_SIZE
VALIDATION_STEPS = len(valid_data.classes) // BATCH_SIZE
print(f"Steps per epoch: {STEPS_PER_EPOCH}, Validation steps: {VALIDATION_STEPS}")

model.fit(train_data, validation_data=valid_data, epochs=15, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS)
model.save('drowsiness_model.h5', overwrite=True)