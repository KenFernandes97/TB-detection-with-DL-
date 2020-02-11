import keras
import numpy as np
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os

# Location of dataset values used in training
train_path = '/Users/kennethfernandes/Documents/Disso Documents/Lung Segmentation/CXR_png'
# Location of dataset values used in evaluating
test_path = '/Users/kennethfernandes/Documents/Disso Documents/Lung Segmentation/test'

for filename in os.listdir(train_path):
	print(filename)
for filename in os.listdir(test_path):
	print(filename)

# No. of images (in dataset) used in traning
no_train_samples = 800
# No. of images (in dataset) used in evaluation
no_test_samples = 96

# No. of images in fwd/bkwrd pass
train_batch_size = 10
test_batch_size = 10

# image dimensions inputted into network
image_size = 224

# # No. of steps required in an iteration
train_steps = np.ceil(no_train_samples / train_batch_size)
val_steps = np.ceil(no_test_samples / test_batch_size)

print(train_steps)
print(val_steps)

# Set up img generators
print("Re-processing Images")

train_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    train_path, #pointing to the path where the data is stored
    target_size=(image_size, image_size), #the target size of the image output
    batch_size=train_batch_size) #size of the batches

valid_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    test_path,
    target_size=(image_size, image_size),
    batch_size=test_batch_size)

test_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    test_path,
    target_size=(image_size, image_size),
    batch_size=test_batch_size,
    shuffle=False)
print("Complete")

# Create a MobibleNet model
mobile = keras.applications.mobilenet.MobileNet()

# Modify the model
# Choose the 2nd layer from the last
x = mobile.layers[-2].output

# Add a dropout and dense layer for predictions
x = Dropout(0.25)(x)
print(x)
predictions = Dense(3, activation='softmax')(x)

# Create a new model with the new outputs
print("Creating Model")
model = Model(inputs=mobile.input, outputs=predictions)
print("Complete")
# Prevent everything except the last 23 layers from being trained
for layer in model.layers[:-23]:
    layer.trainable = False

# Define Top2 and Top3 Accuracy
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# Compile the model
print("Compiling Model")
model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
print("Complete")
# Add weights to make the model more sensitive to TB
class_weights={
    0: 1.0,  # Normal
    1: 3.0,  # TB
}

filepath = "model.h5"

# Declare a checkpoint to save the best version of the model
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1,
                             save_best_only=True, mode='max')

# Reduce the learning rate as the learning stagnates
reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

# Fit the model
history = model.fit_generator(train_batches,
                              steps_per_epoch=train_steps,
                              class_weight=class_weights,
                              validation_data=valid_batches,
                              validation_steps=val_steps,
                              epochs=10,
                              verbose=1,
                              callbacks=callbacks_list)


