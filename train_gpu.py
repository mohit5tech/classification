import os
import time
import shutil
import pathlib
import itertools

# Import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Import deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('All modules loaded')

##### Data preprocessing #####
data_dir2 = 'DATASET'
data_dir = 'chest_xray/train'

filepaths = []
labels = []

# Load data from directories
folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)
        
folds = ['Pleural', 'Cardiomegaly']
for fold in folds:
    foldpath = os.path.join(data_dir2, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

# Create dataframe
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis=1)

# Train, validation, test split
train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(dummy_df, train_size=0.6, shuffle=True, random_state=123)

# Image and batch size settings
batch_size = 64
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# Function for custom preprocessing
def scalar(img):
    return img

# Generators
tr_gen = ImageDataGenerator(preprocessing_function=scalar)
ts_gen = ImageDataGenerator(preprocessing_function=scalar)

train_gen = tr_gen.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
    color_mode='rgb', shuffle=True, batch_size=batch_size
)
valid_gen = ts_gen.flow_from_dataframe(
    valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
    color_mode='rgb', shuffle=True, batch_size=batch_size
)
test_gen = ts_gen.flow_from_dataframe(
    test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
    color_mode='rgb', shuffle=False, batch_size=batch_size
)

# GPU strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Model definition under strategy scope
with strategy.scope():
    class_count = len(train_gen.class_indices.keys())

    # Load ResNet50 base model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], channels))

    # Add custom layers
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(class_count, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Freeze base model
    base_model.trainable = False

    # Compile model
    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

# Training parameters
epochs = 15

# Train model
history = model.fit(
    x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=False
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_gen)
print("Test Accuracy:", test_accuracy)

# Save model
model.save('chest_x-raysResNet50-MultiGPU.h5')

# Generate classification report
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
classes = list(train_gen.class_indices.keys())
print(classification_report(test_gen.classes, y_pred, target_names=classes))
