

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/ML_Project/Covid19-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""# Import"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import os
import tensorflow as tf
import cv2
import pickle

from google.colab import drive
drive.mount('/content/drive')

"""# Load Data using Tensorflow"""

Data=tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/ML_Project/Covid19-dataset/train",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=8,
    image_size=(600, 600),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

Test =tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/ML_Project/Covid19-dataset/test",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=8,
    image_size=(600, 600),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

"""## Sample Example of the Covid X-Ray Images"""

img = cv2.imread('/content/drive/MyDrive/ML_Project/Covid19-dataset/train/Covid/01.jpeg')
plt.imshow(img)

"""## Sample Data for the Normal X-ray"""

img = cv2.imread('/content/drive/MyDrive/ML_Project/Covid19-dataset/train/Normal/01.jpeg')
plt.imshow(img)

"""## Sample Data for the Pneumonia X-ray"""

img = cv2.imread('/content/drive/MyDrive/ML_Project/Covid19-dataset/train/Viral Pneumonia/01.jpeg')
plt.imshow(img)

"""# **Modeling**"""

IMG_SIZE = (600, 600)

"""## **Load the MobileNetV2 for the Feature Extraction**"""

IMG_SHAPE =IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False, weights='imagenet')

base_model.trainable = False

"""## Data Augmentation"""

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

"""## **Required Layers**"""

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer_1 = tf.keras.layers.Dense(128)
prediction_layer = tf.keras.layers.Dense(3)

"""## ***NEURAL NETWORK ARCHITECTURE***"""

inputs = tf.keras.Input(shape=(600, 600, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = prediction_layer_1(x)
outputs = prediction_layer(x)
outputs = tf.keras.activations.softmax(outputs)
model = tf.keras.Model(inputs, outputs)

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
,loss='sparse_categorical_crossentropy',
metrics=["accuracy"])

# model.fit(Data,epochs=80)
# model.save("/content/drive/MyDrive/DrawingWeights/class6.h5")

# model = tf.keras.models.load_model('/content/drive/MyDrive/ML_Project/covid.h5')

"""## **Testing Accuracy**"""

model.evaluate(Test)

"""# **Sample Predictions**"""

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

