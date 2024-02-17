# -*- coding: utf-8 -*-

!pip install tensorflow tensorflow-datasets

import os

import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

# Display a few sample images from the MNIST dataset
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.show()

"""# **Step 2:** Pre-processing and prepare the data for giving to the CNN.
# a. Encoding classes using one hot encoder
"""

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

"""# Step 3: Building the convolutional network model"""

model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model1.summary()

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
historymodel1 = model1.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=15, validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))

plt.plot(historymodel1.history['accuracy'], label='Train Accuracy (ReLU/Sigmoid)')
plt.plot(historymodel1.history['val_accuracy'], label='Test Accuracy (ReLU/Sigmoid)')

"""# Step 4: Vary the number of layers"""

model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model2.summary()

# Compile and train the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
historymodel2 = model2.fit(x_train.reshape(-1, 28, 28, 1), y_train, verbose = 0, epochs=10, validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))

plt.plot(historymodel2.history['accuracy'], label='Train Accuracy (ReLU/Tanh)')
plt.plot(historymodel2.history['val_accuracy'], label='Test Accuracy (ReLU/Tanh)')

"""# Step 5: Implement the architecture of LeNet 5"""

model_lenet5 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model_lenet5.summary()

model_lenet5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lenet5 = model_lenet5.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=15, validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))

plt.plot(model_lenet5.history['accuracy'], label='Train Accuracy (LeNet-5)')
plt.plot(model_lenet5.history['val_accuracy'], label='Test Accuracy (LeNet-5)')

"""# Step 6: Compare models and plot the results"""

# Plot accuracy for Step 3 models
plt.plot(historymodel1.history['accuracy'], label='Train Accuracy (ReLU/Sigmoid)')
plt.plot(historymodel1.history['val_accuracy'], label='Test Accuracy (ReLU/Sigmoid)')

# Plot accuracy for Step 4 models
plt.plot(historymodel2.history['accuracy'], label='Train Accuracy (ReLU/Tanh)')
plt.plot(historymodel2.history['val_accuracy'], label='Test Accuracy (ReLU/Tanh)')

# Plot accuracy for LeNet-5 model
plt.plot(model_lenet5.history['accuracy'], label='Train Accuracy (LeNet-5)')
plt.plot(model_lenet5.history['val_accuracy'], label='Test Accuracy (LeNet-5)')

plt.title('Model Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""# **Conclusion**
## Through this process, I discovered that tailoring the CNN architecture and paying attention to preprocessing steps are crucial for achieving optimal performance, showcasing the practical application of different layers in convolutional neural networks.

# **Observation and Learning:**
##  As I experimented with varying CNN architectures, I observed that the choice of layers significantly influences model performance. Additionally, hands-on exploration reinforced the critical role of preprocessing steps like one-hot encoding and feature normalization in optimizing model efficiency.
"""