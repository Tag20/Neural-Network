# -*- coding: utf-8 -*-
"""NNDLExp3-BackPropogation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r9taF-PfLw00iutp5jwAQ8b5Vm0CKNXw
"""

#NNDL Experiment 3
#C115
#Tanish Vaidya
#Batch: EB2
#Aim: Implementation of Backpropagation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf

"""# **Step 1:**
## Load the IRIS dataset available on Kaggle in your notebooks and Reading it.
"""

df = pd.read_csv('/content/Iris.csv')

"""# **Step 2:**

## Pre-processing of the dataset.
"""

df = pd.get_dummies(df, columns=['Species'], prefix=['Species'])

"""**A:** Convert categorical values to numeric values using one hot encoder"""

print(df.columns)

df.head(5)

df.tail(5)

df.describe()

#a. Convert categorical values to numeric values using one hot encoder
#encoder = OneHotEncoder()
#species_encoded = encoder.fit_transform(df[['Species']]).toarray()

#encoder = OneHotEncoder()
#species_encoded = encoder.fit_transform(df[['Species']]).toarray()

df

df.nunique()

"""**B:** Remove the species column from the original dataset"""

df = df.drop('Species', axis=1)

"""**C:** Append the one hot encoded columns to the data frame"""

scaler = StandardScaler()
df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])

df

"""# **Step 3:**
## Building the three-layer feedforward neural network.

**A.**  Build the three-layer feedforward neural network, use sigmoid as the activation.
"""

X = df.drop(columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])  # Assuming these are the encoded species columns
y = df[['Species_Iris-setosa', 'Species_Iris-versicolor', 'Species_Iris-virginica']]

"""**B.** Split the data into training and testing sets"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""**C:** Build the three-layer feedforward neural network"""

input_shape = X_train.shape[1]

model = tf.keras.Sequential()

"""Input layer with 8 neurons (assuming 8 features after one-hot encoding)"""

from keras.layers import Dense
model.add(Dense(units=2, input_dim=4, activation='sigmoid', name='input_layer'))

"""Hidden layer with 2 neurons"""

model.add(Dense(units=2,activation='sigmoid', name='hidden_layer'))

"""Output layer with 3 neurons for one-hot encoding"""

model.add(Dense(units=3,activation='sigmoid', name='output_layer'))

"""# Display the model summary"""

model.summary()

"""**D.** Initialize the network with random weights and biases
The weights and biases are automatically initialized when you add layers in Keras

**E.** Use sigmoid as the activation function.
Sigmoid activation is already specified in the model

**F.** Use Mean Squared Error (MSE) as the loss function.
"""

#Compile the model
o=tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=o, loss='mean_squared_error', metrics=['accuracy'])

"""# **Step 4:**
## Train the neural network

Train the model with 5000 iterations
"""

history = model.fit(X_train,
          y_train,
          epochs=500,
          batch_size=5,
          validation_split=0.1,
          verbose=2)

"""## Plot the training and validation accuracy"""

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nTest Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')