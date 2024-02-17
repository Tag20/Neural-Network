# -*- coding: utf-8 -*-

#Aim: Transfer Learning
from keras.applications import VGG16

# Loading the VGG16 model
vgg_model = VGG16()

# Explore the model's summary to see its layers and parameters
vgg_model.summary()

"""# **Step 2:** Using the pre-trained model for prediction"""

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('/content/DOGGO2.jpg')

plt.imshow(img)

img_path = '/content/DOGGO2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict the class of the image using VGG16
predictions = vgg_model.predict(x)
decoded_predictions = decode_predictions(predictions, top=3)[0]

print(decoded_predictions)

"""# **Step 3:**Using the pre-trained model as a feature extractor"""

from keras.models import Model
feature_extractor_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_pool').output)

# Use the model as a feature extractor
features = feature_extractor_model.predict(x)

feature_extractor_model.summary()

"""# **Step 4:**Add new layers to the model and summarize the parameters"""

from keras.layers import Dense, Flatten
from keras.models import Sequential

# Add new layers to the VGG16 model
new_model = Sequential()
new_model.add(vgg_model)
new_model.add(Flatten())
new_model.add(Dense(256, activation='relu'))
new_model.add(Dense(10, activation='sigmoid'))

# Summarize the parameters of the new model
new_model.summary()

"""# **Step 5:** Define the layers which are trainable"""

# Set the layers of the VGG16 model as non-trainable
for layer in vgg_model.layers:
    layer.trainable = False

# Define the layers which are trainable in the new model
for layer in new_model.layers:
    print(layer.trainable)

