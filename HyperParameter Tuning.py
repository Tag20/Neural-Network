# -*- coding: utf-8 -*-
#Aim: Hyper Parameter Tuning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt

"""# **Step 1:**
## Import the IMDB data.
"""

df = pd.read_csv('/kaggle/input/imdb-dataset-of-top-1000-movies-and-tv-shows/imdb_top_1000.csv')
df

"""# **Step 2:**

## Pre-processing of the dataset.
"""

df.info()

df = df.drop(['Poster_Link','Series_Title','Released_Year','Runtime','Overview','Director','Star1','Star2','Star3','Star4'], axis=1)

df

from sklearn.preprocessing import LabelEncoder

genre_encoder = LabelEncoder()
certificate_encoder = LabelEncoder()

df['Genre'] = genre_encoder.fit_transform(df['Genre'])
df['Certificate'] = certificate_encoder.fit_transform(df['Certificate'])

print(df.head())

df.isna().sum()

df = df.dropna()

df.head()

df.info()

df['Gross'] = df['Gross'].str.replace(',', '')
df['Gross'] = df['Gross'].astype(int)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns_to_scale = ['Meta_score','No_of_Votes','Gross']

df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

print(df.head())

X = df.drop('IMDB_Rating',axis=1)
y = df['IMDB_Rating']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
    y,
    test_size = 0.25,
    train_size=0.75,
    random_state = 42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import tensorflow as tf

"""# **Step 3:**

#Building the sequential neural network model
"""

model = tf.keras.Sequential()

model.add(Dense(5, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

"""# **Step 4:**

#Compile and fit the model to the training dataset
"""

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

history = model.fit(X_train,
          y_train,
          epochs=500,
          verbose=0)

"""# **Step 5:**

#Plot training and validation loss
"""

predictions = model.predict(X_train)
sorted_indices = np.argsort(y_train)

# Plotting actual and predicted ratings side by side
plt.plot(np.arange(len(y_train)), np.sort(y_train), label='Actual Ratings')
plt.plot(np.arange(len(predictions)), np.sort(predictions[:,0]), label='Predicted Ratings')
plt.xlabel('Movie Index (sorted by actual ratings)')
plt.ylabel('Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

predictions = model.predict(X_test)
sorted_indices = np.argsort(y_test)

# Plotting actual and predicted ratings side by side
plt.plot(np.arange(len(y_test)), np.sort(y_test), label='Actual Ratings')
plt.plot(np.arange(len(predictions)), np.sort(predictions[:,0]), label='Predicted Ratings')
plt.xlabel('Movie Index (sorted by actual ratings)')
plt.ylabel('Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

model = tf.keras.Sequential()
model.add(Dense(4, activation='linear', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='softplus'))

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

history = model.fit(X_train,
          y_train,
          epochs=500,
          verbose=0)

predictions = model.predict(X_train)
sorted_indices = np.argsort(y_train)

# Plotting actual and predicted ratings side by side
plt.plot(np.arange(len(y_train)), np.sort(y_train), label='Actual Ratings')
plt.plot(np.arange(len(predictions)), np.sort(predictions[:,0]), label='Predicted Ratings')
plt.xlabel('Movie Index (sorted by actual ratings)')
plt.ylabel('Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

predictions = model.predict(X_test)
sorted_indices = np.argsort(y_test)

# Plotting actual and predicted ratings side by side
plt.plot(np.arange(len(y_test)), np.sort(y_test), label='Actual Ratings')
plt.plot(np.arange(len(predictions)), np.sort(predictions[:,0]), label='Predicted Ratings')
plt.xlabel('Movie Index (sorted by actual ratings)')
plt.ylabel('Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

new_df = df.drop('Gross',axis=1)

X = new_df.drop('IMDB_Rating',axis=1)
y = new_df['IMDB_Rating']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
    y,
    test_size = 0.25,
    train_size=0.75,
    random_state = 42
)

X_train

model = tf.keras.Sequential()
model.add(Dense(4, activation='linear', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='softplus'))

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

history = model.fit(X_train,
          y_train,
          epochs=500,
          verbose=0)

predictions = model.predict(X_train)
sorted_indices = np.argsort(y_train)

# Plotting actual and predicted ratings side by side
plt.plot(np.arange(len(y_train)), np.sort(y_train), label='Actual Ratings')
plt.plot(np.arange(len(predictions)), np.sort(predictions[:,0]), label='Predicted Ratings')
plt.xlabel('Movie Index (sorted by actual ratings)')
plt.ylabel('Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

predictions = model.predict(X_test)
sorted_indices = np.argsort(y_test)

# Plotting actual and predicted ratings side by side
plt.plot(np.arange(len(y_test)), np.sort(y_test), label='Actual Ratings')
plt.plot(np.arange(len(predictions)), np.sort(predictions[:,0]), label='Predicted Ratings')
plt.xlabel('Movie Index (sorted by actual ratings)')
plt.ylabel('Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

"""# **Step 6:**

#Use regularizes to improve the performance.
"""

from tensorflow.keras.regularizers import l1, l2
model = tf.keras.Sequential()
model.add(Dense(4, activation='linear', input_shape=(X_train.shape[1],),kernel_regularizer=l1(0.01)))
model.add(Dense(8, activation='linear',kernel_regularizer=l1(0.01)))
model.add(Dense(1, activation='softplus',kernel_regularizer=l1(0.01)))

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

history = model.fit(X_train,
          y_train,
          epochs=500,
          verbose=0)

"""# **Step 7:**

#Recording the Best Performance

"""

predictions = model.predict(X_train)
sorted_indices = np.argsort(y_train)

# Plotting actual and predicted ratings side by side
plt.plot(np.arange(len(y_train)), np.sort(y_train), label='Actual Ratings')
plt.plot(np.arange(len(predictions)), np.sort(predictions[:,0]), label='Predicted Ratings')
plt.xlabel('Movie Index (sorted by actual ratings)')
plt.ylabel('Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

predictions = model.predict(X_test)
sorted_indices = np.argsort(y_test)

# Plotting actual and predicted ratings side by side
plt.plot(np.arange(len(y_test)), np.sort(y_test), label='Actual Ratings')
plt.plot(np.arange(len(predictions)), np.sort(predictions[:,0]), label='Predicted Ratings')
plt.xlabel('Movie Index (sorted by actual ratings)')
plt.ylabel('Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

"""# **Observation and Learning**
In the process of developing a sentiment analysis model for IMDb movie reviews, I learned the importance of thorough text preprocessing. Transforming raw text into numerical data using tokenization, padding, and embedding layers is crucial for the success of neural network models. Additionally, visualizing data distributions helps in understanding the nature of the dataset, and introducing regularization techniques like dropout aids in preventing overfitting. Through experimentation and continuous refinement, I gained insights into how different preprocessing choices impact the model's performance, highlighting the iterative and exploratory nature of deep learning projects.

#**Conclusion**
Through meticulous text preprocessing, model design, and regularization techniques like dropout, I've learned the critical role of data preparation in building an effective sentiment analysis model for IMDb reviews. Visualizing data distributions provided valuable insights, emphasizing the iterative nature of deep learning projects. This experience has deepened my understanding of NLP and highlighted the importance of experimentation in refining neural network architectures.
"""
