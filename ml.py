import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python import layers
import numpy as np

dfile = "Iris.csv"
df = pd.read_csv(dfile)
df.head()

X = df.iloc[:,0:4].values
Y = df.iloc[:,4].values

encoder = LabelEncoder()
y1 = encoder.fit_transform(Y)

pickle.dump(encoder, open('enc.pkl', 'wb'))

y = pd.get_dummies(y1).values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)
model = tf.keras.Sequential([
tf.keras.layers.Dense(10, activation='relu'),
tf.keras.layers.Dense(10, activation='relu'),
tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(Xtrain, Ytrain, batch_size=50, epochs=100)

model.save("mymodel.h5")
