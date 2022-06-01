import pandas as pd, pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python import layers
import numpy as np

def load_file():
    dfile = "Iris.csv"
    df = pd.read_csv(dfile)
    df.head()

    X = df.iloc[:,0:4].values
    Y = df.iloc[:,4].values
    # print(type(X))
    # print(type(Y))
    return X, Y

def encoder_transform(Y):
    encoder = LabelEncoder()
    y1 = encoder.fit_transform(Y)
    y = pd.get_dummies(y1).values

    return y

def train_test(X, y):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
        ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(Xtrain, Ytrain, batch_size=50, epochs=100)

    loss, accuracy = model.evaluate(Xtest, Ytest, verbose=0)
    # print("Test loss: ", loss)
    # print("Test accuracy: ", accuracy)

    y_pred = model.predict(Xtest)

    actual = np.argmax(Ytest, axis=1)
    predicted = np.argmax(y_pred, axis=1)

    # print(f"Actual: {actual}")
    # print(f"Predicted: {predicted}")

    return loss, accuracy, actual, predicted

if __name__ == "__main__":

    loaded_result = load_file()
    encoded_result = encoder_transform(loaded_result[1])
    train_test_result = train_test(loaded_result[0], encoded_result)
