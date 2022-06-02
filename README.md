# iris-ml-fastapi-demo
A demo of fastapi and ML based on Iris dataset

# ml.py
Using the .csv, we make a model, compile, train and test, and fit using the given attributes
This model is done 3 layers deep
We save the encoder as file for later use in inverse_transform
And save the model in h5 format for prediction
ml.py is run only once to generate a usable model

# main.py
This is the fastapi interface with routes to homepage and predictions
The predictions page has an input of .json format which is filtered only to body
The values are extracted according to attribute and converted to numpy array
After loading the h5 model file, we predict our given data using it
The pickled encoder is loaded and inverse_transform is applied to return the names mapped
This file is executed through {server/localhost}/docs, using predictions path

# Thank you
abubakr-tb
