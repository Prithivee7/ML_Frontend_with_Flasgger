# -*- coding: utf-8 -*-

from flask import Flask, request
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
import json
import sklearn

app = Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "This is an IRIS classifier"


@app.route('/predict', methods=["GET"])
def predict_flower_type():
    """This is a get method which takes in sepal length, sepal width, petal length and petal width as parameters
    and returns the type of the flower as response.
    ---
    parameters:
      - name: SepalLengthCm
        in: query
        type: number
        required: true
      - name: SepalWidthCm
        in: query
        type: number
        required: true
      - name: PetalLengthCm
        in: query
        type: number
        required: true
      - name: PetalWidthCm
        in: query
        type: number
        required: true
    responses:
        200:
            description: Success

    """
    sepal_length = request.args.get("SepalLengthCm")
    sepal_width = request.args.get("SepalWidthCm")
    petal_length = request.args.get("PetalLengthCm")
    petal_width = request.args.get("PetalWidthCm")
    prediction = classifier.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    return "The Flower is of type "+str(prediction[0])


@app.route('/predict_file', methods=["POST"])
def batch_prediction():
    """This function takes a file as input which has sepal length, sepal width, petal length and petal width
    and returns an array of predictions based on the number of data points
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: Success

    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)

    return str(list(prediction))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
