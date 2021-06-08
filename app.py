#inmporting all the essential libraries
from flask import Flask
from flask_restful import Resource,Api
from flask_cors import CORS
import numpy as np      # the numpy library is for numpy arrays
import pandas as pd     # pandas is for preprossing the CSV files and converting into dataframes
import matplotlib.pyplot as plt  # matplotlib is for visualising the data
import seaborn as sns





app = Flask(__name__)
api=Api(app)




from sklearn.model_selection import train_test_split







class Predict(Resource):
    def get(self):
        return "hello!!!"

api.add_resource(Predict,'/predict')




if __name__ == '__main__':
    app.run(debug=True)
































