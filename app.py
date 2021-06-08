#inmporting all the essential libraries
from flask import Flask
from flask_restful import Resource,Api





app = Flask(__name__)
api=Api(app)










class Predict(Resource):
    def get(self):
        return "hello!!!"

api.add_resource(Predict,'/predict')




if __name__ == '__main__':
    app.run(debug=True)
































