from flask import Blueprint
from flask_restful import Api
from resources.prediction import Prediction
from resources.model import Model

api_bp = Blueprint("api", __name__)
api = Api(api_bp)

api.add_resource(Prediction, "/predict")
api.add_resource(Model, "/model")
