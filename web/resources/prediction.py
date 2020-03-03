from flask_restful import Resource
from flask_restful import reqparse
from flask import current_app as app
import werkzeug


class Prediction(Resource):

    def post(self):
        """Get prediction for a photo"""

        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str)
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')

        args = parser.parse_args()

        temp_path = 'temp/temp.jpg'

        args['file'].save(temp_path)

        return app.yolo_service.predict(temp_path), 200
