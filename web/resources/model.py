import werkzeug
from flask_restful import Resource
from flask_restful import reqparse
from flask import current_app as app
from pathlib import Path
import shutil


class Model(Resource):

    def post(self):
        """
        Saves model archive
        :return:
        """
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str)
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')

        args = parser.parse_args()

        model = args['file']

        model_save_path = Path("models") / model.filename
        model.save(model_save_path)
        app.logger.info(f"Model saved to {model_save_path}")

        app.logger.info(f"Unpacking model")
        shutil.unpack_archive(str(model_save_path), extract_dir="models/latest")

        return '', 200
