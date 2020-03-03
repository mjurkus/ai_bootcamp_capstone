from flask import Flask, render_template
from flask import request, jsonify
from services.yolo_service import YoloService
import flask
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app = Flask(__name__)


def create_app(config_filename):
    app.config.from_object(config_filename)

    from app import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    app.config['MAX_CONTENT_LENGHT'] = 16 * 1024 * 1024
    app.app_context().push()
    app.yolo_service = YoloService()

    return app


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app = create_app('config')
    app.run(debug=True, host="0.0.0.0", port=8888)
