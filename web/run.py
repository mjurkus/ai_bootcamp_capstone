from flask import Flask, render_template, redirect, url_for
from flask import request, jsonify
from services.yolo_service import YoloService
import logging
from flask_bootstrap import Bootstrap
from domain.current_models import get_current_models, update_current_model, store_model

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app = Flask(__name__)


def create_app(config_filename):
    app.config.from_object(config_filename)

    from app import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    app.app_context().push()
    app.yolo_service = YoloService()

    Bootstrap(app)

    return app


@app.route('/')
def home():
    return render_template("index.html", current_models=get_current_models())


@app.route("/upload_model", methods=["POST"])
def upload_model():
    file = request.files['model']
    model_id = request.form.get('model_id')

    store_model(file, model_id)

    return redirect("/")


@app.route('/activate_model')
def activate_model():
    model_id = request.args.get('id')
    update_current_model(model_id)
    return redirect("/")


if __name__ == '__main__':
    app = create_app('config')
    app.run(debug=True, host="0.0.0.0", port=8888)
