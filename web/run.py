import logging
from _datetime import datetime

from flask import Flask, render_template, redirect, url_for
from flask import request
from flask_bootstrap import Bootstrap

from domain.current_models import get_current_models, update_current_model, store_model
from domain.predictions import get_predictions_table
from services.yolo_service import YoloService

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


@app.route("/predict_photo", methods=["POST"])
def predict_photo():
    file = request.files['photo_file']
    file_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = f"temp/{file_name}.jpg"
    file.save(file_name)

    result = app.yolo_service.predict_and_save(file_name)

    return redirect(url_for('.show_prediction', q=result))


@app.route("/prediction")
def show_prediction():
    destination = request.args.get('q')
    prediction_photo = None
    detections = get_predictions_table([])

    print(destination)

    if destination is not None:
        prediction_photo = destination + '/photo.jpg'
        detections = get_predictions_table([line.strip().split("::") for line in
                                            open("static/predictions/" + destination + "/detections.txt").readlines()])

    return render_template("prediction.html", prediction_photo=prediction_photo, detections=detections)


@app.route('/activate_model')
def activate_model():
    model_id = request.args.get('id')
    update_current_model(model_id)
    return redirect("/")


if __name__ == '__main__':
    app = create_app('config')
    app.run(debug=True, host="0.0.0.0", port=8888)
