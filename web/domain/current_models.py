from flask_table import Table, Col, LinkCol, BoolCol
import os
from os import listdir
from os.path import isdir
import shutil
from flask import current_app as app


class CurrentModelsTable(Table):
    classes = ["table", "table-striped"]
    name = Col("Model name")
    active = BoolCol('Active', yes_display='Yes', no_display='No')
    link = LinkCol('Activate', 'activate_model', url_kwargs=dict(id='id'), anchor_attrs={'class': 'myclass'})


def get_current_models():
    only_models = [f for f in listdir("models") if isdir(f"models/{f}")]

    models = []

    for model in only_models:
        models.append(dict(name=model, active=is_model_active(model), id=model))

    return CurrentModelsTable(models)


def store_model(model_file, model_id):
    model_save_path = os.path.join("models", f"{model_id}.zip")
    model_file.save(model_save_path)
    shutil.unpack_archive(model_save_path, extract_dir=f"models/{model_id}")
    update_current_model(model_id)
    app.yolo_service.activate_model(model_id)


def update_current_model(model_id):
    with open("models/.current", "w") as f:
        f.write(model_id)


def is_model_active(model_id):
    return open("models/.current").readline() == model_id
