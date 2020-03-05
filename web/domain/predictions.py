from flask_table import Table, Col


class PredictionsTable(Table):
    classes = ["table", "table-striped"]
    name = Col("Class name")
    accuracy = Col("Accuracy")


def get_predictions_table(predictions):
    preds = []
    print(predictions)
    for pred in predictions:
        preds.append(dict(name=pred[0], accuracy=pred[1]))

    return PredictionsTable(preds)
