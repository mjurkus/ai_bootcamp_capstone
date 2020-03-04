from typing import List, Tuple

from flask import current_app as app
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2


class YoloService:
    IMAGE_DIMS = 416

    model: tf.keras.Model
    class_names: List[str]

    def __init__(self):
        app.logger.info("Initializing YoloService")
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # if len(physical_devices) > 0:
        #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

        model_path = "models/latest"

        if Path(model_path).exists():
            self.initialize_model(model_path)
        else:
            app.logger.info("No models exits")

    def initialize_model(self, model_path):
        """Initalizes model and class names from model_path"""
        self.model = self._load_model(model_path)
        infer = self.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        app.logger.info(infer.structured_outputs)

        self.class_names = self._load_class_names(model_path)
        app.logger.info(f"Model {model_path} successfully initialized")

    def predict(self, img):
        app.logger.info(f"Decoding {img}")
        img_raw = tf.image.decode_image(open(img, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = YoloService.transform_img(img, YoloService.IMAGE_DIMS)

        boxes, scores, classes, nums = self.model(img)

        app.logger.info("Detections:")
        for i in range(nums[0]):
            app.logger.info('\t{}, {}, {}'.format(self.class_names[int(classes[0][i])],
                                                  np.array(scores[0][i]),
                                                  np.array(boxes[0][i])))

        predictions = self._get_predictions(img_raw, (boxes, scores, classes, nums), self.class_names)

        results = []
        for prediction in predictions:
            x1y1, x2y2, class_name, accuracy = prediction

            results.append({
                "x1": str(x1y1[0]),
                "y1": str(x1y1[1]),
                "x2": str(x2y2[0]),
                "y2": str(x2y2[1]),
                "class_name": class_name,
                "accuracy": str(accuracy)
            })

        return results

    def _load_model(self, model_path) -> tf.keras.Model:
        return tf.saved_model.load(model_path)

    def _load_class_names(self, model_path):
        return [c.strip() for c in open(f"{model_path}/assets/classes.txt").readlines()]

    def _get_predictions(self, img, outputs, class_names) -> List[Tuple]:
        boxes, objectness, classes, nums = outputs
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
        wh = np.flip(img.shape[0:2])

        results = []

        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            class_name = class_names[int(classes[i])]
            accuracy = objectness[i]

            results.append((x1y1, x2y2, class_name, accuracy.numpy()))

        return results

    @staticmethod
    def transform_img(image, size):
        image = tf.image.resize(image, (size, size))
        return image / 255
