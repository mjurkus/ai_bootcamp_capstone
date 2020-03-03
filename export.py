import shutil
import time
from absl import app, logging
import numpy as np
import tensorflow as tf

from model import yolo_model, transform_img


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    class_names = [c.strip() for c in open("data/classes.txt").readlines()]

    model = yolo_model(n_classes=len(class_names))

    model.load_weights('checkpoints/yolov3_train_14.tf').expect_partial()
    logging.info('weights loaded')

    model_save_path = 'model/latest'

    logging.info(f"saving model to {model_save_path}")
    tf.saved_model.save(model, model_save_path)
    shutil.copy2("data/classes.txt", model_save_path + "/assets")

    logging.info(f"Loading model from {model_save_path}")
    model = tf.saved_model.load(model_save_path)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)
    logging.info(f"Model loaded")

    img_raw = tf.image.decode_image(open("data/images/2B9 VASILIOK/15.maxresdefault.jpg", 'rb').read(), channels=3)

    logging.info(f"Performing sanity check")
    img = tf.expand_dims(img_raw, 0)
    img = transform_img(img, 416)

    t1 = time.time()
    boxes, scores, classes, nums = model(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('Detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    logging.info(f"Sanity check complete")

    logging.info(f"Archiving saved model to model/latest.zip")
    shutil.make_archive("model/latest", "zip", model_save_path)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass