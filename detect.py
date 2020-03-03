import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf

from model import yolo_model, transform_img


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    print(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        print(x1y1)
        print(x2y2)
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    class_names = [c.strip() for c in open("data/classes.txt").readlines()]

    model = yolo_model(n_classes=len(class_names))

    model.load_weights('checkpoints/yolov3_train_14.tf').expect_partial()
    logging.info('weights loaded')

    img_raw = tf.image.decode_image(open("data/images/2B9 VASILIOK/15.maxresdefault.jpg", 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_img(img, 416)

    t1 = time.time()
    boxes, scores, classes, nums = model(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite('./output.jpg', img)
    logging.info('output saved to: {}'.format('./output.jpg'))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass