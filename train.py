import tensorflow as tf

from absl import app, logging, flags
from globals import yolo_anchors, yolo_anchor_masks
from helpers import load_dataset, freeze_all
from model import transform_img, transform_targets, yolo_model, create_yolo_loss
from absl.flags import FLAGS
from DLHelper.image.TrainCycle import TrainCycle

flags.DEFINE_boolean('finetune', False, 'finetune model')
flags.DEFINE_boolean('latest_weights', True, 'Use latest weights from default checkpoints dir')
flags.DEFINE_string('weights', "checkpoints", 'weights path')

IMG_SIZE = 416
BATCH_SIZE = 6


def main(_argv):
    class_names = [c.strip() for c in open("data/classes.txt").readlines()]

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_ds = load_dataset("data/images/military_train.tfrecord", "data/classes.txt", IMG_SIZE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.map(
        lambda x, y: (
            transform_img(x, IMG_SIZE),
            transform_targets(y, yolo_anchors, yolo_anchor_masks, IMG_SIZE),
        )
    )
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_ds = load_dataset("data/images/military_val.tfrecord", "data/classes.txt", IMG_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.map(
        lambda x, y: (
            transform_img(x, IMG_SIZE),
            transform_targets(y, yolo_anchors, yolo_anchor_masks, IMG_SIZE),
        )
    )

    model = yolo_model(
        size=IMG_SIZE,
        n_classes=len(class_names),
        training=True,
    )

    if FLAGS.finetune:
        latest_weights = tf.train.latest_checkpoint("checkpoints")

        if FLAGS.latest_weights:
            if latest_weights is None:
                raise RuntimeError("Latest weights were None")
            else:
                model.load_weights(latest_weights)
                logging.info(f"Loaded weights from '{latest_weights}'")
        else:
            model.load_weights(FLAGS.weights)
            logging.info(f"Loaded weights from '{FLAGS.weights}'")
    else:
        pretrained = yolo_model(n_classes=80, training=True)
        pretrained.load_weights("yolo3_coco_ckp/yolov3.tf")

        model.get_layer("yolo_darknet").set_weights(
            pretrained.get_layer("yolo_darknet").get_weights()
        )

    freeze_all(model.get_layer("yolo_darknet"))

    optimizer = tf.keras.optimizers.Adam(lr=1e-2)
    loss = [
        create_yolo_loss(yolo_anchors[mask], n_classes=len(class_names)) for mask in yolo_anchor_masks
    ]
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1, factor=0.3, patience=2),
        tf.keras.callbacks.EarlyStopping(patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/yolo_train_{epoch}.tf", verbose=1, save_weights_only=True, save_best_only=True
        ),
        # TrainCycle(epochs=100, lr=(1e-5, 1e-2), batch_size=BATCH_SIZE, train_set_size=914),
        # tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.CSVLogger('logs/training.log', append=True),
    ]

    model.fit(
        train_ds, epochs=100, callbacks=callbacks, validation_data=val_ds, verbose=1
    )


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
