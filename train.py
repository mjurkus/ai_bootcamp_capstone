from code import load_dataset, freeze_all
from model import transfrom_img, transform_targets, yolo_model, create_yolo_loss
import numpy as np
import tensorflow as tf

anchors = (
    np.array(
        [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)], np.float32
    )
    / 416
)
anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

IMG_SIZE = 416
BATCH_SIZE = 6

class_names = [c.strip() for c in open("data/classes.txt").readlines()]

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_ds = load_dataset("data/images/military_train.tfrecord", "data/classes.txt", IMG_SIZE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(
    lambda x, y: (
        transfrom_img(x, IMG_SIZE),
        transform_targets(y, anchors, anchor_masks, IMG_SIZE),
    )
)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


val_ds = load_dataset("data/images/military_val.tfrecord", "data/classes.txt", IMG_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.map(
    lambda x, y: (
        transfrom_img(x, IMG_SIZE),
        transform_targets(y, anchors, anchor_masks, IMG_SIZE),
    )
)

pretrained = yolo_model(
    size=IMG_SIZE, anchors=anchors, masks=anchor_masks, n_classes=80, training=True
)
pretrained.load_weights("checkpoints/yolov3-tiny.tf")

model = yolo_model(
    size=IMG_SIZE,
    anchors=anchors,
    masks=anchor_masks,
    n_classes=len(class_names),
    training=True,
)

model.get_layer("yolo_darknet").set_weights(
    pretrained.get_layer("yolo_darknet").get_weights()
)
freeze_all(model.get_layer("yolo_darknet"))

optimizer = tf.keras.optimizers.Adam(lr=0.001)
loss = [
    create_yolo_loss(anchors[mask], n_classes=len(class_names)) for mask in anchor_masks
]
model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=3, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/yolov3_train_{epoch}.tf", verbose=1, save_weights_only=True
    ),
]

history = model.fit(
    train_ds, epochs=2, callbacks=callbacks, validation_data=val_ds, verbose=1
)