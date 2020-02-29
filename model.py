import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import l2

from code import BatchNormalization
from code import yolo_boxes, broadcast_iou
from code import yolo_nms


def create_conv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = "same"
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = "valid"

    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        kernel_regularizer=l2(0.0005),
    )(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return x


def create_darknet(name):
    x = inputs = Input([None, None, 3])
    x = create_conv(x, 16, 3)
    x = MaxPool2D(2, 2, "same")(x)
    x = create_conv(x, 32, 3)
    x = MaxPool2D(2, 2, "same")(x)
    x = create_conv(x, 64, 3)
    x = MaxPool2D(2, 2, "same")(x)
    x = create_conv(x, 128, 3)
    x = MaxPool2D(2, 2, "same")(x)
    x = x_8 = create_conv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, "same")(x)
    x = create_conv(x, 512, 3)
    x = MaxPool2D(2, 1, "same")(x)
    x = create_conv(x, 1024, 3)

    return Model(inputs, (x_8, x), name=name)


def create_yolo_conv(filters, name):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = create_conv(x, filters=filters, size=1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = create_conv(x=x, filters=filters, size=1)

        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def create_yolo_out(filters, anchors, n_classes, name=None):
    def yolo_out(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = create_conv(x=x, filters=filters * 2, size=3)
        x = create_conv(
            x=x, filters=anchors * (n_classes + 5), size=1, batch_norm=False
        )
        x = Lambda(
            lambda x: tf.reshape(
                x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, n_classes + 5)
            )
        )(x)

        return Model(inputs, x, name=name)(x_in)

    return yolo_out


def yolo_model(size, anchors, masks, n_classes, training=False):
    x = inputs = Input([size, size, 3], name="input")

    x_8, x = create_darknet(name="yolo_darknet")(x)

    x = create_yolo_conv(256, name="yolo_conv_0")(x)
    out_0 = create_yolo_out(256, len(masks[0]), n_classes, name="yolo_out_0")(x)

    x = create_yolo_conv(128, name="yolo_conv_1")((x, x_8))
    out_1 = create_yolo_out(128, len(masks[1]), n_classes, name="yolo_out_1")(x)

    if training:
        return Model(inputs, (out_0, out_1), name="yolo3")

    boxes_0 = Lambda(
        lambda x: yolo_boxes(x, anchors[masks[0]], n_classes), name="yolo_boxes_0"
    )(out_0)
    boxes_1 = Lambda(
        lambda x: yolo_boxes(x, anchors[masks[1]], n_classes), name="yolo_boxes_1"
    )(out_1)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, n_classes), name="yolo_nms")(
        (boxes_0[:3], boxes_1[:3])
    )

    return Model(inputs, outputs, name="yolo")


def transfrom_img(image, size):
    image = tf.image.resize(image, (size, size))
    image = image / 255
    return image


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]
                )
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]
                )
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(
        box_wh[..., 1], anchors[..., 1]
    )
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def create_yolo_loss(anchors, n_classes, ignore_treshold=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, n_classes
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(
                broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))),
                axis=-1,
            ),
            (pred_box, true_box, obj_mask),
            tf.float32,
        )
        ignore_mask = tf.cast(best_iou < ignore_treshold, tf.float32)

        # 5. calculate all losses
        xy_loss = (
                obj_mask
                * box_loss_scale
                * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        )
        wh_loss = (
                obj_mask
                * box_loss_scale
                * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        )
        obj_loss = keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * keras.losses.sparse_categorical_crossentropy(
            true_class_idx, pred_class
        )

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss
