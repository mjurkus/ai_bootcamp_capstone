import numpy as np

yolo_anchors = (
        np.array(
            [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)], np.float32
        )
        / 416
)
yolo_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])