from toai.imports import *
from matplotlib.patches import Rectangle

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def show_predictions_iou(model, dataset, img_dims, cat_encoder):
    cols = 4
    rows = 2
    figsize = (5 * cols, 5 * rows)

    _, ax = plt.subplots(rows, cols, figsize=figsize)

    for x, y in dataset.take(1):
        for i in range(8):
            idx = (i // cols, i % cols) if rows > 1 else i % cols
            ax[idx].imshow(x[i].numpy())

            p_categories, p_bounding_boxes = model.predict(x)
            p_category = p_categories[i].argmax()
            p_bounding_box = p_bounding_boxes[i]
            ax[idx].text(
                (p_bounding_box[0] + p_bounding_box[2]) * img_dims[0] - 25,
                p_bounding_box[1] * img_dims[1],
                cat_encoder.inverse_transform([p_category])[0],
                color="white",
                backgroundcolor="blue",
                fontsize=16,
            )
            p_rect = make_rectangle(*p_bounding_box, img_dims, "b", 3)
            ax[idx].add_patch(p_rect)

            y_categories, y_bounding_boxes = y
            y_category = y_categories[i].numpy()
            y_bounding_box = y_bounding_boxes[i].numpy()
            ax[idx].text(
                y_bounding_box[0] * img_dims[0] + 5,
                y_bounding_box[1] * img_dims[1],
                cat_encoder.inverse_transform([y_category])[0],
                backgroundcolor="red",
                fontsize=16,
            )
            y_rect = make_rectangle(*y_bounding_box, img_dims, "r")
            ax[idx].add_patch(y_rect)
            
            iou_score = bb_intersection_over_union(
                y_bounding_boxes[i], p_bounding_boxes[i]
            )

            ax[idx].set_title(f"IOU: {iou_score:.5f}")
            

def make_report(model, dataset, cat_encoder):
    print(
        classification_report(
            [category.numpy() for _, (category, _) in dataset.unbatch()],
            model.predict(dataset)[0].argmax(axis=1),
            target_names=cat_encoder.classes_,
        )
    )

def make_rectangle(x0, y0, width, height, img_dims, color, linewidth=2):
    return Rectangle(
        (x0 * img_dims[1], y0 * img_dims[0]),
        width * img_dims[1],
        height * img_dims[0],
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )

def show_prediction(model, dataset, img_dims, n_image, cat_encoder):
    for x, y in dataset.take(1):
        fig, ax = plt.subplots(1)
        ax.imshow(x[n_image].numpy())

        p_categories, p_bounding_boxes = model.predict(x)
        p_category = p_categories[n_image].argmax()
        p_bounding_box = p_bounding_boxes[n_image]
        plt.text(
            (p_bounding_box[0] + p_bounding_box[2]) * img_dims[0] - 25,
            p_bounding_box[1] * img_dims[1],
            cat_encoder.inverse_transform([p_category])[0],
            color="white",
            backgroundcolor="blue",
            fontsize=16,
        )
        p_rect = make_rectangle(*p_bounding_box, img_dims, "b", 3)
        ax.add_patch(p_rect)

        y_categories, y_bounding_boxes = y
        y_category = y_categories[n_image].numpy()
        y_bounding_box = y_bounding_boxes[n_image].numpy()
        plt.text(
            y_bounding_box[0] * img_dims[0] + 5,
            y_bounding_box[1] * img_dims[1],
            cat_encoder.inverse_transform([y_category])[0],
            backgroundcolor="red",
            fontsize=16,
        )
        y_rect = make_rectangle(*y_bounding_box, img_dims, "r")
        ax.add_patch(y_rect)

        plt.show()
        
def show_history(history, contains: str, skip: int = 0) -> None:
    history_df = pd.DataFrame(history.history)
    history_df[list(history_df.filter(regex=contains))].iloc[skip:].plot()