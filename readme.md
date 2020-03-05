## Military vehicles detection with YOLOv3

#### The problem

Detect, localize and classify military vehicles from the given photo.

### Goals
- Create TF2 model to solve given problem
- Create Android application to take photos and get predictions
- ~~Convert said model to TFLite and integrate it into Android Application~~ 
- Create web application instead of TFLite conversion

#### Model and dataset

- Model is based on YOLOv3.  
- Used COCO weights for transfer learning.
- Gathered, localized and labeled data from the internet.
- Supports 13 vehicle classes (for now)


###  Results

- Model has hard time to detect and classify images for many different angles, pint jobs and other
- Need more data to achieve better results (this takes A LOT of time)
- WEB application to upload "latest model" and use it for prediction
- Android application to take images and get predictions via API



#### Resources:
- https://medium.com/analytics-vidhya/yolo-v3-theory-explained-33100f6d193
- https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe
- https://github.com/zzh8829/yolov3-tf2