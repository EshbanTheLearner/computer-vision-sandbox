import numpy as np
import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import time
from yolov3 import YOLOv3Net

devices = tf.config.experimental.list_physical_devices("GPU")
assert len(devices) > 0, "Not enough GPU devices available"
tf.config.experimental.set_memory_growth(devices[0], True)

model_size = (416, 416, 3)
num_classes = 80
class_name = "./data/coco.names"
max_output_size = 40
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfg_file = "cfg/yolov3.cfg"
weight_file = "weights/yolov3_weights.tf"
img_filename = "data/images/test.jpg"

def main():
    model = YOLOv3Net(cfg_file, model_size, num_classes)
    model.load_weights(weight_file)
    
    class_names = load_class_names(class_name)
    
    image = cv2.imread(img_filename)
    image = np.array(image)
    image = tf.expand_dims(image, 0)
    
    resized_frame = resize_image(image, (model_size[0], model_size[1]))
    pred = model.predict(resized_frame)
    
    boxes, scores, classes, nums = output_boxes(
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )

    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)
    win_name = "Image Detection"

    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()