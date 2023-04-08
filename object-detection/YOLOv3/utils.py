import tensorflow as tf
import numpy as np
import cv2
import time

def resize_image(inputs, model_size):
    inputs = tf.image.resize(inputs, model_size)
    return inputs

def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class, iou_threshold, confidence_threshold):
    center_x, center_y, width, height, confidence, classes = tf.split(
        inputs, 
        [1, 1, 1, 1, 1, -1],
        axis=-1
    )
    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0

    inputs = tf.concat([
        top_left_x,
        top_left_y,
        bottom_right_x,
        bottom_right_y,
        confidence,
        classes
    ], axis=-1)

    boxes_dicts = non_max_suppression(
        inputs,
        model_size,
        max_output_size,
        max_output_size_per_class,
        iou_threshold,
        confidence_threshold
    )

    return boxes_dicts

def draw_outputs(img, boxes, objectness, classes, nums, class_names):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes = np.array(boxes)
    for i in range(nums):
        x1y1 = tuple((boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))
        img = cv2.rectangle(img, (x1y1), (x2y2), (255, 0, 0), 2)
        img = cv2.putText(img, "{} {:.4f}".format(class_names[int(classes[i])], objectness[i]), (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        return img

def load_class_names(file_name):
    with open(file_name, "r") as f:
        class_names = f.read().splitlines()
    return class_names

def non_max_suppression(inputs, model_size, max_output_size, max_output_size_per_class, iou_threshold, confidence_threshold):
    pass