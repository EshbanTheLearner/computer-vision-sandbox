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
    pass

def load_class_names(file_name):
    pass

def non_max_suppression(inputs, model_size, max_output_size, max_output_size_per_class, iou_threshold, confidence_threshold):
    pass