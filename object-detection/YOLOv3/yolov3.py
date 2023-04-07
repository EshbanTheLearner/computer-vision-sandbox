import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D

devices = tf.config.experimetnal.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(devices[0], True)

def parse_cfg():
    pass

def YOLOv3Net(cfg_file, model_size, num_classes):
    pass
