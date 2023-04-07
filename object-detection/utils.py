import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm

import torch
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim

def parse_annotation(annotation_path, image_dir, img_size):
    pass

def calc_gt_offsets(pos_anc_coords, ggt_bbox_mapping):
    pass

def gen_anc_centers(out_size):
    pass

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, model="a2p"):
    pass

def generate_proposals(anchors, offsets):
    pass

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    pass

def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    pass

def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
    pass

def display_img(img_data, fig, axes):
    pass

def display_bbox(bboxes, fig, ax, classes=None, in_format="xyxy", color="y", line_width=3):
    pass

def display_grid(x_points, y_points, fig, ax, special_point=None):
    pass