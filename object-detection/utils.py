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
    img_h, img_w = img_size
    with open(annotation_path, "r") as f:
        tree = ET.parse(f)
    root = tree.getroot()
    img_paths, gt_boxes_all, gt_classes_all = [], [], []
    for object_ in root.findall("image"):
        img_path = os.path.join(image_dir, object_.get("name"))
        img_paths.append(img_path)
        orig_w = int(object_.get("width"))
        orig_h = int(object_.get("height"))
        groundtruth_boxes, groundtruth_classes = [], []
        for box_ in object_.findall("box"):
            xmin = float(box_.get("xtl"))
            ymin = float(box_.get("ytl"))
            xmax = float(box_.get("xbr"))
            ymax = float(box_.get("ybr"))
            bbox = torch.Tensor([xmin, ymin, xmax, ymax])
            bbox[[0, 2]] = bbox[[0, 2]] * img_w/orig_w
            bbox[[1, 3]] = bbox[[1, 3]] * img_h/orig_h
            groundtruth_boxes.append(bbox.tolist())
            label = box_.get("label")
            groundtruth_classes.append(label)
        gt_boxes_all.append(torch.Tensor(groundtruth_boxes))
        gt_classes_all.append(groundtruth_classes)
    return gt_boxes_all, gt_classes_all, img_paths

def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt="xyxy", out_fmt="cxcywh")
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt="xyxy", out_fmt="cxcywh")
    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[: ,2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]
    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    tw_ = torch.log(gt_w/anc_w)
    th_ = torch.log(gt_h/anc_h)
    return torch.stack([tx_, ty_, tw_, th_], dim=-1)

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