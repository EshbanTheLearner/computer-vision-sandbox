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
    out_h, out_w = out_size
    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5
    return anc_pts_x, anc_pts_y

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode="a2p"):
    assert mode in ["a2p", "p2a"]
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1)
    if mode == "a2p":
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)
    proj_bboxes.resize_as_(bboxes)
    return proj_bboxes

def generate_proposals(anchors, offsets):
    anchors = ops.box_convert(anchors, in_fmt="xyxy", out_fmt="cxcywh")
    proposals_ = torch.zeros_like(anchors)
    proposals_[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    proposals_[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
    proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])
    proposals = ops.box_convert(proposals_, in_fmt="cxcywh", out_fmt="xyxy")
    return proposals

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    n_anc_boxes = len(anc_scales) * len(anc_ratios)
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0), anc_pts_y.size(dim=0), n_anc_boxes, 4)
    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale * ratio
                    h = scale
                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2
                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1
            anc_base[:, ix, jx, :] = ops.clip_boxes_to_images(anc_boxes, size=out_size)
    return anc_base

def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    tot_anc_boxes = anc_boxes_flat.size(dim=1)
    ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)
    return ious_mat

def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
    pass

def display_img(img_data, fig, axes):
    pass

def display_bbox(bboxes, fig, ax, classes=None, in_format="xyxy", color="y", line_width=3):
    pass

def display_grid(x_points, y_points, fig, ax, special_point=None):
    pass