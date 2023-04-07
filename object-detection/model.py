import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import ops

from utils import *

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
    
    def forward(self, img_data):
        return self.backbone(img_data)

class ProposalModule(nn.Module):
    def __init__(self, in_features, hidden_dim=512, n_anchors=9, p_dropout=0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors*4, kernel_size=1)

    def forward(self, feature_map, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = "eval"
        else:
            mode = "train"
        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))
        reg_offsets_pred = self.reg_head(out)
        conf_scores_pred = self.conf_head(out)
        if mode == "train":
            conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
            conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
            offsets_pos = reg_offsets_pred.contiguous().view(-1, 4)[pos_anc_ind]
            proposals = generate_proposals(pos_anc_coords, offsets_pos)
            return conf_scores_pos, conf_scores_neg, offsets_pos, proposals
        elif mode == "eval":
            return conf_scores_pred, reg_offsets_pred

class RegionProposalNetwork(nn.Module):
    def __init__(self, img_size, out_size, out_channels):
        super().__init__()
        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size
        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h
        self.anc_scales = [2, 4, 6]
        self.anc_ratios = [0.5, 1, 1.5]
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3
        self.w_conf = 1
        self.w_reg = 5
        self.feature_extractor = FeatureExtractor()
        self.proposal_module = ProposalModule(out_channels, n_anchors=self.n_anc_boxes)

    def forward(self, images, gt_bboxes, gt_classes):
        batch_size = images.size(dim=0)
        feature_map = self.feature_extractor(images)
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
        gt_bboxes_proj = project_bboxes(gt_bboxes, self.width_scale_factor, self.height_scale_factor, mode="p2a")
        positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, positive_anc_coords, negative_anc_coords, positive_anc_ind_sep = get_req_anchors(
            anc_boxes_all,
            gt_bboxes_proj,
            gt_classes
        )
        conf_scores_pos, conf_scores_neg, offset_pos, proposals = self.proposal_module(
            feature_map, positive_anc_ind,
            negative_anc_ind, positive_anc_coords
        )
        cls_loss = calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size)
        reg_loss = calc_bbox_reg_loss(GT_offsets, offset_pos, batch_size)
        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss
        return total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos

    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.feature_extractor(images)
            anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
            anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
            anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
            anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
            conf_scores_pred, offsets_pred = self.proposal_module(feature_map)
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)
            proposals_final, conf_scores_final = [], []
            for i in range(batch_size):
                conf_scores = torch.sigmoid(conf_scores_pred[i])
                offsets = offsets_pred[i]
                anc_boxes = anc_boxes_flat[i]
                proposals = generate_proposals(anc_boxes, offsets)
                conf_idx = torch.where(conf_scores >= conf_scores_pos, nms_thresh)[0]
                conf_scores_pos = conf_scores[conf_idx]
                proposals_pos = proposals[conf_idx]
                nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
                conf_scores_pos = conf_scores_pos[nms_idx]
                proposals_pos = proposals_pos[nms_idx]
                proposals_final.append(proposals_pos)
                conf_scores_final.append(conf_scores_pos)
            return proposals_final, conf_scores_final, feature_map

class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()
        self.roi_size = roi_size
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.cls_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, feature_map, proposals_list, gt_classes=None):
        pass

class TwoStageDetector(nn.Module):
    def __init__(self, img_size, out_size, out_channels, n_classes, roi_size):
        super().__init__()
        self.rpn = RegionProposalNetwork(img_size, out_size, out_channels)
        self.classifier = ClassificationModule(out_channels, n_classes, roi_size)

    def forward(self, images, gt_bboxes, gt_classes):
        pass

    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        pass

def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
    pass

def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
    pass