# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean

from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector


import sys
import logging
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("debug_log.txt", mode='w'),  # Save logs to a file
        logging.StreamHandler()  # Optional: Also print logs to the console
    ]
)
# Create a class to redirect print statements to logging
class PrintLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.logger.log(self.level, message)

    def flush(self):
        pass  # No need to flush for logging

# Redirect print statements to logging
sys.stdout = PrintLogger(logging.getLogger(), level=logging.INFO)
sys.stderr = PrintLogger(logging.getLogger(), level=logging.ERROR)  # Redirect errors

# Function to check and log ranges
def log_tensor_range(name, tensor):
    if tensor.numel() == 0:
        logging.warning(f"{name} is an empty tensor.")
    #else:
        #logging.info(f"{name} - min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}, mean: {tensor.mean().item():.6f}")

@MODELS.register_module()
class CrossKDGFL_ADD(CrossKDSingleStageDetector):

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        tea_x = self.teacher.extract_feat(batch_inputs)
        tea_cls_scores, tea_bbox_preds, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_crosskd_single, tea_x,
                        self.teacher.bbox_head.scales, module=self.teacher)
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_crosskd_single, stu_x,
                        self.bbox_head.scales, module=self)
        reused_cls_scores, reused_bbox_preds = multi_apply(
            self.reuse_teacher_head, tea_cls_hold, tea_reg_hold, stu_cls_hold,
            stu_reg_hold, self.teacher.bbox_head.scales)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        


        losses = self.loss_by_feat(tea_cls_scores, tea_bbox_preds, tea_x,
                                   stu_cls_scores, stu_bbox_preds, stu_x,
                                   reused_cls_scores, reused_bbox_preds,
                                   batch_gt_instances, batch_img_metas,
                                   batch_gt_instances_ignore)

        return losses

    def forward_crosskd_single(self, x, scale, module):
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.gfl_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred, cls_feat_hold, reg_feat_hold

    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        module = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        reused_cls_score = module.gfl_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.gfl_reg(reused_reg_feat)).float()
        return reused_cls_score, reused_bbox_pred

    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_feats: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            feats: List[Tensor],
            reused_cls_scores: List[Tensor],
            reused_bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl,\
            new_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.bbox_head.prior_generator.strides,
                avg_factor=avg_factor)

        new_avg_factor = sum(new_avg_factor)
        new_avg_factor = reduce_mean(new_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / new_avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / new_avg_factor, losses_dfl))
        losses = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

        losses_cls_kd, losses_reg_kd, kd_avg_factor = multi_apply(
            self.pred_mimicking_loss_single,
            anchor_list,
            tea_cls_scores,
            tea_bbox_preds,
            reused_cls_scores,
            reused_bbox_preds,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.bbox_head.prior_generator.strides,
            avg_factor=avg_factor)
        kd_avg_factor = sum(kd_avg_factor)
        kd_avg_factor = reduce_mean(kd_avg_factor).clamp_(min=1).item()
        losses_reg_kd = list(map(lambda x: x / kd_avg_factor, losses_reg_kd))
        losses.update(
            dict(loss_cls_kd=losses_cls_kd, loss_reg_kd=losses_reg_kd))

        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(feats, tea_feats)
            ]
            losses.update(loss_feat_kd=losses_feat_kd)

        for name, param in self.named_parameters():
            if torch.isnan(param.data).any():
                print(f"NaN in parameter data: {name}")
            if torch.isinf(param.data).any():
                print(f"Inf in parameter data: {name}")

        return losses
    
    def distances_to_xywh_with_normalization(self, encoder_output):
        """
        Convert encoder output (distances [left, top, right, bottom]) to [x, y, w, h] format 
        and normalize using batch mean and variance.

        Args:
            encoder_output (Tensor): Shape (N, 4), distances [left, top, right, bottom].
            anchors (Tensor): Shape (N, 2), anchor coordinates [x_center, y_center].

        Returns:
            Tensor: Normalized bounding boxes in [x, y, w, h] format, shape (N, 4).
        """
        if (encoder_output < 0).any():
            raise ValueError("Distances (left, top, right, bottom) must be non-negative.")



        # Unpack distances
        left, top, right, bottom = encoder_output[:, 0], encoder_output[:, 1], encoder_output[:, 2], encoder_output[:, 3]
        x_center, y_center = 0, 0
        
        # Calculate coordinates of the bounding box
        x_min = x_center - left
        x_max = x_center + right
        y_min = y_center - top
        y_max = y_center + bottom
                
        if (x_min > x_max).any() or (y_min > y_max).any():
            raise ValueError("Computed coordinates are invalid: x_min > x_max or y_min > y_max.")
        
        # Convert to [x, y, w, h] format
        x = (x_min + x_max) / 2  # Center x
        y = (y_min + y_max) / 2  # Center y
        w = x_max - x_min        # Width
        h = y_max - y_min        # Height

            # To avoid issues with zero or extremely small w and h
        eps = 1e-8
        #w = torch.clamp(w, min=eps)
        #h = torch.clamp(h, min=eps)
            # Apply log transform to w and h
        log_w = torch.log(w)
        log_h = torch.log(h)
        # Stack results into a single tensor
        boxes_log = torch.stack([x, y, log_w, log_h], dim=1)
        
        # Normalize the boxes using batch mean and standard deviation
        mean = boxes_log.mean(dim=0, keepdim=True)  # Shape (1, 4), mean for x, y, w, h
        std = boxes_log.std(dim=0, keepdim=True)    # Shape (1, 4), std for x, y, w, h
        
    # Check for zero or extremely small std and handle it
        #eps = 1e-8
        # Handle zero or very small std by adding eps
        #std = std + eps

        # Normalize the boxes
        # Normalize the boxes in log-space
        normalized_boxes = (boxes_log - mean) / std
        return normalized_boxes


    def process_bbox_predictions(self, cls_score, tea_bbox_pred, labels, anchors, stride):
        """
        Processes bounding box predictions for positive anchors.

        Args:
            tea_bbox_pred (Tensor): Bounding box predictions from the teacher model with shape (N, C, H, W).
            labels (Tensor): Labels for all anchors with shape (N x num_total_anchors).
            anchors (Tensor): Anchor coordinates with shape (N x num_total_anchors, 4).
            stride (tuple): The stride of the feature map (e.g., (8, 8)).

        Returns:
            dict: Processed results containing:
                - pos_bbox_pred (Tensor): Positive bounding box predictions.
                - pos_decode_bbox_pred (Tensor): Decoded positive bounding boxes.
                - pos_anchors (Tensor): Positive anchors.
                - pos_inds (Tensor): Indices of positive anchors.
        """

        # Move to CPU for reliable checking
        tea_bbox_pred_cpu = tea_bbox_pred.cpu()



        if torch.isinf(tea_bbox_pred_cpu).any():
            print("Inf found in tea_bbox_pred before permutation")
        #print("tea_bbox_pred min:", tea_bbox_pred.min().item())
        #print("tea_bbox_pred max:", tea_bbox_pred.max().item())
        #print("tea_bbox_pred dtype:", tea_bbox_pred.dtype)
        #print("tea_bbox_pred device:", tea_bbox_pred.device)


        # Reshape bounding box predictions

                # Check for NaNs and Infs
        if torch.isnan(tea_bbox_pred_cpu).any():
            print("NaN found in tea_bbox_pred before permutation")
            tea_bbox_pred = torch.clamp(tea_bbox_pred, min=-1e4, max=1e4)


        bbox_pred = tea_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.bbox_head.reg_max + 1))
        labels = labels.reshape(-1)  # Flatten labels

        # Identify background class index and positive indices
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        # Prepare result dictionary
        #results = {"pos_bbox_pred": None, "pos_decode_bbox_pred": None, "pos_anchors": None, "pos_inds": pos_inds}
        pos_bbox_pred_corners = None
        # Process positive indices if any exist
        if len(pos_inds) > 0:
            pos_bbox_pred = bbox_pred[pos_inds]  # Positive bbox predictions
            pos_anchors = anchors[pos_inds]  # Positive anchors
            pos_anchor_centers = self.bbox_head.anchor_center(pos_anchors) / stride[0]  # Convert to centers

                        # Check for NaNs or Infs in pos_anchor_centers
            if torch.isnan(pos_anchor_centers).any():
                print("NaN found in pos_anchor_centers")
            if torch.isinf(pos_anchor_centers).any():
                print("Inf found in pos_anchor_centers")


            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            # Convert logits to bounding box corners
            pos_bbox_pred_corners = self.bbox_head.integral(pos_bbox_pred)

            # Check for NaNs or Infs in pos_bbox_pred_corners
            if torch.isnan(pos_bbox_pred_corners).any():
                print("NaN found in pos_bbox_pred_corners")
            if torch.isinf(pos_bbox_pred_corners).any():
                print("Inf found in pos_bbox_pred_corners")


            # Decode bounding box predictions
            #pos_decode_bbox_pred = self.bbox_head.bbox_coder.decode(pos_anchor_centers, pos_bbox_pred_corners)
            #pos_diffrent_bbox = self.distances_to_xywh_with_normalization(pos_bbox_pred_corners)
            pos_decode_bbox_pred = self.bbox_head.bbox_coder.decode(pos_anchor_centers, pos_bbox_pred_corners)

            log_tensor_range("pos_decode_bbox_pred", pos_decode_bbox_pred)
            log_tensor_range("pos_bbox_pred_corners", pos_bbox_pred_corners)


        else:
            weight_targets = bbox_pred.new_tensor(0)
        return pos_bbox_pred_corners, weight_targets


    def pred_mimicking_loss_single(self, anchors, tea_cls_score, tea_bbox_pred, 
                                   reused_cls_score, reused_bbox_pred, stu_bbox,
                                   labels_list, 
                                   label_weights,
                                    bbox_targets_list,
                                                stride,
                                      avg_factor):
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4) # (n x num total anchors, 4)

        # classification branch distillation
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels) #(N x h x w, num_classes)
        reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels) #(N x h x w, num_classes)
        label_weights = label_weights.reshape(-1) #(N x num_total_anchors)


        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # regression branch distillation, corner, chua sua
        tea_pos_decode_bbox_pred, weight_targets = self.process_bbox_predictions(tea_cls_score, tea_bbox_pred, labels_list, anchors, stride) #x1y1x2y2 #(num_positive_anchors,4)
        reused_pos_decode_bbox_pred, _ = self.process_bbox_predictions(tea_cls_score, reused_bbox_pred, labels_list, anchors, stride) #x1y1x2y2 #(num_positive_anchors,4)
        if tea_pos_decode_bbox_pred != None and reused_pos_decode_bbox_pred != None and not torch.isnan(reused_bbox_pred).any() :
            analyze_bboxes(reused_pos_decode_bbox_pred.detach().cpu(), tea_pos_decode_bbox_pred.detach().cpu())
            loss_reg_kd = self.loss_reg_kd(
                reused_pos_decode_bbox_pred,
                tea_pos_decode_bbox_pred,
                weight=weight_targets, # (num_positive_anchors,)
                avg_factor=1.0)
        else:
            loss_reg_kd = tea_bbox_pred.sum() * 0

        return loss_cls_kd, loss_reg_kd, weight_targets.sum()
    
import torch
import cv2
import matplotlib.pyplot as plt

def analyze_bboxes(distances, points, image=None, ground_truth=None):
    """
    Analyze bounding boxes using distances from points to edges, detect invalid boxes, overlaps, and IoU.
    
    Args:
        distances (Tensor): Tensor of shape (n, 4) with distances [left, top, right, bottom].
        points (Tensor): Tensor of shape (n, 2) with points [x, y].
        image (ndarray, optional): Image for visual debugging, shape (H, W, C).
        ground_truth (Tensor, optional): Tensor of shape (m, 4) with ground truth boxes in "xyxy" format.
    
    Returns:
        dict: Dictionary containing analysis results, such as invalid boxes, overlaps, and IoU matrix.
    """
    results = {}

    # 1. Reconstruct Bounding Boxes from Distances
    left = points[:, 0] - distances[:, 0]
    top = points[:, 1] - distances[:, 1]
    right = points[:, 0] + distances[:, 2]
    bottom = points[:, 1] + distances[:, 3]
    bboxes = torch.stack([left, top, right, bottom], dim=-1)

    # 2. Detect Invalid Boxes
    width = right - left
    height = bottom - top
    invalid_widths = (width <= 0)
    invalid_heights = (height <= 0)
    invalid_boxes = invalid_widths | invalid_heights
    if invalid_boxes.any():
        print(f"Invalid boxes detected: {bboxes[invalid_boxes]}")
    results['invalid_boxes'] = bboxes[invalid_boxes].tolist() if invalid_boxes.any() else []

    # 3. Detect Overlaps
    def has_overlap(box1, box2):
        # Check if two boxes overlap
        return not (box1[2] <= box2[0] or box1[0] >= box2[2] or 
                    box1[3] <= box2[1] or box1[1] >= box2[3])
    
    overlaps = []
    n_boxes = bboxes.size(0)
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            if has_overlap(bboxes[i], bboxes[j]):
                overlaps.append((i, j))
    #if overlaps:
        #print(f"Overlapping boxes: {overlaps}")
    results['overlaps'] = overlaps

    # 4. Visual Debugging
    if image is not None:
        img_with_bboxes = image.copy()
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox.int().tolist()
            img_with_bboxes = cv2.rectangle(img_with_bboxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        plt.imshow(img_with_bboxes)
        plt.title("Bounding Boxes")
        plt.show()

    # 5. Compare with Ground Truth (IoU)
    def compute_iou(box_a, box_b):
        # Compute Intersection over Union (IoU)
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        
        union_area = box_a_area + box_b_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    if ground_truth is not None:
        ious = torch.zeros((n_boxes, ground_truth.size(0)))
        for i in range(n_boxes):
            for j in range(ground_truth.size(0)):
                ious[i, j] = compute_iou(bboxes[i], ground_truth[j])
        results['iou_matrix'] = ious.numpy()
        #print(f"IoU matrix:\n{ious}")

    return results