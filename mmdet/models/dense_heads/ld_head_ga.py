# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from ..utils import multi_apply, unpack_gt_instances
from .gfl_head import GFLHead


@MODELS.register_module()
class LDHead(GFLHead):
    """Localization distillation Head. (Short description)

    It utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student. Original paper: `Localization
    Distillation for Object Detection. <https://arxiv.org/abs/2102.12252>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss_ld (:obj:`ConfigDict` or dict): Config of Localization
            Distillation Loss (LD), T is the temperature for distillation.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_ld: ConfigType = dict(
                     type='LocalizationDistillationLoss',
                     loss_weight=0.25,
                     T=10),

                 **kwargs) -> dict:

        super().__init__(
            num_classes=num_classes, in_channels=in_channels, **kwargs)
        self.loss_ld = MODELS.build(loss_ld)

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

        # Check for NaNs and Infs
        if torch.isnan(tea_bbox_pred_cpu).any():
            print("NaN found in tea_bbox_pred before permutation")
            return None, bbox_pred.new_tensor(0)

        if torch.isinf(tea_bbox_pred_cpu).any():
            print("Inf found in tea_bbox_pred before permutation")
        #print("tea_bbox_pred min:", tea_bbox_pred.min().item())
        #print("tea_bbox_pred max:", tea_bbox_pred.max().item())
        #print("tea_bbox_pred dtype:", tea_bbox_pred.dtype)
        #print("tea_bbox_pred device:", tea_bbox_pred.device)

        if not tea_bbox_pred.is_contiguous():
            tea_bbox_pred = tea_bbox_pred.contiguous()

        # Reshape bounding box predictions

        bbox_pred = tea_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.bbox_head.reg_max + 1))
        labels = labels.reshape(-1)  # Flatten labels

        # Identify background class index and positive indices
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        # Prepare result dictionary
        #results = {"pos_bbox_pred": None, "pos_decode_bbox_pred": None, "pos_anchors": None, "pos_inds": pos_inds}
        pos_decode_bbox_pred = None
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
            pos_decode_bbox_pred = self.bbox_head.bbox_coder.decode(pos_anchor_centers, pos_bbox_pred_corners)
        else:
            weight_targets = bbox_pred.new_tensor(0)
        return pos_decode_bbox_pred, weight_targets

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            stride: Tuple[int], soft_targets: Tensor,
                            avg_factor: int):
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            soft_targets (Tensor): Soft BBox regression targets.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        
        
            # Replace NaNs and Infs
        bbox_pred = torch.where(torch.isnan(bbox_pred), torch.zeros_like(bbox_pred), bbox_pred)
        bbox_pred = torch.where(torch.isinf(bbox_pred), torch.zeros_like(bbox_pred), bbox_pred)

        soft_targets = soft_targets.permute(0, 2, 3,
                                            1).reshape(-1,
                                                       4 * (self.reg_max + 1))

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]


            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)

            pos_soft_targets = soft_targets[pos_inds]
            ###
            pos_bbox_pred_corners_soft = self.integral(pos_soft_targets)
            pos_decode_bbox_pred_soft = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners_soft)

            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)

            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)
            
            ###


            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

            # ld loss
            # loss_ld = self.loss_ld(
            #     pred_corners,
            #     soft_corners,
            #     weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
            #     avg_factor=4.0)
            

            # # regression branch distillation
            # tea_pos_decode_bbox_pred, weight_targets = self.process_bbox_predictions(tea_cls_score, pos_decode_bbox_targets, labels_list, anchors, stride) #x1y1x2y2 #(num_positive_anchors,4)
            # reused_pos_decode_bbox_pred, _ = self.process_bbox_predictions(tea_cls_score, pos_decode_bbox_pred, labels_list, anchors, stride) #x1y1x2y2 #(num_positive_anchors,4)

            loss_ld = self.loss_ld(
                pos_decode_bbox_pred,
                pos_decode_bbox_pred_soft,
                weight=weight_targets, # (num_positive_anchors,)
                avg_factor=4.0)

        else:
            loss_ld = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_dfl, loss_ld, weight_targets.sum()

    def loss(self, x: List[Tensor], out_teacher: Tuple[Tensor],
             batch_data_samples: SampleList) -> dict:
        """
        Args:
            x (list[Tensor]): Features from FPN.
            out_teacher (tuple[Tensor]): The output of teacher.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        outs = self(x)
        soft_targets = out_teacher[1]
        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              soft_targets)
        losses = self.loss_by_feat(
            *loss_inputs, batch_gt_instances_ignore=batch_gt_instances_ignore)

        return losses

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            soft_targets: List[Tensor],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            soft_targets (list[Tensor]): Soft BBox regression targets.
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
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl, losses_ld, \
            avg_factor = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.prior_generator.strides,
                soft_targets,
                avg_factor=avg_factor)

        avg_factor = sum(avg_factor) + 1e-6
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = [x / avg_factor for x in losses_bbox]
        losses_dfl = [x / avg_factor for x in losses_dfl]
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_ld=losses_ld)
