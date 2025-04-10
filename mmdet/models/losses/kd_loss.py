# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.registry import MODELS
from .utils import weight_reduce_loss, weighted_loss

import sys
import logging
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("debug_log.txt"),  # Save logs to a file
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

@weighted_loss
def knowledge_distillation_js_div_loss(pred,
                                       soft_label,
                                       T,
                                       class_reduction='mean',
                                       detach_target=True):
    """
    Loss function for knowledge distillation using Jensen-Shannon divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, n + 1).
        T (float): Temperature for distillation.
        class_reduction (str): 'mean' or 'sum' over class dimension.
        detach_target (bool): Remove soft_label from automatic differentiation.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    # Compute the probability distributions
    P = F.softmax(soft_label / T, dim=1)
    Q = F.softmax(pred / T, dim=1)
    if detach_target:
        P = P.detach()
    
    # Compute the mixture distribution M
    M = 0.5 * (P + Q)

    # Compute KL divergences
    kl_pm = F.kl_div(torch.log(P), M, reduction='none')  # KL(P || M)
    kl_qm = F.kl_div(torch.log(Q), M, reduction='none')  # KL(Q || M)

    # Compute JS divergence
    js_loss = 0.5 * (kl_pm + kl_qm)

    # Reduce over the class dimension
    if class_reduction == 'mean':
        js_loss = js_loss.sum(dim=1)  # Sum over classes to get per-sample loss
    elif class_reduction == 'sum':
        js_loss = js_loss.sum(dim=1)
    else:
        raise NotImplementedError(f"Unknown class_reduction: {class_reduction}")
    
    # Apply temperature scaling
    js_loss = js_loss * (T * T)
    return js_loss


def bbox_to_gaussian(bboxes):
    """
    Convert bounding boxes to Gaussian parameters (mean and covariance).
    Args:
        bboxes (Tensor): Tensor of shape (N, 4) with bounding boxes (x1, y1, x2, y2).
    Returns:
        means (Tensor): Tensor of shape (N, 2) with the means of the Gaussian (center_x, center_y).
        covs (Tensor): Tensor of shape (N, 2, 2) with covariance matrices.
    """
    x1, y1, x2, y2 = bboxes.unbind(dim=1)
    
    # Calculate center of the bounding box (mean of the Gaussian)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    means = torch.stack([center_x, center_y], dim=-1)
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Calculate covariance (diagonal matrix since no rotation)
    a2 = (width / 2) ** 2  # Variance along x-axis
    b2 = (height / 2) ** 2  # Variance along y-axis
    covs = torch.zeros(bboxes.size(0), 2, 2).to(bboxes.device)
    covs[:, 0, 0] = a2  # Var(x)
    covs[:, 1, 1] = b2  # Var(y)
    
    return means, covs

def bbox_to_gaussian_xywh(bboxes):
    """
    Convert bounding boxes in (x, y, w, h) format to Gaussian parameters (mean and covariance).
    Args:
        bboxes (Tensor): Tensor of shape (N, 4) with bounding boxes (center_x, center_y, width, height).
    Returns:
        means (Tensor): Tensor of shape (N, 2) with the means of the Gaussian (center_x, center_y).
        covs (Tensor): Tensor of shape (N, 2, 2) with covariance matrices.
    """
    # Extract components of the bounding boxes
    center_x, center_y, width, height = bboxes.unbind(dim=1)
    
    # Means (center of the bounding box)
    means = torch.stack([center_x, center_y], dim=-1)
    
    # Calculate variance (assuming no rotation, diagonal covariance)
    a2 = (width / 2) ** 2  # Variance along x-axis
    b2 = (height / 2) ** 2  # Variance along y-axis
    covs = torch.zeros(bboxes.size(0), 2, 2).to(bboxes.device)
    covs[:, 0, 0] = a2  # Var(x)
    covs[:, 1, 1] = b2  # Var(y)
    
    return means, covs

def regression_loss(pred_bboxes, target_bboxes, js_divergence, tau=1):
    """
    Compute the regression loss L_reg using the geometric mean JS divergence.

    Parameters:
    - pred_bboxes (torch.Tensor): Predicted bounding boxes of shape (N, 4), 
                                  where each box is (x1, y1, x2, y2).
    - target_bboxes (torch.Tensor): Ground-truth bounding boxes of shape (N, 4),
                                    where each box is (x1, y1, x2, y2).
    - js_divergence (torch.Tensor): Tensor of shape (N,) with the geometric JS divergence 
                                    for each distribution pair.
    - tau (float): Offset hyperparameter (default: 1.0).

    Returns:
    - torch.Tensor: Regression loss of shape (N,).
    """

    # Assertions to ensure input validity
    assert pred_bboxes.shape == target_bboxes.shape, "Shape mismatch between predicted and target bboxes"
    assert js_divergence.ndim == 1 and js_divergence.shape[0] == pred_bboxes.shape[0], \
        "JS divergence must be a tensor of shape (N,) matching the batch size"
    
    # Calculate f(JS^G_alpha(N_P, N_G))
    js_term = js_divergence ** 2  # f(.) is the square operation on JS divergence
    loss_fractional = 1 / (tau + js_term)  # Use Eq (12) to compute loss term

    # Incorporate normalized regression terms
    regression_loss = 1 - loss_fractional

    return regression_loss



@weighted_loss
def geometric_js_divergence(stu_corner, tea_corner, alpha=0.5):
    """
    Compute the geometric Jensen-Shannon (JS) divergence between two batches of Gaussian distributions.
    
    Parameters:
    - pred_means (torch.Tensor): Tensor of shape (N, 2) with the means of the predicted distributions.
    - pred_covs (torch.Tensor): Tensor of shape (N, 2, 2) with the covariance matrices of the predicted distributions.
    - target_means (torch.Tensor): Tensor of shape (N, 2) with the means of the target distributions.
    - target_covs (torch.Tensor): Tensor of shape (N, 2, 2) with the covariance matrices of the target distributions.
    - alpha (float): Weight parameter for the geometric mixture (default: 0.5).
    
    Returns:
    - torch.Tensor: Tensor of shape (N,) with the geometric JS divergence for each distribution pair.
    """
# Assertions to ensure valid inputs
    assert stu_corner.ndim == 2 and stu_corner.shape[1] == 4, \
        "Predicted bounding boxes must be a tensor of shape (N, 4)"
    assert tea_corner.ndim == 2 and tea_corner.shape[1] == 4, \
        "Target bounding boxes must be a tensor of shape (N, 4)"
    assert 0 <= alpha <= 1, "Alpha must be in the range [0, 1]"

        # Check for NaNs or Infs in reused_pos_decode_bbox_pred
    if torch.isnan(stu_corner).any():
        print("NaN found in reused_pos_decode_bbox_pred")
    if torch.isinf(stu_corner).any():
        print("Inf found in reused_pos_decode_bbox_pred")

    # Similarly for tea_pos_decode_bbox_pred
    if torch.isnan(tea_corner).any():
        print("NaN found in tea_pos_decode_bbox_pred")
    if torch.isinf(tea_corner).any():
        print("Inf found in tea_pos_decode_bbox_pred")


 # Convert bounding boxes to Gaussian parameters
    target_means, target_covs = bbox_to_gaussian_distances(tea_corner) #xywh
    pred_means, pred_covs = bbox_to_gaussian_distances(stu_corner)

        # After bbox_to_gaussian
    if torch.isnan(pred_means).any():
        print("NaN found in pred_means")
    if torch.isnan(pred_covs).any():
        print("NaN found in pred_covs")


    identity = torch.eye(pred_covs.size(-1)).to(pred_covs.device)
    regularization=1e-6
    # Regularize covariance matrices
    pred_covs += identity * regularization
    target_covs += identity * regularization
    # Assertions to check Gaussian conversion validity
    assert pred_means.shape == target_means.shape, \
        "Converted means must have the same shape for predicted and target distributions"
    assert pred_covs.shape == target_covs.shape, \
        "Converted covariance matrices must have the same shape for predicted and target distributions"

    # Compute Σ_α (harmonic mean of covariance matrices)
    inv_pred_covs = torch.linalg.inv(pred_covs)  # (N, 2, 2)
    inv_target_covs = torch.linalg.inv(target_covs)  # (N, 2, 2)
    sigma_alpha_inv = (1 - alpha) * inv_pred_covs + alpha * inv_target_covs  # (N, 2, 2)
    sigma_alpha = torch.linalg.inv(sigma_alpha_inv)  # (N, 2, 2)

    # Compute μ_α (weighted mean)
    mu_alpha = sigma_alpha @ (
        (1 - alpha) * (inv_pred_covs @ pred_means.unsqueeze(-1)) +
        alpha * (inv_target_covs @ target_means.unsqueeze(-1))
    )  # (N, 2, 1)
    mu_alpha = mu_alpha.squeeze(-1)  # (N, 2)

    # Trace term
    weighted_cov_sum = (1 - alpha) * pred_covs + alpha * target_covs  # (N, 2, 2)
    #print("pred_cor:", pred_covs)
    #print("tar_cor:", target_covs)
    #print("sigma_alpha_inv:", sigma_alpha_inv)
    #print("weighted_cov_sum:", weighted_cov_sum)

    trace_term = torch.einsum('bij,bij->b', sigma_alpha_inv, weighted_cov_sum)  # (N,)

    # Log-determinant term
    det_sigma_alpha = torch.linalg.det(sigma_alpha)  # (N,)
    det_pred_covs = torch.linalg.det(pred_covs)  # (N,)
    det_target_covs = torch.linalg.det(target_covs)  # (N,)
    log_det_term = torch.log(det_sigma_alpha) - (1 - alpha) * torch.log(det_pred_covs) - alpha * torch.log(det_target_covs)  # (N,)

    # Log determinants and check bounds
    log_tensor_range("det_pred_covs", det_pred_covs)
    log_tensor_range("det_target_covs", det_target_covs)
    if (det_pred_covs <= 0).any() or (det_target_covs <= 0).any():
        logging.error("Non-positive determinants found in covariance matrices!")
    log_tensor_range("log_det_term", log_det_term)


    # Mean contribution
    diff_pred = (mu_alpha - pred_means)  # (N, 2)
    diff_target = (mu_alpha - target_means)  # (N, 2)
    mean_pred_term = (1 - alpha) * torch.einsum('bi,bij,bj->b', diff_pred, sigma_alpha_inv, diff_pred)  # (N,)
    mean_target_term = alpha * torch.einsum('bi,bij,bj->b', diff_target, sigma_alpha_inv, diff_target)  # (N,)
    if not torch.all(torch.isfinite(trace_term)):
        print("Trace Term contains invalid values!")
        print("Trace Term:", trace_term)
    log_tensor_range("trace_term", trace_term)

    assert torch.all(torch.isfinite(trace_term)), "Trace Term is invalid (NaN or Inf)"

    if not torch.all(torch.isfinite(mean_pred_term)):
        print("Mean Contribution (Pred) contains invalid values!")
        print("Mean Contribution (Pred):", mean_pred_term)
    assert torch.all(torch.isfinite(mean_pred_term)), "Mean Contribution (Pred) is invalid (NaN or Inf)"

    if not torch.all(torch.isfinite(mean_target_term)):
        print("Mean Contribution (Target) contains invalid values!")
        print("Mean Contribution (Target):", mean_target_term)
    assert torch.all(torch.isfinite(mean_target_term)), "Mean Contribution (Target) is invalid (NaN or Inf)"

    if not torch.all(torch.isfinite(log_det_term)):
        print("Log Determinant Term contains invalid values!")
        print("Log Determinant Term:", log_det_term)
        print("Determinants of Predicted Covariance Matrices:", torch.linalg.det(pred_covs))
        print("Determinants of Target Covariance Matrices:", torch.linalg.det(target_covs))
    assert torch.all(torch.isfinite(log_det_term)), "Log Determinant Term is invalid (NaN or Inf)"

    # Combine terms
    js_divergence = 0.5 * (trace_term + log_det_term - 2 + mean_pred_term + mean_target_term)  # (N,)
    log_tensor_range("js_divergence", js_divergence)


    # Ensure non-negative divergence
    if not torch.all(js_divergence >= -1e-6):
        print("Predicted Means:", pred_means)
        print("Target Means:", target_means)
        print("Predicted Covariances:", pred_covs)
        print("Target Covariances:", target_covs)
        print("Trace Term:", trace_term)
        print("Log-Determinant Term:", log_det_term)
        print("Mean Contribution (Pred):", mean_pred_term)
        print("Mean Contribution (Target):", mean_target_term)
        print("Geometric JS Divergence:", js_divergence)

    assert torch.all(js_divergence >= -1e-6), "Geometric JS divergence should not be significantly negative"
    #js_divergence = torch.clamp(js_divergence, min=0.0)
    loss = regression_loss(stu_corner, tea_corner,js_divergence)
    if (loss < 0).any():
        print("Predicted Means:", pred_means)
        print("Target Means:", target_means)
        print("Predicted Covariances:", pred_covs)
        print("Target Covariances:", target_covs)
        print("Trace Term:", trace_term)
        print("Log-Determinant Term:", log_det_term)
        print("Mean Contribution (Pred):", mean_pred_term)
        print("Mean Contribution (Target):", mean_target_term)
        print("Geometric JS Divergence:", js_divergence)

        print("loss before weight:", loss)
        breakpoint()
    log_tensor_range("loss before weight", loss)

    return loss

@MODELS.register_module()
class KnowledgeDistillationJSDivLoss(nn.Module):
    def __init__(self,
                 class_reduction='mean',
                 reduction='mean',
                 loss_weight=1.0,
                 T=10):
        super(KnowledgeDistillationJSDivLoss, self).__init__()
        assert T >= 1
        self.class_reduction = class_reduction
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_js_div_loss(
            pred,
            soft_label,
            weight,
            class_reduction=self.class_reduction,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd

@weighted_loss
def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       class_reduction='mean',
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none')
    if class_reduction == 'mean':
        kd_loss = kd_loss.mean(1)
    elif class_reduction == 'sum':
        kd_loss = kd_loss.sum(1)
    else:
        raise NotImplementedError
    kd_loss = kd_loss * (T * T)
    return kd_loss


def kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       class_reduction='mean',
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none')
    if class_reduction == 'mean':
        kd_loss = kd_loss.mean(1)
    elif class_reduction == 'sum':
        kd_loss = kd_loss.sum(1)
    else:
        raise NotImplementedError
    kd_loss = kd_loss * (T * T)
    return kd_loss

@weighted_loss
def knowledge_distillation_kl_div_loss_ls(pred,
                                       soft_label,
                                       T,
                                       loss_weight,
                                       class_reduction='mean',
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    if detach_target:
        soft_label_ls = soft_label.detach()
    loss_ls = kd_loss_ls(pred, soft_label_ls, T, True)

    loss_kd = loss_weight * loss_ls
    N = loss_weight.numel() // 4
    weight_reshaped = loss_weight.view(N, 4)
    reg_weights_reverted = weight_reshaped[:, 0]

    target = F.softmax(soft_label, dim=1)
    if detach_target:
        target = target.detach()
    target_ratio = sum_distributions_loop(target)
    
    # # Step 2: Apply Softmax
    # softmax_pred = F.softmax(pred, dim=1)
    # pred_ratio = sum_distributions_loop(softmax_pred)
    # loss_ratio = kl_div_loss(pred_ratio, target_ratio, T)
    
    # loss_add = reg_weights_reverted * loss_ratio
    # loss_add_duplicated = torch.repeat_interleave(loss_add, repeats=4)
    #total_loss = loss_kd + loss_add_duplicated

    return loss_ls 



def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss_ls(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def sum_distributions_loop(input_tensor):
    """
    Performs sum distributions via convolution for (left + right) and (top + bottom)
    distributions using a looped approach.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (4*N, n+1),
                                     where 4 corresponds to (top, bottom, left, right).

    Returns:
        torch.Tensor: Sum distributions for each object, shape (N, 2, 2n +1).
                      The second dimension corresponds to:
                      [0] -> left + right,
                      [1] -> top + bottom.
    """
    # Validate input dimensions
    total_rows, num_bins = input_tensor.shape
    assert total_rows % 4 == 0, "Input tensor's first dimension must be a multiple of 4."
    N = total_rows // 4
    n = num_bins - 1

    # Reshape to (N, 4, n+1)
    input_reshaped = input_tensor.view(N, 4, num_bins)

    # Extract distributions
    top = input_reshaped[:, 0, :]      # Shape: (N, n+1)
    bottom = input_reshaped[:, 1, :]   # Shape: (N, n+1)
    left = input_reshaped[:, 2, :]     # Shape: (N, n+1)
    right = input_reshaped[:, 3, :]    # Shape: (N, n+1)

    # Initialize the output tensor for sum distributions
    # 2 sums per object: [left + right, top + bottom]
    sum_bins = 2 * n + 1
    sum_distributions = torch.zeros(N, 2, sum_bins, device=input_tensor.device, dtype=input_tensor.dtype)

    for i in range(N):
        # Retrieve the distributions for the i-th object
        left_dist = left[i]       # Shape: (n+1,)
        right_dist = right[i]     # Shape: (n+1,)
        top_dist = top[i]         # Shape: (n+1,)
        bottom_dist = bottom[i]   # Shape: (n+1,)

        # Define a helper function for convolution
        def convolve_distributions(dist_a, dist_b):
            # Reshape distributions to (batch_size=1, channels=1, length=n+1)
            dist_a_reshaped = dist_a.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n+1)
            dist_b_reshaped = dist_b.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n+1)

            # Flip the second distribution for convolution
            dist_b_flipped = torch.flip(dist_b_reshaped, dims=[2])

            # Pad the first distribution on both sides with n zeros to achieve 'full' convolution
            dist_a_padded = F.pad(dist_a_reshaped, (n, n))  # Shape: (1, 1, 2n +1)

            # Perform convolution
            conv_result = F.conv1d(dist_a_padded, dist_b_flipped)  # Shape: (1, 1, 2n +1)

            return conv_result.squeeze()  # Shape: (2n +1,)

        # Compute left + right sum
        sum_left_right = convolve_distributions(left_dist, right_dist)  # Shape: (2n +1,)

        # Compute top + bottom sum
        sum_top_bottom = convolve_distributions(top_dist, bottom_dist)  # Shape: (2n +1,)

        # Assign the results to the output tensor
        sum_distributions[i, 0, :] = sum_left_right
        sum_distributions[i, 1, :] = sum_top_bottom
        # Compute the ratio: Sum_LR / Sum_TB
    # To avoid division by zero, add a small epsilon to the denominator
    epsilon = 1e-8
    ratio_distributions = sum_distributions[:, 0, :] / (sum_distributions[:, 1, :] + epsilon)

    return ratio_distributions

def kd_quality_focal_loss(pred,
                          target,
                          weight=None,
                          beta=1,
                          reduction='mean',
                          avg_factor=None):
    num_classes = pred.size(1)
    if weight is not None:
        weight = weight[:, None].repeat(1, num_classes)

    target = target.detach().sigmoid()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    focal_weight = torch.abs(pred.sigmoid() - target).pow(beta)
    loss = loss * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 class_reduction='mean',
                 reduction='mean',
                 loss_weight=1.0,
                 T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.class_reduction = class_reduction
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        
        total_loss = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            loss_weight=weight,
            class_reduction=self.class_reduction,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)


        return total_loss
    
@MODELS.register_module()
class KnowledgeDistillationKLDivLoss_ls(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 class_reduction='mean',
                 reduction='mean',
                 loss_weight=1.0,
                 T=10):
        super(KnowledgeDistillationKLDivLoss_ls, self).__init__()
        assert T >= 1
        self.class_reduction = class_reduction
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss_ls(
            pred,
            soft_label,
            weight,
            class_reduction=self.class_reduction,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd

@MODELS.register_module()
class KnowledgeDistillationGeometricJSLoss(nn.Module):
    """Loss function for knowledge distilling using geometric JS divergence."""

    def __init__(self, class_reduction='mean', reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationGeometricJSLoss, self).__init__()
        assert T >= 1
        self.class_reduction = class_reduction
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T



    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function to calculate loss based on geometric JS divergence."""
        reduction = reduction_override if reduction_override else self.reduction

        # Calculate geometric JS divergence for each pair of Gaussian distributions
        js_divergence = geometric_js_divergence(pred, target)

        # Apply weight and reduction if specified
        if weight is not None:
            js_divergence = js_divergence * weight

        if reduction == 'mean':
            js_divergence = js_divergence.mean()
        elif reduction == 'sum':
            js_divergence = js_divergence.sum()

        return self.loss_weight * js_divergence
        #return js_divergence


@MODELS.register_module()
class KDQualityFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(KDQualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss = self.loss_weight * kd_quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss

import torch

def normalize_distances(distances, epsilon=1e-6, k=1.0):
    """
    Normalize distances using Log Transformation + Mean and Variance Normalization
    and shift the output to make it non-negative.
    
    Args:
        distances (Tensor): Tensor of shape (n, 4), distances [left, top, right, bottom].
        epsilon (float): Small value to avoid log(0).
    
    Returns:
        Tensor: Shifted and normalized distances, shape (n, 4).
    """

        # Step 1: Ensure non-negative input for log transformation
    distances = torch.clamp(distances, min=epsilon)

    # Step 2: Apply log transformation
    log_distances = torch.log(distances)
    
    # Step 3: Mean and Variance Normalization

    return log_distances


def bbox_to_gaussian_distances(distances):
    """
    Convert distances [left, top, right, bottom] directly into Gaussian parameters (mean and covariance).

    Args:
        distances (Tensor): Tensor of shape (n, 4), distances [left, top, right, bottom].

    Returns:
        means (Tensor): Tensor of shape (n, 2) with the means of the Gaussian (center_x, center_y).
        covs (Tensor): Tensor of shape (n, 2, 2) with covariance matrices.
    """
    # Extract distances
    left, top, right, bottom = distances[:, 0], distances[:, 1], distances[:, 2], distances[:, 3]

    log_tensor_range("Before: left",left)
    log_tensor_range("top",top)
    log_tensor_range("right",right)
    log_tensor_range("bottom",bottom)
    distances = normalize_distances(distances)
    left, top, right, bottom = distances[:, 0], distances[:, 1], distances[:, 2], distances[:, 3]

    log_tensor_range("After: left",left)
    log_tensor_range("top",top)
    log_tensor_range("right",right)
    log_tensor_range("bottom",bottom)

    
    # Calculate center of the bounding box (mean of the Gaussian)
    center_x = (right - left) / 2 + left
    center_y = (bottom - top) / 2 + top
    means = torch.stack([center_x, center_y], dim=-1)
    
    # Calculate width and height
    width = right + left
    height = top + bottom
    
    # Calculate covariance (diagonal matrix since no rotation)
    a2 = (width / 2) ** 2  # Variance along x-axis
    b2 = (height / 2) ** 2  # Variance along y-axis
    covs = torch.zeros(distances.size(0), 2, 2).to(distances.device)
    covs[:, 0, 0] = a2  # Var(x)
    covs[:, 1, 1] = b2  # Var(y)
    
    return means, covs


def main():
    reused_pos_decode_bbox_pred = torch.tensor([[10, 15, 20, 25], [30, 35, 40, 50]], dtype=torch.float32)  # (N, 4)
    tea_pos_decode_bbox_pred = torch.tensor([[12, 17, 22, 27], [32, 37, 42, 52]], dtype=torch.float32)  # (N, 4)

    try:
        js_divergence = geometric_js_divergence(reused_pos_decode_bbox_pred, tea_pos_decode_bbox_pred)
        print("Geometric JS Divergence:", js_divergence)
    except AssertionError as e:
        print("Error:", e)

if __name__ == "__main__":
    # Call the main function when this file is run directly
    main()