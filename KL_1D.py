import torch
import torch.nn.functional as F

# Set temperature and number of bins
T = 2  # Temperature for distillation
num_bins = 5  # Discrete bins

# Example logits for teacher and student (representing one coordinate)
teacher_logits = torch.tensor([2.0, 1.0, 0.5, -0.5, -1.0])
student_logits = torch.tensor([1.0, 1.5, 0.8, -0.2, -0.5])

# Convert logits to soft distributions
teacher_probs = F.softmax(teacher_logits / T, dim=0)
student_probs = F.softmax(student_logits / T, dim=0)

# Calculate KL divergence (per bin)
kd_loss = F.kl_div(F.log_softmax(student_logits / T, dim=0), teacher_probs, reduction='none')
kd_loss_sum = kd_loss.sum()  # Total KL divergence for this coordinate

print("Teacher probabilities:", teacher_probs) #tensor([0.3863, 0.2343, 0.1825, 0.1107, 0.0862])
print("Student probabilities:", student_probs) #tensor([0.2375, 0.3050, 0.2149, 0.1304, 0.1122])
print("KL divergence loss per bin:", kd_loss) # [ 0.1879, -0.0618, -0.0299, -0.0181, -0.0227]
print("Total KL divergence for coordinate:", kd_loss_sum) #tensor(0.0554) tong phia tren


# Assume 5 bins per coordinate for simplicity
num_coords = 4  # For left, top, right, bottom
teacher_logits = torch.tensor([[2.0, 1.0, 0.5, -0.5, -1.0]] * num_coords)
student_logits = torch.tensor([[1.0, 1.5, 0.8, -0.2, -0.5]] * num_coords)

# Softmax for each coordinate
teacher_probs = F.softmax(teacher_logits / T, dim=1)
student_probs = F.softmax(student_logits / T, dim=1)

# KL divergence for each coordinate
kd_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1), teacher_probs, reduction='none')
kd_loss_sum = kd_loss.sum(dim=1)  # Sum over bins for each coordinate
total_kd_loss = kd_loss_sum.mean()  # Mean loss over all coordinates

print("Teacher probabilities per coordinate:", teacher_probs)
print("Student probabilities per coordinate:", student_probs)
print("KL divergence per bin:", kd_loss)
print("KL divergence per coordinate:", kd_loss_sum)
print("Total KL divergence for bounding box:", total_kd_loss) #sum the losses per coordinate
###################
# Simulate logits for 3 bounding boxes, each with 4 coordinates (left, top, right, bottom), and 5 bins per coordinate
batch_size = 3
num_coords = 4
teacher_logits = torch.randn(batch_size, num_coords, num_bins)
student_logits = torch.randn(batch_size, num_coords, num_bins)

# Classification scores to create weights
tea_cls_score = torch.randn(batch_size, num_coords)
reg_weights = tea_cls_score.max(dim=1)[0].sigmoid()  # Confidence-based weights

# Convert logits to probability distributions
teacher_probs = F.softmax(teacher_logits / T, dim=2)
student_probs = F.softmax(student_logits / T, dim=2)

# Calculate KL divergence loss for each box and each coordinate
kd_loss = F.kl_div(F.log_softmax(student_logits / T, dim=2), teacher_probs, reduction='none')
kd_loss_sum = kd_loss.sum(dim=2)  # Sum over bins for each coordinate
weighted_kd_loss = (kd_loss_sum * reg_weights[:, None]).mean()  # Apply weights and average

print("Teacher probabilities per bbox coordinate:", teacher_probs)
print("Student probabilities per bbox coordinate:", student_probs)
print("KL divergence per bin per coordinate:", kd_loss)
print("KL divergence per coordinate per bbox:", kd_loss_sum)
print("Weighted total KD loss:", weighted_kd_loss)
