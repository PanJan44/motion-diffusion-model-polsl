from diffusion.nn import mean_flat, sum_flat
import torch
import numpy as np

def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l2(a, b):
    return (a - b) ** 2

def normalize_quaternion(q):
    """
    q: [B, 21, 4, T] B->batch size, T number of frames
    returns [B, 21, 4, T]
    """
    return q / (q.norm(dim=2, keepdim=True) + 1e-8)

def quaternion_geodesic_distance(q1, q2):
    q1, q2 = normalize_quaternion(q1), normalize_quaternion(q2)
    dot = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0)
    return 2.0 * torch.acos(torch.abs(dot))

def quaternion_loss(q1, q2):
    """
    q1, q2: [B, 21, 4, T] B->batch size, T number of frames
    """
    q1, q2 = normalize_quaternion(q1), normalize_quaternion(q2)
    return 1.0 - torch.sum(q1 * q2, dim=2) ** 2

def quaternion_geodesic_distance_loss(a, b, mask, loss_fn=quaternion_loss, epsilon=1e-8, entries_norm=True):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen

    B, _, _, frames_num = a.shape
    non_rot_a = torch.cat([a[:, :67], a[:, 151:]], dim=1)  # [B, 137, 1, frames_num]
    non_rot_b = torch.cat([b[:, :67], b[:, 151:]], dim=1)
    non_rot_loss = diff_l2(non_rot_a, non_rot_b) # [B, 137, 1, frames_num]

    # quaternions
    rot_a = a[:, 67:151, 0, :].view(B, 21, 4, frames_num)  # [B, 21, 4, frames_num] 21 -> all but root
    rot_b = b[:, 67:151, 0, :].view(B, 21, 4, frames_num)
    rot_loss = loss_fn(rot_a, rot_b).unsqueeze(2) # [B, 21, 1, frames_num]

    loss = torch.cat([non_rot_loss, rot_loss], dim=1)  # [B, 158, 1, frames_num]
    loss = (loss * mask.float()).sum()

    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val

def masked_l2(a, b, mask, loss_fn=diff_l2, epsilon=1e-8, entries_norm=True):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    loss = loss_fn(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val


def masked_goal_l2(pred_goal, ref_goal, cond, all_goal_joint_names):
    all_goal_joint_names_w_traj = np.append(all_goal_joint_names, 'traj')
    target_joint_idx = [[np.where(all_goal_joint_names_w_traj == j)[0][0] for j in sample_joints] for sample_joints in cond['target_joint_names']]
    loc_mask = torch.zeros_like(pred_goal[:,:-1], dtype=torch.bool)
    for sample_idx in range(loc_mask.shape[0]):
        loc_mask[sample_idx, target_joint_idx[sample_idx]] = True
    loc_mask[:, -1, 1] = False  # vertical joint of 'traj' is always masked out
    loc_loss = masked_l2(pred_goal[:,:-1], ref_goal[:,:-1], loc_mask, entries_norm=False)
    
    heading_loss = masked_l2(pred_goal[:,-1:, :1], ref_goal[:,-1:, :1], cond['is_heading'].unsqueeze(1).unsqueeze(1), loss_fn=angle_l2, entries_norm=False)

    loss =  loc_loss + heading_loss
    return loss
