"""
3D Detection Criterion and Bounding Box Matching Loss for RayQuery.
Computes:
1. Spatial Nearest-Neighbor Assignment between 3D Queries and Ground Truth Boxes
2. Focal Classification Loss for foreground vs background queries
3. Smooth L1 3D Regression Loss on center (x, y, z), dimensions (l, w, h), and yaw
4. Communication Sparsity Regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RayQueryCriterion(nn.Module):
    def __init__(self, cls_weight: float = 2.0, reg_weight: float = 1.0,
                 pos_dist_thresh: float = 4.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.pos_dist_thresh = pos_dist_thresh

    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                gt_boxes: torch.Tensor, bytes_transferred: int = 0):
        """
        Args:
            pred_logits: (num_queries, 1)
            pred_boxes:  (num_queries, 7) -> [cx, cy, cz, l, w, h, yaw]
            gt_boxes:    (G, 7)
            bytes_transferred: int (bytes used in transmission)
        Returns:
            total_loss, loss_dict
        """
        device = pred_logits.device
        num_queries = pred_logits.shape[0]
        
        target_cls = torch.zeros((num_queries, 1), device=device, dtype=torch.float32)
        target_reg = torch.zeros_like(pred_boxes)
        reg_mask = torch.zeros((num_queries,), device=device, dtype=torch.bool)
        
        num_gt = gt_boxes.shape[0]
        
        if num_gt > 0:
            pred_centers = pred_boxes[:, :3] # (num_queries, 3)
            gt_centers = gt_boxes[:, :3]     # (G, 3)
            
            # Compute Pairwise Euclidean Distance Matrix (num_queries, G)
            dists = torch.cdist(pred_centers, gt_centers) # (num_queries, G)
            
            # Match each ground truth to its closest query
            min_dists, matched_queries = torch.min(dists, dim=0) # (G,)
            
            for g_idx in range(num_gt):
                q_idx = matched_queries[g_idx]
                if min_dists[g_idx] <= self.pos_dist_thresh:
                    target_cls[q_idx] = 1.0
                    target_reg[q_idx] = gt_boxes[g_idx]
                    reg_mask[q_idx] = True
                    
        # 1. Focal / BCE Classification Loss
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_cls, reduction="mean"
        )
        
        # 2. Smooth L1 3D Regression Loss (only on matched positive queries)
        if reg_mask.sum() > 0:
            p_pos = pred_boxes[reg_mask]
            t_pos = target_reg[reg_mask]
            
            # Center regression: (x, y, z)
            loss_center = F.smooth_l1_loss(p_pos[:, :3], t_pos[:, :3], reduction="mean")
            # Dimension regression: (l, w, h)
            loss_dim = F.smooth_l1_loss(torch.log(p_pos[:, 3:6].clamp(min=1e-3)),
                                        torch.log(t_pos[:, 3:6].clamp(min=1e-3)), reduction="mean")
            # Heading regression: sin(theta_pred - theta_gt)
            yaw_diff = p_pos[:, 6] - t_pos[:, 6]
            loss_yaw = torch.mean(torch.abs(torch.sin(yaw_diff)))
            
            reg_loss = loss_center + loss_dim + 0.5 * loss_yaw
        else:
            reg_loss = torch.tensor(0.0, device=device)
            
        # 3. Communication constraint (encourages sparse transmission)
        comm_loss = 0.001 * (bytes_transferred / 1024.0)
        
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss + comm_loss
        
        return total_loss, {
            "loss_cls": cls_loss.item(),
            "loss_reg": reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            "loss_total": total_loss.item(),
            "matched_pos": int(reg_mask.sum().item())
        }
