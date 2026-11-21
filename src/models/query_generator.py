"""
Ray-Uncertainty 3D Object Query Generator.
Generates 3D proposal queries and computes their Ray-Depth Uncertainty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RayUncertaintyQueryGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_queries = cfg.num_queries
        self.feat_dim = cfg.feat_dim
        
        # Learnable 3D Reference Points (cx, cy, cz) in local vehicle coordinates
        # Uniformly distributed over the region of interest
        init_x = torch.linspace(cfg.x_min + 5, cfg.x_max - 5, int(self.num_queries**0.5))
        init_y = torch.linspace(cfg.y_min + 5, cfg.y_max - 5, int(self.num_queries**0.5))
        grid_x, grid_y = torch.meshgrid(init_x, init_y, indexing="ij")
        ref_points = torch.zeros(self.num_queries, 3)
        ref_points[:, 0] = grid_x.flatten()[:self.num_queries]
        ref_points[:, 1] = grid_y.flatten()[:self.num_queries]
        ref_points[:, 2] = 0.0  # nominal ground level
        self.ref_points = nn.Parameter(ref_points, requires_grad=True)
        
        # Query Embedding Feature
        self.query_embed = nn.Embedding(self.num_queries, self.feat_dim)
        
        # Proposal Refinement Head
        self.score_head = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.box_head = nn.Linear(self.feat_dim, 7) # (dx, dy, dz, l, w, h, yaw)

    def forward(self, features: torch.Tensor, depth_entropy: torch.Tensor,
                intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        """
        Sample visual features and depth entropy at the 3D reference points.
        
        Args:
            features: (Ncam, feat_dim, H_feat, W_feat)
            depth_entropy: (Ncam, 1, H_feat, W_feat)
            intrinsics: (Ncam, 3, 3)
            extrinsics: (Ncam, 4, 4)
        Returns:
            queries: (num_queries, feat_dim)
            ref_boxes: (num_queries, 7) -> (cx, cy, cz, l, w, h, yaw)
            scores: (num_queries,)
            ray_uncertainty: (num_queries,) -> scalar depth-uncertainty for each query
        """
        device = features.device
        ncam, _, h_feat, w_feat = features.shape
        
        # Expand reference points: (num_queries, 4) in homogeneous coordinates
        pts_3d = torch.cat([self.ref_points, torch.ones((self.num_queries, 1), device=device)], dim=-1)
        
        sampled_feats = torch.zeros((self.num_queries, self.feat_dim), device=device)
        sampled_entropies = torch.ones((self.num_queries,), device=device) # default high uncertainty
        valid_cam_count = torch.zeros((self.num_queries, 1), device=device)
        
        # Project 3D reference points to each camera plane
        for c in range(ncam):
            K = intrinsics[c]      # (3, 3)
            ext = extrinsics[c]    # (4, 4)
            
            # Ego -> Camera: T_cam_ego = inv(ext)
            T_cam_ego = torch.linalg.inv(ext)
            pts_cam = (T_cam_ego @ pts_3d.T).T # (num_queries, 4)
            depth = pts_cam[:, 2]              # Z coordinate (optical axis)
            
            # Points strictly in front of the camera
            in_front = depth > self.cfg.d_min
            
            # 2D projection: p_img = K * p_cam / Z
            pts_img = (K @ pts_cam[:, :3].T).T # (num_queries, 3)
            u = pts_img[:, 0] / (depth + 1e-6)
            v = pts_img[:, 1] / (depth + 1e-6)
            
            # Normalize to [-1, 1] for grid_sample
            u_norm = (u / (self.cfg.img_w - 1)) * 2 - 1
            v_norm = (v / (self.cfg.img_h - 1)) * 2 - 1
            
            in_fov = in_front & (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
            
            if in_fov.any():
                grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0) # (1, 1, num_queries, 2)
                f_samp = F.grid_sample(features[c:c+1], grid, align_corners=False).view(self.feat_dim, self.num_queries).T # (num_queries, feat_dim)
                e_samp = F.grid_sample(depth_entropy[c:c+1], grid, align_corners=False).view(self.num_queries) # (num_queries,)
                
                mask = in_fov.float().unsqueeze(-1)
                sampled_feats += f_samp * mask
                # Assign lowest entropy (most certain view)
                sampled_entropies = torch.where(in_fov, torch.minimum(sampled_entropies, e_samp), sampled_entropies)
                valid_cam_count += mask
                
        # Normalize aggregated feature
        valid_mask = (valid_cam_count > 0).float()
        sampled_feats = sampled_feats / (valid_cam_count + 1e-6) * valid_mask
        
        # Combine with learnable query embedding
        queries = self.query_embed.weight + sampled_feats
        
        # Prediction
        scores = self.score_head(queries).squeeze(-1) # (num_queries,)
        offsets = self.box_head(queries)              # (num_queries, 7)
        
        ref_boxes = torch.zeros_like(offsets)
        ref_boxes[:, :3] = self.ref_points + offsets[:, :3] # refined center
        ref_boxes[:, 3:6] = torch.exp(offsets[:, 3:6])      # dimensions (l, w, h)
        ref_boxes[:, 6] = offsets[:, 6]                     # yaw angle
        
        return queries, ref_boxes, scores, sampled_entropies
