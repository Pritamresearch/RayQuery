"""
RayQuery End-to-End Model.
Integrates Multi-View Backbone, Query Generation, Ray-Uncertainty Scoring,
and Epistemic Anti-Dilution Cross-Attention.
"""
import torch
import torch.nn as nn
from configs.config import RayQueryConfig
from src.models.camera_backbone import CameraDepthBackbone
from src.models.query_generator import RayUncertaintyQueryGenerator
from src.models.demand_supply_selector import DemandSupplyQuerySelector, EpistemicAntiDilutionCrossAttention

class RayQueryDetector(nn.Module):
    def __init__(self, cfg: RayQueryConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = CameraDepthBackbone(feat_dim=cfg.feat_dim, d_bins=cfg.d_bins)
        self.query_gen = RayUncertaintyQueryGenerator(cfg)
        self.selector = DemandSupplyQuerySelector(alpha_uncertainty=cfg.alpha_uncertainty)
        self.fusion = EpistemicAntiDilutionCrossAttention(feat_dim=cfg.feat_dim)
        
        # Final Detection Output Heads
        self.final_cls = nn.Linear(cfg.feat_dim, cfg.num_classes)
        self.final_reg = nn.Linear(cfg.feat_dim, 7) # (dx, dy, dz, l, w, h, yaw)
        
    def forward(self, batch_data: dict, top_k_transmit: int = None):
        """
        Args:
            batch_data: Dict with 'images', 'intrinsics', 'extrinsics', 'rel_poses', 'n_agents'
            top_k_transmit: Number of queries transmitted per collaborator (for bandwidth sweeping)
        """
        if top_k_transmit is None:
            top_k_transmit = self.cfg.topk_transmit
            
        images = batch_data["images"]          # (N, Ncam, 3, H, W)
        if images.dim() == 6:
            images = images.squeeze(0)
            
        intrinsics = batch_data["intrinsics"]  # (N, Ncam, 3, 3)
        if intrinsics.dim() == 5:
            intrinsics = intrinsics.squeeze(0)
            
        extrinsics = batch_data["extrinsics"]  # (N, Ncam, 4, 4)
        if extrinsics.dim() == 5:
            extrinsics = extrinsics.squeeze(0)
            
        rel_poses = batch_data["rel_poses"]    # (N, 3): (dx, dy, dyaw) in ego frame
        if rel_poses.dim() == 3:
            rel_poses = rel_poses.squeeze(0)
            
        n_agents = batch_data["n_agents"]
        if isinstance(n_agents, torch.Tensor):
            n_agents = int(n_agents.flatten()[0].item())
        else:
            n_agents = int(n_agents)
        
        agent_queries = []
        agent_boxes = []
        agent_scores = []
        agent_uncertainties = []
        
        # Step 1: Local Feature Extraction & Ray-Uncertainty Query Generation per Agent
        for i in range(n_agents):
            imgs_i = images[i]  # (Ncam, 3, H, W)
            feats, d_prob, d_entropy = self.backbone(imgs_i)
            
            queries, boxes, scores, uncertainties = self.query_gen(
                feats, d_entropy, intrinsics[i], extrinsics[i]
            )
            agent_queries.append(queries)
            agent_boxes.append(boxes)
            agent_scores.append(scores)
            agent_uncertainties.append(uncertainties)
            
        ego_queries = agent_queries[0]
        ego_uncertainties = agent_uncertainties[0]
        ego_boxes = agent_boxes[0]
        
        # Step 2: Bandwidth-Constrained Query Selection & Transmission
        transmitted_queries = []
        transmitted_uncertainties = []
        bytes_transferred = 0
        
        for j in range(1, n_agents):
            # Select top-K queries based on Ray-Depth Uncertainty & Utility
            selected_idx, _ = self.selector(
                agent_scores[j], agent_uncertainties[j], agent_boxes[j], top_k=top_k_transmit
            )
            
            q_selected = agent_queries[j][selected_idx]
            u_selected = agent_uncertainties[j][selected_idx]
            box_selected = agent_boxes[j][selected_idx]
            
            # Coordinate transformation from collaborator frame into Ego frame
            dx, dy, dyaw = rel_poses[j]
            cos_y = torch.cos(dyaw)
            sin_y = torch.sin(dyaw)
            
            # Rotate & translate query coordinates to ego reference frame
            transformed_boxes = box_selected.clone()
            orig_x = box_selected[:, 0]
            orig_y = box_selected[:, 1]
            transformed_boxes[:, 0] = orig_x * cos_y - orig_y * sin_y + dx
            transformed_boxes[:, 1] = orig_x * sin_y + orig_y * cos_y + dy
            transformed_boxes[:, 6] += dyaw
            
            transmitted_queries.append(q_selected)
            transmitted_uncertainties.append(u_selected)
            
            # Honest Byte Accounting:
            # 7 floats per box (28 bytes) + 128 int8 quantized features (128 bytes) + 1 float uncertainty (4 bytes)
            # Total ~ 160 bytes per transmitted query
            bytes_transferred += len(selected_idx) * 160
            
        # Step 3: Epistemic Anti-Dilution Query Fusion
        if len(transmitted_queries) > 0:
            all_collab_queries = torch.cat(transmitted_queries, dim=0)
            all_collab_uncertainties = torch.cat(transmitted_uncertainties, dim=0)
            fused_ego_queries = self.fusion(
                ego_queries, ego_uncertainties, all_collab_queries, all_collab_uncertainties
            )
        else:
            fused_ego_queries = ego_queries
            
        # Step 4: Final 3D Bounding Box Predictions
        pred_logits = self.final_cls(fused_ego_queries)
        pred_offsets = self.final_reg(fused_ego_queries)
        
        final_boxes = ego_boxes.clone()
        final_boxes[:, :3] += pred_offsets[:, :3]
        final_boxes[:, 3:6] *= torch.exp(pred_offsets[:, 3:6])
        final_boxes[:, 6] += pred_offsets[:, 6]
        
        return {
            "pred_logits": pred_logits,
            "pred_boxes": final_boxes,
            "bytes_transferred": bytes_transferred,
            "kb_transferred": bytes_transferred / 1024.0,
            "n_collaborators": n_agents - 1
        }
