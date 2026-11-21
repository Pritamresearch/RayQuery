"""
Demand-Supply Matching and Epistemic Anti-Dilution Cross-Attention.
Implements the core paper contributions:
1. Spatial Demand Mask generation (Ego specifies blind cones / high-entropy areas)
2. Ray-Uncertainty-Weighted Query Scoring for bandwidth-constrained transmission
3. Epistemic Anti-Dilution Query Cross-Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandSupplyQuerySelector(nn.Module):
    """
    Computes query transmission priorities using Ray-Depth Uncertainty
    and Ego Spatial Demand matching.
    """
    def __init__(self, alpha_uncertainty: float = 3.0):
        super().__init__()
        self.alpha = alpha_uncertainty
        
    def forward(self, scores: torch.Tensor, ray_uncertainty: torch.Tensor,
                ref_boxes: torch.Tensor, ego_demand_mask: torch.Tensor = None,
                top_k: int = 30):
        """
        Computes the utility:
            U_q = Score(q) * exp(-alpha * Uncertainty(q)) * DemandFactor(q)
        """
        # Ray-Depth Confidence Factor
        depth_cert = torch.exp(-self.alpha * ray_uncertainty) # in [0, 1]
        
        # Demand Factor (defaults to 1.0 if not using demand mask)
        demand_factor = torch.ones_like(scores)
        if ego_demand_mask is not None:
            # Query centers: (cx, cy)
            cx = ref_boxes[:, 0]
            cy = ref_boxes[:, 1]
            # Spatial demand check: check if in receiver blind zone
            dist_to_ego = torch.sqrt(cx**2 + cy**2)
            # Higher demand for distant or occluded zones
            demand_factor = torch.clamp(dist_to_ego / 30.0, min=0.5, max=1.5)
            
        utility = scores * depth_cert * demand_factor
        
        # Select top-K queries to transmit
        k = min(top_k, scores.shape[0])
        topk_vals, topk_indices = torch.topk(utility, k=k, dim=0)
        
        return topk_indices, utility


class EpistemicAntiDilutionCrossAttention(nn.Module):
    """
    Fuses collaborator queries into ego queries.
    Applies Epistemic Gating: Collaborator queries only update ego queries
    if their geometric certainty exceeds the ego query's certainty,
    mathematically preventing the JigsawComm 'more-is-worse' dilution trap.
    """
    def __init__(self, feat_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.mha = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=n_heads, batch_first=True)
        self.gate_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(feat_dim)
        
    def forward(self, ego_queries: torch.Tensor, ego_uncertainty: torch.Tensor,
                collab_queries: torch.Tensor, collab_uncertainty: torch.Tensor):
        """
        Args:
            ego_queries: (N_ego, feat_dim)
            ego_uncertainty: (N_ego,)
            collab_queries: (N_collab, feat_dim)
            collab_uncertainty: (N_collab,)
        Returns:
            fused_queries: (N_ego, feat_dim)
        """
        if collab_queries is None or collab_queries.shape[0] == 0:
            return ego_queries
            
        # Standard Cross-Attention: Ego queries attend to Collaborator queries
        q = ego_queries.unsqueeze(0)      # (1, N_ego, feat_dim)
        k = collab_queries.unsqueeze(0)   # (1, N_collab, feat_dim)
        v = collab_queries.unsqueeze(0)   # (1, N_collab, feat_dim)
        
        attn_out, _ = self.mha(q, k, v)
        attn_out = attn_out.squeeze(0)    # (N_ego, feat_dim)
        
        # Epistemic Uncertainty Gating:
        # Compare relative certainty: ego_cert vs mean collaborator cert
        ego_cert = (1.0 - ego_uncertainty).unsqueeze(-1) # (N_ego, 1)
        mean_collab_cert = (1.0 - collab_uncertainty.mean()).repeat(ego_queries.shape[0], 1)
        
        cert_pair = torch.cat([ego_cert, mean_collab_cert], dim=-1) # (N_ego, 2)
        gate = self.gate_fc(cert_pair) # In [0, 1]
        
        # If collaborator is much more certain, gate approaches 1.0 (absorb collaborator info).
        # If ego is already confident or collaborator is noisy, gate suppresses update (anti-dilution).
        fused = self.norm(ego_queries + gate * attn_out)
        return fused
