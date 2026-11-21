"""
Configuration for RayQuery: Depth-Uncertainty & Demand-Guided Query Exchange
for Camera-Only Cooperative 3D Detection (V2V).
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import os

@dataclass
class RayQueryConfig:
    # --- Experiment Identity ---
    exp_name: str = "rayquery_baseline"
    seed: int = 42
    device: str = "cuda"  # fallback to 'cpu' if cuda not available
    
    # --- Dataset Path (Refers to existing Culver City subset without duplication) ---
    data_root: str = r"C:\PhD\Collaborative autonomy\coop_v2v\coop_v2v\test_culver_city"
    n_cams: int = 4
    img_h: int = 192
    img_w: int = 256
    
    # --- 3D Metric Coordinate Range (Ego frame, in meters) ---
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -3.0
    z_max: float = 3.0
    
    # --- Camera Depth Discretization (for ray uncertainty estimation) ---
    d_min: float = 2.0
    d_max: float = 52.0
    d_bins: int = 50  # 1 meter per depth bin
    feat_dim: int = 128  # query feature dimension
    
    # --- Query Configuration ---
    num_queries: int = 100       # maximum local queries per agent
    max_collaborators: int = 4   # maximum surrounding CAVs
    topk_transmit: int = 30      # default top-k queries transmitted per collaborator
    
    # --- Novelty Switches & Hyperparameters ---
    alpha_uncertainty: float = 3.0   # penalty weight for ray-depth uncertainty
    use_demand_guided: bool = True   # ego broadcasts spatial demand mask
    use_epistemic_gate: bool = True  # anti-dilution uncertainty gating at fusion
    
    # --- Communication Byte Budget Sweeps (for AP vs. Bandwidth curves) ---
    # Measured in approximate query capacity: [5, 10, 20, 30, 50, 100]
    budget_queries_sweep: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 50, 100])
    
    # --- Training & Optimization ---
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 80
    num_classes: int = 1  # 3D vehicle detection
    
    # Loss weights
    cls_weight: float = 2.0
    reg_weight: float = 1.0
    uncertainty_weight: float = 0.5
    
    def validate(self):
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
