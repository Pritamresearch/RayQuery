"""
Verification & Smoke Test for RayQuery.
Tests data loading, Ray-Depth Uncertainty query generation,
Epistemic Anti-Dilution cross-attention, and byte accounting on real OPV2V Culver City data.
"""
import sys
import os
import torch

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import RayQueryConfig
from src.data.opv2v_dataset import OPV2VQueryDataset
from src.models.rayquery_model import RayQueryDetector
from src.utils.metrics import evaluate_ap

def run_verification():
    print("=================================================================")
    print(" RayQuery: Depth-Uncertainty & Demand-Guided Cooperative Detection")
    print(" Pipeline Verification & Baseline Test")
    print("=================================================================\n")
    
    cfg = RayQueryConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Configuration Loaded: Device = {device}")
    print(f"    Data Root: {cfg.data_root}")
    
    # 1. Dataset Verification
    print("\n[2] Initializing OPV2V Dataset Loader (Val split)...")
    dataset = OPV2VQueryDataset(cfg, split="validate")
    print(f"    Found {len(dataset)} validation sequences.")
    
    sample = dataset[0]
    print(f"    Sample frame loaded successfully:")
    print(f"      - Number of Agents: {sample['n_agents']}")
    print(f"      - Camera Images:    {sample['images'].shape}")
    print(f"      - Relative Poses:   \n{sample['rel_poses'].numpy().round(2)}")
    print(f"      - 3D Ground Truth:  {sample['gt_boxes'].shape[0]} vehicles")
    
    # 2. Model Initialization
    print("\n[3] Initializing RayQuery Model Architecture...")
    model = RayQueryDetector(cfg).to(device)
    
    # Move sample to device
    batch_data = {
        "n_agents": sample["n_agents"],
        "images": sample["images"].to(device),
        "intrinsics": sample["intrinsics"].to(device),
        "extrinsics": sample["extrinsics"].to(device),
        "rel_poses": sample["rel_poses"].to(device),
        "gt_boxes": sample["gt_boxes"].to(device)
    }
    
    # 3. Model Forward Pass
    print("\n[4] Running Forward Pass (Simulating 5G-V2X Transmission)...")
    model.eval()
    with torch.no_grad():
        out = model(batch_data, top_k_transmit=30)
        
    print(f"    Forward Pass Successful:")
    print(f"      - Prediction Logits: {out['pred_logits'].shape}")
    print(f"      - Predicted 3D Boxes: {out['pred_boxes'].shape}")
    print(f"      - Bandwidth Payload:  {out['kb_transferred']:.2f} KB/frame")
    print(f"      - Collaborators Fused: {out['n_collaborators']}")
    
    # 4. Metric Verification
    pred_boxes_np = out["pred_boxes"].cpu().numpy()
    pred_scores_np = torch.sigmoid(out["pred_logits"]).squeeze().cpu().numpy()
    gt_boxes_np = sample["gt_boxes"].numpy()
    
    ap_results = evaluate_ap(pred_boxes_np, pred_scores_np, gt_boxes_np)
    print("\n[5] Metrics Check (Untrained Random Weights Initial Baseline):")
    for th, ap in ap_results.items():
        print(f"      - AP@{th:.1f}: {ap * 100:.2f}%")
        
    print("\n=================================================================")
    print(" ALL PIPELINE CHECKS PASSED: RayQuery is ready for research & training!")
    print("=================================================================")

if __name__ == "__main__":
    run_verification()
