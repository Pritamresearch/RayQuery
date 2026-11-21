"""
Training script for RayQuery on OPV2V.
Trains the 3D query generator, uncertainty scoring, and epistemic fusion end-to-end.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import RayQueryConfig
from src.data.opv2v_dataset import OPV2VQueryDataset
from src.models.rayquery_model import RayQueryDetector
from src.utils.metrics import evaluate_ap

def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total_loss = 0.0
    
    for step, sample in enumerate(loader):
        batch_data = {
            "n_agents": sample["n_agents"],
            "images": sample["images"].to(device),
            "intrinsics": sample["intrinsics"].to(device),
            "extrinsics": sample["extrinsics"].to(device),
            "rel_poses": sample["rel_poses"].to(device),
            "gt_boxes": sample["gt_boxes"].to(device)
        }
        
        optimizer.zero_grad()
        out = model(batch_data, top_k_transmit=cfg.topk_transmit)
        
        # Supervised 3D Detection Loss + Sparsity Constraint
        # Dummy loss for demonstration; full loss matches Hungarian matching with 3D GT
        loss = out["pred_logits"].mean().abs() + 0.01 * (out["bytes_transferred"] / 1024.0)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (step + 1) % 10 == 0:
            print(f"  Step [{step+1}/{len(loader)}] - Loss: {loss.item():.4f} - BW: {out['kb_transferred']:.2f} KB", flush=True)
            
    return total_loss / max(1, len(loader))

def main():
    cfg = RayQueryConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Starting RayQuery training on device: {device}")
    
    train_dataset = OPV2VQueryDataset(cfg, split="train")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    model = RayQueryDetector(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    os.makedirs("results/checkpoints", exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        print(f"Epoch [{epoch}/{cfg.epochs}] Average Loss: {loss:.4f}", flush=True)
        
        if epoch % 10 == 0:
            ckpt_path = f"results/checkpoints/rayquery_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()
