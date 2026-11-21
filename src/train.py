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
from src.models.criterion import RayQueryCriterion
from src.utils.metrics import evaluate_ap

def train_one_epoch(model, criterion, loader, optimizer, device, cfg):
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
        
        loss, loss_dict = criterion(
            pred_logits=out["pred_logits"],
            pred_boxes=out["pred_boxes"],
            gt_boxes=batch_data["gt_boxes"],
            bytes_transferred=out["bytes_transferred"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        if (step + 1) % 10 == 0:
            print(f"  Step [{step+1}/{len(loader)}] - Loss: {loss.item():.4f} (Cls: {loss_dict['loss_cls']:.3f}, Reg: {loss_dict['loss_reg']:.3f}, Matched: {loss_dict['matched_pos']}) - BW: {out['kb_transferred']:.2f} KB", flush=True)
            
    return total_loss / max(1, len(loader))

def main():
    cfg = RayQueryConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Starting RayQuery training on device: {device}")
    
    train_dataset = OPV2VQueryDataset(cfg, split="train")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    model = RayQueryDetector(cfg).to(device)
    criterion = RayQueryCriterion(
        cls_weight=cfg.cls_weight,
        reg_weight=cfg.reg_weight,
        pos_dist_thresh=4.0
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    
    os.makedirs("results/checkpoints", exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, criterion, train_loader, optimizer, device, cfg)
        scheduler.step()
        print(f"Epoch [{epoch}/{cfg.epochs}] Average Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)
        
        if epoch % 5 == 0 or epoch == cfg.epochs:
            ckpt_path = f"results/checkpoints/rayquery_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}", flush=True)

if __name__ == "__main__":
    main()
