"""
Training script for RayQuery on OPV2V with Pretrained ResNet Backbone
and GPU Acceleration (Automatic Mixed Precision on RTX 3050).
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

def train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg):
    model.train()
    total_loss = 0.0
    
    for step, sample in enumerate(loader):
        batch_data = {
            "n_agents": sample["n_agents"],
            "images": sample["images"].to(device, non_blocking=True),
            "intrinsics": sample["intrinsics"].to(device, non_blocking=True),
            "extrinsics": sample["extrinsics"].to(device, non_blocking=True),
            "rel_poses": sample["rel_poses"].to(device, non_blocking=True),
            "gt_boxes": sample["gt_boxes"].to(device, non_blocking=True)
        }
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Training on NVIDIA GPU
        with torch.amp.autocast("cuda"):
            out = model(batch_data, top_k_transmit=cfg.topk_transmit)
            loss, loss_dict = criterion(
                pred_logits=out["pred_logits"],
                pred_boxes=out["pred_boxes"],
                gt_boxes=batch_data["gt_boxes"],
                bytes_transferred=out["bytes_transferred"]
            )
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        if (step + 1) % 10 == 0:
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"  Step [{step+1}/{len(loader)}] - Loss: {loss.item():.4f} "
                  f"(Cls: {loss_dict['loss_cls']:.3f}, Reg: {loss_dict['loss_reg']:.3f}, "
                  f"Matched: {loss_dict['matched_pos']}) | GPU VRAM: {vram_mb:.0f}MB", flush=True)
            
    return total_loss / max(1, len(loader))

def main():
    cfg = RayQueryConfig()
    
    # Verify and enforce CUDA GPU
    assert torch.cuda.is_available(), "CUDA GPU is required for training!"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("=" * 65)
    print(f" RayQuery GPU Training Pipeline")
    print(f" Target Device:  {gpu_name} ({total_vram_gb:.2f} GB VRAM)")
    print(f" Total Epochs:   {cfg.epochs} | Learning Rate: {cfg.lr}")
    print(f" Backbone:       Pretrained ResNet-18 (ImageNet Weights)")
    print(f" Precision:      AMP FP16 (Tensor Core Accelerated)")
    print("=" * 65 + "\n")
    
    train_dataset = OPV2VQueryDataset(cfg, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x[0],
        pin_memory=True
    )
    
    model = RayQueryDetector(cfg).to(device)
    criterion = RayQueryCriterion(
        cls_weight=cfg.cls_weight,
        reg_weight=cfg.reg_weight,
        pos_dist_thresh=4.0
    ).to(device)
    
    # Differential learning rate: slightly lower LR for pretrained backbone layers
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr * 0.5},
        {"params": head_params, "lr": cfg.lr}
    ], weight_decay=cfg.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    
    os.makedirs("results/checkpoints", exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, criterion, train_loader, optimizer, scaler, device, cfg)
        scheduler.step()
        
        lr_now = scheduler.get_last_lr()[0]
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"Epoch [{epoch}/{cfg.epochs}] Average Loss: {loss:.4f} | LR: {lr_now:.6f} | VRAM: {vram_mb:.0f}MB\n", flush=True)
        
        if epoch % 10 == 0 or epoch == cfg.epochs:
            ckpt_path = f"results/checkpoints/rayquery_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[+] Checkpoint saved: {ckpt_path}\n", flush=True)

if __name__ == "__main__":
    main()
