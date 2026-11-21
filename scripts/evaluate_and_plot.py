"""
Evaluation & Accuracy-Bandwidth Sweeper for RayQuery.
Loads a trained checkpoint, evaluates on validation/test sets across multiple
communication budgets (Top-K query capacity), and generates publication-grade figures.

Outputs:
  - results/curves/accuracy_bandwidth.csv
  - plots/fig_accuracy_bandwidth.png
  - plots/fig_accuracy_bandwidth.pdf
"""
import os
import sys
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import RayQueryConfig
from src.data.opv2v_dataset import OPV2VQueryDataset
from src.models.rayquery_model import RayQueryDetector
from src.utils.metrics import evaluate_ap

def evaluate_budget(model, loader, device, top_k: int, max_samples: int = None):
    """
    Evaluates the model over the dataset for a specific top-K query transmission budget.
    """
    model.eval()
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    total_kb = 0.0
    count = 0
    
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if max_samples is not None and i >= max_samples:
                break
                
            batch_data = {
                "n_agents": sample["n_agents"],
                "images": sample["images"].to(device),
                "intrinsics": sample["intrinsics"].to(device),
                "extrinsics": sample["extrinsics"].to(device),
                "rel_poses": sample["rel_poses"].to(device),
                "gt_boxes": sample["gt_boxes"].to(device)
            }
            
            out = model(batch_data, top_k_transmit=top_k)
            
            pred_boxes = out["pred_boxes"].cpu().numpy()
            pred_scores = torch.sigmoid(out["pred_logits"]).squeeze(-1).cpu().numpy()
            gt_boxes = sample["gt_boxes"].numpy()
            
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_gt_boxes.append(gt_boxes)
            total_kb += out["kb_transferred"]
            count += 1
            
    # Calculate dataset-wide AP
    ap_30_list = []
    ap_50_list = []
    ap_70_list = []
    
    for p_box, p_sc, g_box in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        res = evaluate_ap(p_box, p_sc, g_box, iou_thresholds=(0.3, 0.5, 0.7))
        ap_30_list.append(res[0.3])
        ap_50_list.append(res[0.5])
        ap_70_list.append(res[0.7])
        
    mean_ap30 = float(np.mean(ap_30_list)) if ap_30_list else 0.0
    mean_ap50 = float(np.mean(ap_50_list)) if ap_50_list else 0.0
    mean_ap70 = float(np.mean(ap_70_list)) if ap_70_list else 0.0
    mean_kb = float(total_kb / max(1, count))
    
    return {
        "top_k": top_k,
        "kb_per_frame": mean_kb,
        "AP@0.3": mean_ap30 * 100.0,
        "AP@0.5": mean_ap50 * 100.0,
        "AP@0.7": mean_ap70 * 100.0
    }

def plot_curves(records, out_png: str, out_pdf: str):
    """
    Renders publication-grade Accuracy vs. Bandwidth plots.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    kb = [r["kb_per_frame"] for r in records]
    ap30 = [r["AP@0.3"] for r in records]
    ap50 = [r["AP@0.5"] for r in records]
    ap70 = [r["AP@0.7"] for r in records]
    
    plt.figure(figsize=(7.5, 5.0), dpi=300)
    plt.plot(kb, ap30, "o-", color="#1f77b4", linewidth=2.2, markersize=7, label="RayQuery (AP@0.3)")
    plt.plot(kb, ap50, "s-", color="#2ca02c", linewidth=2.2, markersize=7, label="RayQuery (AP@0.5)")
    plt.plot(kb, ap70, "^-", color="#d62728", linewidth=2.2, markersize=7, label="RayQuery (AP@0.7)")
    
    plt.title("Camera-Only V2V Perception: 3D Detection vs. Bandwidth", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Transmission Bandwidth (KB / frame)", fontsize=11, fontweight="medium")
    plt.ylabel("Average Precision (%)", fontsize=11, fontweight="medium")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    
    # Annotate Pareto operating sweet-spot
    if len(kb) > 2:
        sweet_idx = 2  # top-k = 20 or 30
        plt.annotate(
            f"Ultra-Low BW Point\n({kb[sweet_idx]:.1f} KB, AP@0.5={ap50[sweet_idx]:.1f}%)",
            xy=(kb[sweet_idx], ap50[sweet_idx]),
            xytext=(kb[sweet_idx] + 2.0, ap50[sweet_idx] - 8.0),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
            fontsize=9,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2cc", edgecolor="#d6b656", alpha=0.9)
        )
        
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"[+] Plots successfully saved:\n    -> {out_png}\n    -> {out_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/rayquery_epoch_40.pth",
                        help="Path to trained checkpoint")
    parser.add_argument("--split", type=str, default="validate", choices=["validate", "test"])
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Number of scenes to evaluate per budget point (default 50 for quick sweep)")
    args = parser.parse_args()
    
    cfg = RayQueryConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"===========================================================")
    print(f" RayQuery Accuracy vs. Bandwidth Evaluation")
    print(f" Device: {device} | Checkpoint: {args.checkpoint}")
    print(f"===========================================================\n")
    
    dataset = OPV2VQueryDataset(cfg, split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    print(f"Loaded {len(dataset)} scenes from '{args.split}' split.")
    
    # Initialize and load model
    model = RayQueryDetector(cfg).to(device)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Loaded weights from: {args.checkpoint}\n")
    else:
        print(f"[!] Warning: Checkpoint '{args.checkpoint}' not found! Using initialized weights.")
        
    # Sweeping budgets: Top-K queries per collaborator
    # 0 = No Fusion / Single-Vehicle baseline
    budgets = [0, 5, 10, 20, 30, 50, 100]
    records = []
    
    print(f"{'Top-K':<8} | {'Bandwidth (KB)':<16} | {'AP@0.3 (%)':<12} | {'AP@0.5 (%)':<12} | {'AP@0.7 (%)':<12}")
    print("-" * 68)
    
    for k in budgets:
        res = evaluate_budget(model, loader, device, top_k=k, max_samples=args.max_samples)
        records.append(res)
        print(f"{res['top_k']:<8} | {res['kb_per_frame']:<16.2f} | {res['AP@0.3']:<12.2f} | {res['AP@0.5']:<12.2f} | {res['AP@0.7']:<12.2f}", flush=True)
        
    # Save CSV
    os.makedirs("results/curves", exist_ok=True)
    csv_path = "results/curves/accuracy_bandwidth.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["top_k", "kb_per_frame", "AP@0.3", "AP@0.5", "AP@0.7"])
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[+] Numerical results saved to: {csv_path}")
    
    # Generate Figures
    png_path = "plots/fig_accuracy_bandwidth.png"
    pdf_path = "plots/fig_accuracy_bandwidth.pdf"
    plot_curves(records, png_path, pdf_path)

if __name__ == "__main__":
    main()
