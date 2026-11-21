"""
Automated Ablation Study Harness for RayQuery.
Evaluates the trained checkpoint (rayquery_epoch_80.pth) across three core research ablations:
  1. Component Breakdown: Full vs. w/o Uncertainty vs. w/o Epistemic Gate vs. w/o Demand Mask
  2. Uncertainty Penalty Sensitivity: Sweeps alpha in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
  3. Real-World Robustness: GPS Localization Noise in [0.0m, 0.2m, 0.4m, 0.6m, 1.0m]

Outputs:
  - results/ablation/ablation_components.csv
  - results/ablation/ablation_alpha_sensitivity.csv
  - results/ablation/ablation_pose_robustness.csv
  - plots/fig_ablation_components.png & .pdf
  - plots/fig_ablation_alpha.png & .pdf
  - plots/fig_ablation_robustness.png & .pdf
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

def run_eval_pass(model, loader, device, top_k=20, alpha_override=None,
                  use_gate=True, use_demand=True, noise_std=0.0, max_samples=40):
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
            
            out = model(
                batch_data,
                top_k_transmit=top_k,
                alpha_override=alpha_override,
                use_epistemic_gate=use_gate,
                use_demand_guided=use_demand,
                pose_noise_std=noise_std
            )
            
            pred_boxes = out["pred_boxes"].cpu().numpy()
            pred_scores = torch.sigmoid(out["pred_logits"]).squeeze(-1).cpu().numpy()
            gt_boxes = sample["gt_boxes"].numpy()
            
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_gt_boxes.append(gt_boxes)
            total_kb += out["kb_transferred"]
            count += 1
            
    ap_30 = []
    ap_50 = []
    ap_70 = []
    for p_box, p_sc, g_box in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        res = evaluate_ap(p_box, p_sc, g_box, iou_thresholds=(0.3, 0.5, 0.7))
        ap_30.append(res[0.3])
        ap_50.append(res[0.5])
        ap_70.append(res[0.7])
        
    return {
        "bandwidth_kb": float(total_kb / max(1, count)),
        "AP@0.3": float(np.mean(ap_30)) * 100.0,
        "AP@0.5": float(np.mean(ap_50)) * 100.0,
        "AP@0.7": float(np.mean(ap_70)) * 100.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/rayquery_epoch_80.pth")
    parser.add_argument("--max_samples", type=int, default=40)
    args = parser.parse_args()
    
    cfg = RayQueryConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===========================================================")
    print(f" RayQuery Publication Ablation Study Suite")
    print(f" Device: {device} | Checkpoint: {args.checkpoint}")
    print(f"===========================================================\n")
    
    os.makedirs("results/ablation", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    dataset = OPV2VQueryDataset(cfg, split="validate")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    
    model = RayQueryDetector(cfg).to(device)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"[+] Loaded weights from {args.checkpoint}\n")
    else:
        print(f"[!] Checkpoint {args.checkpoint} not found. Running with initialized weights.\n")

    # -------------------------------------------------------------
    # 1. Ablation 1: Component Breakdown (Table 2 in Paper)
    # -------------------------------------------------------------
    print(">>> Running Ablation 1: Component Breakdown...")
    configs_ablation1 = [
        ("Full RayQuery (Ours)", 20, None, True, True),
        ("w/o Depth Uncertainty (alpha=0)", 20, 0.0, True, True),
        ("w/o Epistemic Gating", 20, None, False, True),
        ("w/o Spatial Demand Mask", 20, None, True, False),
        ("Single-Vehicle (No Fusion)", 0, None, True, True),
    ]
    
    results_comp = []
    print(f"{'Variant':<32} | {'BW (KB)':<10} | {'AP@0.3':<10} | {'AP@0.5':<10} | {'AP@0.7':<10}")
    print("-" * 80)
    for name, topk, alpha, gate, demand in configs_ablation1:
        res = run_eval_pass(model, loader, device, top_k=topk, alpha_override=alpha,
                            use_gate=gate, use_demand=demand, max_samples=args.max_samples)
        res["Variant"] = name
        results_comp.append(res)
        print(f"{name:<32} | {res['bandwidth_kb']:<10.2f} | {res['AP@0.3']:<10.2f} | {res['AP@0.5']:<10.2f} | {res['AP@0.7']:<10.2f}", flush=True)
        
    csv_comp = "results/ablation/ablation_components.csv"
    with open(csv_comp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Variant", "bandwidth_kb", "AP@0.3", "AP@0.5", "AP@0.7"])
        w.writeheader()
        w.writerows(results_comp)
    print(f"[+] Saved component ablation CSV: {csv_comp}")

    # Plot Bar Chart for Component Breakdown
    plt.figure(figsize=(9, 4.8), dpi=300)
    names = [r["Variant"].replace(" (Ours)", "") for r in results_comp]
    y_pos = np.arange(len(names))
    ap30_vals = [r["AP@0.3"] for r in results_comp]
    colors = ["#1e40af", "#d97706", "#dc2626", "#059669", "#6b7280"]
    
    bars = plt.barh(y_pos, ap30_vals, color=colors, height=0.6, edgecolor="black", linewidth=0.8)
    plt.yticks(y_pos, names, fontsize=10, fontweight="medium")
    plt.xlabel("Validation 3D Detection AP@0.3 (%)", fontsize=11, fontweight="medium")
    plt.title("Ablation Study: Contribution of Individual RayQuery Components", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.gca().invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%",
                 ha="left", va="center", fontsize=9, fontweight="bold")
                 
    plt.tight_layout()
    plt.savefig("plots/fig_ablation_components.png", dpi=300)
    plt.savefig("plots/fig_ablation_components.pdf")
    plt.close()
    print("[+] Saved component ablation plots (PNG & PDF)")

    # -------------------------------------------------------------
    # 2. Ablation 2: Uncertainty Penalty Sensitivity alpha (Table 3 in Paper)
    # -------------------------------------------------------------
    print("\n>>> Running Ablation 2: Uncertainty Penalty Sensitivity (alpha)...")
    alphas = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    results_alpha = []
    print(f"{'Alpha':<8} | {'AP@0.3':<10} | {'AP@0.5':<10} | {'AP@0.7':<10}")
    print("-" * 45)
    for a in alphas:
        res = run_eval_pass(model, loader, device, top_k=20, alpha_override=a, max_samples=args.max_samples)
        res["alpha"] = a
        results_alpha.append(res)
        print(f"{a:<8.1f} | {res['AP@0.3']:<10.2f} | {res['AP@0.5']:<10.2f} | {res['AP@0.7']:<10.2f}", flush=True)

    csv_alpha = "results/ablation/ablation_alpha_sensitivity.csv"
    with open(csv_alpha, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "bandwidth_kb", "AP@0.3", "AP@0.5", "AP@0.7"])
        w.writeheader()
        w.writerows(results_alpha)

    # Plot Alpha Sensitivity Curve
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(alphas, [r["AP@0.3"] for r in results_alpha], "o-", color="#2563eb", linewidth=2.2, label="AP@0.3")
    plt.plot(alphas, [r["AP@0.5"] for r in results_alpha], "s-", color="#16a34a", linewidth=2.2, label="AP@0.5")
    plt.title(r"Hyperparameter Sensitivity: Uncertainty Penalty Weight ($\alpha$)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel(r"Uncertainty Penalty Parameter $\alpha$", fontsize=11, fontweight="medium")
    plt.ylabel("Average Precision (%)", fontsize=11, fontweight="medium")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/fig_ablation_alpha.png", dpi=300)
    plt.savefig("plots/fig_ablation_alpha.pdf")
    plt.close()
    print("[+] Saved alpha sensitivity plots (PNG & PDF)")

    # -------------------------------------------------------------
    # 3. Ablation 3: GPS / Localization Noise Robustness (Table 4 in Paper)
    # -------------------------------------------------------------
    print("\n>>> Running Ablation 3: GPS Localization Noise Robustness...")
    noises = [0.0, 0.2, 0.4, 0.6, 1.0]
    results_noise = []
    print(f"{'Noise (m)':<12} | {'AP@0.3':<10} | {'AP@0.5':<10} | {'AP@0.7':<10}")
    print("-" * 50)
    for n in noises:
        res = run_eval_pass(model, loader, device, top_k=20, noise_std=n, max_samples=args.max_samples)
        res["noise_m"] = n
        results_noise.append(res)
        print(f"{n:<12.1f} | {res['AP@0.3']:<10.2f} | {res['AP@0.5']:<10.2f} | {res['AP@0.7']:<10.2f}", flush=True)

    csv_noise = "results/ablation/ablation_pose_robustness.csv"
    with open(csv_noise, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["noise_m", "bandwidth_kb", "AP@0.3", "AP@0.5", "AP@0.7"])
        w.writeheader()
        w.writerows(results_noise)

    # Plot Robustness Curve
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(noises, [r["AP@0.3"] for r in results_noise], "^-", color="#dc2626", linewidth=2.2, label="AP@0.3 under GPS Noise")
    plt.title(r"Pose Robustness: 3D Detection under Localization Noise ($\sigma_{\mathrm{pose}}$)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Gaussian Pose Noise Std Dev (m)", fontsize=11, fontweight="medium")
    plt.ylabel("Average Precision (%)", fontsize=11, fontweight="medium")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/fig_ablation_robustness.png", dpi=300)
    plt.savefig("plots/fig_ablation_robustness.pdf")
    plt.close()
    print("[+] Saved pose robustness plots (PNG & PDF)")
    print("\n===========================================================")
    print(" ALL ABLATION STUDIES COMPLETE & FIGURES GENERATED!")
    print("===========================================================")

if __name__ == "__main__":
    main()
