"""
Evaluation Metrics for 3D Cooperative Perception.
Computes 3D Bounding Box Average Precision (AP@0.3, AP@0.5, AP@0.7 IoU).
"""
import torch
import numpy as np

def compute_bev_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Approximate 2D BEV IoU between two oriented bounding boxes.
    box: [x, y, z, l, w, h, yaw]
    """
    x1, y1, _, l1, w1, _, _ = box1
    x2, y2, _, l2, w2, _, _ = box2
    
    # Axis-aligned bounding envelope approximation for fast evaluation
    left = max(x1 - l1 / 2, x2 - l2 / 2)
    right = min(x1 + l1 / 2, x2 + l2 / 2)
    bottom = max(y1 - w1 / 2, y2 - w2 / 2)
    top = min(y1 + w1 / 2, y2 + w2 / 2)
    
    inter_w = max(0.0, right - left)
    inter_h = max(0.0, top - bottom)
    inter_area = inter_w * inter_h
    
    area1 = l1 * w1
    area2 = l2 * w2
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def evaluate_ap(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                gt_boxes: np.ndarray, iou_thresholds=(0.3, 0.5, 0.7)):
    """
    Computes Average Precision across IoU thresholds.
    """
    if len(gt_boxes) == 0:
        return {th: 1.0 if len(pred_boxes) == 0 else 0.0 for th in iou_thresholds}
    if len(pred_boxes) == 0:
        return {th: 0.0 for th in iou_thresholds}
        
    # Sort predictions by descending confidence score
    sort_idx = np.argsort(-pred_scores)
    sorted_preds = pred_boxes[sort_idx]
    
    results = {}
    for th in iou_thresholds:
        matched_gt = set()
        tp = np.zeros(len(sorted_preds))
        fp = np.zeros(len(sorted_preds))
        
        for i, pred in enumerate(sorted_preds):
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = compute_bev_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    
            if best_iou >= th and best_gt_idx >= 0:
                tp[i] = 1.0
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1.0
                
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / len(gt_boxes)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
        
        # 11-point interpolated AP
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            p = np.max(precisions[recalls >= r]) if np.any(recalls >= r) else 0.0
            ap += p / 11.0
        results[th] = float(ap)
        
    return results
