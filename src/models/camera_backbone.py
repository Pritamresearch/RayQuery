"""
Multi-Camera Feature Extractor and Categorical Depth Probability Estimator.
Extracts 2D image features and predicts depth distributions for estimating ray uncertainty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraDepthBackbone(nn.Module):
    def __init__(self, feat_dim: int = 128, d_bins: int = 50):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_bins = d_bins
        
        # Lightweight convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # -> H/2, W/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> H/4, W/4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> H/8, W/8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Head that outputs: Feature Map (feat_dim) + Categorical Depth Distribution (d_bins)
        self.depth_head = nn.Conv2d(128, d_bins, kernel_size=1)
        self.feat_head = nn.Conv2d(128, feat_dim, kernel_size=1)
        
    def forward(self, images: torch.Tensor):
        """
        Input:
            images: (B * Ncam, 3, H, W)
        Returns:
            features: (B * Ncam, feat_dim, H/8, W/8)
            depth_prob: (B * Ncam, d_bins, H/8, W/8) - Softmax distribution over depth bins
            depth_entropy: (B * Ncam, 1, H/8, W/8) - Normalized entropy representing ray ambiguity
        """
        x = self.encoder(images)
        features = self.feat_head(x)
        depth_logits = self.depth_head(x)
        depth_prob = F.softmax(depth_logits, dim=1)
        
        # Calculate depth entropy along the ray (higher entropy = higher depth uncertainty)
        eps = 1e-7
        log_prob = torch.log(depth_prob + eps)
        entropy = -torch.sum(depth_prob * log_prob, dim=1, keepdim=True)
        # Normalize entropy between [0, 1]
        max_entropy = torch.log(torch.tensor(float(self.d_bins), device=images.device))
        norm_entropy = entropy / (max_entropy + eps)
        
        return features, depth_prob, norm_entropy
