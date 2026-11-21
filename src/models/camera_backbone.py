"""
Pretrained ResNet-18 Multi-Camera Backbone with Ray-Depth Probability Estimator.
Replaces scratch convnet with ImageNet-pretrained ResNet-18 features,
drastically boosting visual representation quality for 3D object detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CameraDepthBackbone(nn.Module):
    def __init__(self, feat_dim: int = 128, d_bins: int = 50, pretrained: bool = True):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_bins = d_bins
        
        # Load ImageNet-pretrained ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            base_resnet = models.resnet18(weights=weights)
        else:
            base_resnet = models.resnet18(weights=None)
            
        # ResNet-18 early stages: stride 8 output (H/8, W/8)
        self.conv1 = base_resnet.conv1      # 7x7 conv, stride 2
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool  # stride 2 -> stride 4
        self.layer1 = base_resnet.layer1    # stride 4, 64 channels
        self.layer2 = base_resnet.layer2    # stride 8, 128 channels
        
        # Feature adaptation neck
        self.neck = nn.Sequential(
            nn.Conv2d(128, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        
        # Multi-task heads: Feature embedding + Categorical depth distribution
        self.feat_head = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        self.depth_head = nn.Conv2d(feat_dim, d_bins, kernel_size=1)
        
    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (Ncam, 3, H, W) normalized camera images
        Returns:
            features: (Ncam, feat_dim, H/8, W/8)
            depth_prob: (Ncam, d_bins, H/8, W/8)
            norm_entropy: (Ncam, 1, H/8, W/8) - Ray depth uncertainty in [0, 1]
        """
        # ResNet forward pass
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.neck(x)
        features = self.feat_head(x)
        depth_logits = self.depth_head(x)
        depth_prob = F.softmax(depth_logits, dim=1)
        
        # Compute normalized depth entropy along the projection ray
        eps = 1e-7
        log_prob = torch.log(depth_prob + eps)
        entropy = -torch.sum(depth_prob * log_prob, dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(self.d_bins), device=images.device))
        norm_entropy = entropy / (max_entropy + eps)
        
        return features, depth_prob, norm_entropy
