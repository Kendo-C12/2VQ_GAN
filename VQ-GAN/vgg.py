# Global VGG variables
vgg_norm = None
vgg = None
use_vgg = False

import torch.nn.functional as F
import torch

def setup_vgg(device):
    """Initialize VGG for perceptual loss"""
    try:
        from torchvision.models import vgg16
        from torchvision import transforms
        vgg = vgg16(pretrained=True).features[:16].to(device).eval()
        for p in vgg.parameters(): 
            p.requires_grad = False
        vgg_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        use_vgg = True
        print("✓ VGG perceptual loss enabled")
        return True
    except Exception as e:
        print(f"✗ VGG not available: {e}")
        use_vgg = False
        return False

def compute_vgg_loss(x, y):
    if not use_vgg: 
        return torch.tensor(0.0, device=x.device)
    x_in = vgg_norm(x)
    y_in = vgg_norm(y)
    feat_x = vgg(x_in)
    feat_y = vgg(y_in)
    return F.l1_loss(feat_x, feat_y)
