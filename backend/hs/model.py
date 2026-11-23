import warnings

import torch.nn as nn


def build_model(pretrained=True, num_classes=5):
    """ResNet-18 with a 5-output multi-label head. pretrained=True uses ImageNet weights
    (falls back to random init with a warning if the weights cannot be downloaded)."""
    from torchvision import models

    m = None
    if pretrained:
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:  # no internet / blocked
            warnings.warn(f"Could not load ImageNet weights ({e}); using random init.")
    if m is None:
        m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
