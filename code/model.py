"""Compact residual CNN for SynthShapes-32.

Architecture: stem -> 3 residual stages (16,32,64) -> GAP -> linear.
The global-average-pooling head makes the last convolutional block a natural
target for Grad-CAM. Kept small to train on a single CPU core in minutes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.sc(x))


class SmallResNet(nn.Module):
    def __init__(self, num_classes=6, widths=(16, 32, 64)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        self.stage1 = BasicBlock(widths[0], widths[0], 1)
        self.stage2 = BasicBlock(widths[0], widths[1], 2)
        self.stage3 = BasicBlock(widths[1], widths[2], 2)
        self.head = nn.Linear(widths[2], num_classes)
        self._feat = None  # cache for Grad-CAM

    def forward(self, x, return_feat=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        f = self.stage3(x)          # last conv feature map (B,C,H',W')
        if f.requires_grad:
            f.retain_grad()
        self._feat = f
        g = F.adaptive_avg_pool2d(f, 1).flatten(1)
        logits = self.head(g)
        if return_feat:
            return logits, f
        return logits


def count_params(m):
    return sum(p.numel() for p in m.parameters())
