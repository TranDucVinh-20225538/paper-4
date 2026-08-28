"""
Stage 4 adapter module + losses. Locked architecture, not re-derived here:
Stage4_Adapter_And_Control_Specification.md SS1-2.

z' = z + W2 . GELU(W1 . z)              (residual bottleneck, W1: 1024->r, W2: r->1024)
L_orth(z', w) = cos(z', w)^2            (squared cosine similarity vs the fixed nuisance direction)
L_task = CrossEntropy(head(z'), y)      (ISIC rows only, y == -1 ignored -- CSG-Skin's own convention)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 1024
NUM_CLASSES = 8


class BottleneckAdapter(nn.Module):
    """z' = z + W2 . GELU(W1 . z). Stage4_Adapter_And_Control_Specification.md SS1."""

    def __init__(self, dim: int = EMBED_DIM, r: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, r)
        self.act = nn.GELU(approximate="none")  # matches PanDerm's own activation, SS1
        self.up = nn.Linear(r, dim)
        # zero-init the up-projection so the adapter starts as an identity map (delta_z = 0 at init)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def delta(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(z)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.delta(z)


def orthogonality_loss(z_adapted: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """L_orth = mean(cos(z', w)^2). Stage4_Adapter_And_Control_Specification.md SS2."""
    cos = F.cosine_similarity(z_adapted, w.unsqueeze(0).expand_as(z_adapted), dim=1)
    return (cos ** 2).mean()


def task_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CE on rows with a valid label; y == -1 rows (PAD-UFES) ignored, CSG-Skin's own convention."""
    return F.cross_entropy(logits, y, ignore_index=-1)


class TaskHead(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
