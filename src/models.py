from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision


class LogitNorm(nn.Module):
    """Normalize logits by their L2 norm and a temperature factor.

    This layer implements the forward transformation:

        y = x / (||x||_2 + eps) / tau

    where the norm is computed over the last dimension.
    """

    def __init__(self, tau: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
        return logits / (norm + self.eps) / self.tau


def apply_logit_norm(logits: torch.Tensor, tau: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
    """Functional LogitNorm helper for use in training or evaluation code."""
    return LogitNorm(tau=tau, eps=eps)(logits)


def get_resnet18_backbone(num_classes: int = 10) -> nn.Module:
    """Construct a ResNet-18 backbone with a configurable classifier head."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class FeatureExtractor(nn.Module):
    """Wrap a classification backbone and expose the penultimate embedding."""

    def __init__(self, original_model: nn.Module) -> None:
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class ResidualMLPBlock(nn.Module):
    """Small residual MLP block used by the trajectory generator."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AdvancedMLP(nn.Module):
    """Compact MLP head for 4D descriptor distillation."""

    def __init__(self, input_dim: int = 512, output_dim: int = 4, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TrajectoryGenerator(nn.Module):
    """Predict a continuous trajectory over training epochs from a static embedding.

    The module maps a 512-dimensional representation to a sequence of length
    ``sequence_length``. It uses a shared MLP trunk followed by a temporal 1D CNN
    head to model smooth epoch-wise dynamics.
    """

    def __init__(
        self,
        input_dim: int = 512,
        sequence_length: int = 50,
        hidden_dim: int = 256,
        temporal_dim: int = 128,
        num_residual_blocks: int = 3,
        dropout: float = 0.1,
        output_channels: int = 1,
        use_logit_norm: bool = False,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if hidden_dim <= 0 or temporal_dim <= 0:
            raise ValueError("hidden_dim and temporal_dim must be positive")
        if num_residual_blocks < 0:
            raise ValueError("num_residual_blocks must be non-negative")
        if output_channels <= 0:
            raise ValueError("output_channels must be positive")

        self.sequence_length = int(sequence_length)
        self.output_channels = int(output_channels)
        self.use_logit_norm = bool(use_logit_norm)
        self.logit_norm = LogitNorm(tau=tau) if use_logit_norm else nn.Identity()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_residual_blocks)]
        )
        self.sequence_projection = nn.Linear(hidden_dim, temporal_dim * self.sequence_length)
        self.temporal_head = nn.Sequential(
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(temporal_dim, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.sequence_projection(x)
        x = x.view(x.shape[0], -1, self.sequence_length)
        x = self.temporal_head(x)
        x = x.squeeze(1) if self.output_channels == 1 else x.transpose(1, 2)
        return self.logit_norm(x)
*** End Patch