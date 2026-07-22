from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LogitNorm(nn.Module):
    """Normalize logits by their L2 norm and a temperature factor."""
    def __init__(self, tau: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(logits, ord=2, dim=-1, keepdim=True)
        return logits / (norm + self.eps) / self.tau


def apply_logit_norm(logits: torch.Tensor, tau: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
    return LogitNorm(tau=tau, eps=eps)(logits)


def get_resnet18_backbone(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class FeatureExtractor(nn.Module):
    def __init__(self, original_model: nn.Module) -> None:
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class ResidualMLPBlock(nn.Module):
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


@dataclass
class TrajectoryCVAEResult:
    generated_trajectory: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    latent: torch.Tensor


class TrajectoryCVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        sequence_length: int = 50,
        num_classes: int = 100,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        output_activation: str = "none",  # Default to "none" for regression tasks.
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")

        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.output_activation = output_activation

        if self.output_activation not in {"none", "softmax"}:
            raise ValueError("output_activation must be either 'none' or 'softmax'")

        trajectory_dim = self.sequence_length * self.num_classes

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + trajectory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden_dim, self.latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim + self.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trajectory_dim),
        )

    def encode(self, features: torch.Tensor, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, input_dim]")
        if trajectory.ndim != 3:
            raise ValueError("trajectory must have shape [batch, sequence_length, num_classes]")

        batch_size = features.shape[0]
        flattened_trajectory = trajectory.reshape(batch_size, -1)
        encoded = self.encoder(torch.cat([features, flattened_trajectory], dim=-1))
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, input_dim]")
        if latent.ndim != 2:
            raise ValueError("latent must have shape [batch, latent_dim]")

        batch_size = features.shape[0]
        decoded = self.decoder(torch.cat([features, latent], dim=-1))
        decoded = decoded.view(batch_size, self.sequence_length, self.num_classes)
        if self.output_activation == "softmax":
            decoded = torch.softmax(decoded, dim=-1)
        return decoded

    def forward(
        self,
        features: torch.Tensor,
        trajectory: Optional[torch.Tensor] = None,
    ) -> TrajectoryCVAEResult:
        if trajectory is not None:
            mu, logvar = self.encode(features, trajectory)
            latent = self.reparameterize(mu, logvar)
        else:
            batch_size = features.shape[0]
            mu = torch.zeros(batch_size, self.latent_dim, device=features.device, dtype=features.dtype)
            logvar = torch.zeros_like(mu)
            latent = torch.randn_like(mu)

        generated_trajectory = self.decode(features, latent)
        return TrajectoryCVAEResult(
            generated_trajectory=generated_trajectory,
            mu=mu,
            logvar=logvar,
            latent=latent,
        )

    def generate(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward(features, trajectory=None).generated_trajectory


@dataclass
class TrajectoryCVAELossOutput:
    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_divergence: torch.Tensor


def trajectory_cvae_loss(
    predicted_trajectory: torch.Tensor,
    target_trajectory: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    reconstruction_loss: str = "smooth_l1",
    kl_weight: float = 1.0,
) -> TrajectoryCVAELossOutput:
    if reconstruction_loss == "smooth_l1":
        recon_loss = F.smooth_l1_loss(predicted_trajectory, target_trajectory)
    elif reconstruction_loss == "mse":
        recon_loss = F.mse_loss(predicted_trajectory, target_trajectory)
    else:
        raise ValueError("reconstruction_loss must be 'smooth_l1' or 'mse'")

    kl_divergence = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + kl_weight * kl_divergence
    return TrajectoryCVAELossOutput(
        total_loss=total_loss,
        reconstruction_loss=recon_loss,
        kl_divergence=kl_divergence,
    )