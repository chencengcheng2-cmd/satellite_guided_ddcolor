"""
Residual Color Correction Module.

Learns small color corrections to improve DDColor output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCorrectionModule(nn.Module):
    """
    Residual color correction network.

    Takes base colorized output and context vector, outputs a small
    color correction delta.
    """

    def __init__(
        self,
        context_dim: int = 512,
        base_channels: int = 64,
        residual_scale: float = 0.1,
        use_film: bool = True,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.residual_scale = residual_scale
        self.use_film = use_film

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc1 = self._make_block(base_channels, base_channels * 2)
        self.enc2 = self._make_block(base_channels * 2, base_channels * 4)

        # Context fusion
        if use_film:
            from .film_module import FiLMLayer
            self.film1 = FiLMLayer(context_dim, base_channels * 2)
            self.film2 = FiLMLayer(context_dim, base_channels * 4)
        else:
            self.context_proj = nn.Linear(context_dim, base_channels * 4)

        # Decoder
        self.dec1 = self._make_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_block(base_channels * 2, base_channels)

        # Output projection
        self.output_proj = nn.Conv2d(base_channels, 3, 3, padding=1)

        # Skip connection projection
        self.skip_proj = nn.Conv2d(3, base_channels, 1)

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        base_rgb: torch.Tensor,
        context: torch.Tensor,
        gray_input: torch.Tensor = None,
    ) -> dict:
        """
        Compute residual color correction.

        Args:
            base_rgb: DDColor base output (B, 3, H, W), range [0, 1]
            context: Context vector from polar encoder (B, context_dim)
            gray_input: Optional grayscale input (B, 3, H, W)

        Returns:
            Dictionary with:
                - 'final_rgb': Corrected output
                - 'delta_color': Color correction delta
                - 'features': Intermediate features for debugging
        """
        # Initial projection
        x = self.input_proj(base_rgb)

        # Save skip connection
        skip1 = x

        # Encoder
        x = F.avg_pool2d(x, 2)
        x = self.enc1(x)

        if self.use_film:
            x = self.film1(x, context)

        skip2 = x
        x = F.avg_pool2d(x, 2)
        x = self.enc2(x)

        if self.use_film:
            x = self.film2(x, context)

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + skip2
        x = self.dec1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + skip1
        x = self.dec2(x)

        # Output residual
        delta = self.output_proj(x)

        # Scale the residual
        delta = delta * self.residual_scale

        # Alternatively use tanh to bound the residual
        # delta = torch.tanh(delta) * self.residual_scale

        # Apply residual correction
        final_rgb = base_rgb + delta

        # Clip to valid range
        final_rgb = torch.clamp(final_rgb, 0, 1)

        return {
            'final_rgb': final_rgb,
            'delta_color': delta,
        }


class LightCorrectionModule(nn.Module):
    """
    Lightweight residual correction module for faster training.
    """

    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 64,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_scale = residual_scale

        # Simple network
        # Input: RGB (3) + projected context features (hidden_dim/4 for spatial efficiency)
        context_spatial_dim = hidden_dim // 4
        self.conv1 = nn.Conv2d(3 + context_spatial_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 3, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Context projection to spatial features
        self.context_proj = nn.Linear(context_dim, context_spatial_dim)
        from .film_module import FiLMLayer
        self.film1 = FiLMLayer(context_dim, hidden_dim)
        self.film2 = FiLMLayer(context_dim, hidden_dim)

    def forward(
        self,
        base_rgb: torch.Tensor,
        context: torch.Tensor,
    ) -> dict:
        """
        Compute lightweight residual correction.

        Args:
            base_rgb: DDColor base output (B, 3, H, W)
            context: Context vector (B, context_dim)

        Returns:
            Dictionary with final_rgb and delta_color
        """
        B, C, H, W = base_rgb.shape

        # Project context to spatial features
        context_feat = self.context_proj(context)  # (B, hidden_dim)
        context_feat = context_feat.view(B, -1, 1, 1).expand(B, -1, H, W)

        # Concatenate base RGB with context features
        x = torch.cat([base_rgb, context_feat], dim=1)

        # Process
        x = self.relu(self.film1(self.bn1(self.conv1(x)), context))
        x = self.relu(self.film2(self.bn2(self.conv2(x)), context))
        delta = torch.tanh(self.conv3(x)) * self.residual_scale

        # Apply correction
        final_rgb = torch.clamp(base_rgb + delta, 0, 1)

        return {
            'final_rgb': final_rgb,
            'delta_color': delta,
        }


if __name__ == "__main__":
    # Test Residual Correction Module
    base_rgb = torch.randn(4, 3, 256, 256)
    context = torch.randn(4, 512)

    # Test full module
    correction = ResidualCorrectionModule(context_dim=512)
    result = correction(base_rgb, context)

    print(f"Base RGB shape: {base_rgb.shape}")
    print(f"Final RGB shape: {result['final_rgb'].shape}")
    print(f"Delta shape: {result['delta_color'].shape}")
    print(f"Delta range: [{result['delta_color'].min():.4f}, {result['delta_color'].max():.4f}]")

    # Test lightweight module
    light_correction = LightCorrectionModule(context_dim=512)
    result_light = light_correction(base_rgb, context)
    print(f"\nLightweight module:")
    print(f"Final RGB shape: {result_light['final_rgb'].shape}")
