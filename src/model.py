"""
Main model combining Frozen DDColor, Polar Encoder, FiLM and Residual Correction.
"""

import torch
import torch.nn as nn

from .ddcolor_wrapper import DDColorWrapper
from .polar_encoder import PolarContextEncoder
from .correction_module import ResidualCorrectionModule, LightCorrectionModule


class SatelliteGuidedDDColor(nn.Module):
    """
    Satellite-guided DDColor Enhancement Model.

    Architecture:
    1. Frozen DDColor provides base colorization
    2. Polar Encoder extracts context from satellite view
    3. Residual Correction Module (with optional FiLM) applies corrections
    """

    def __init__(
        self,
        ddcolor_weights_path: str,
        ddcolor_code_path: str = None,
        context_dim: int = 512,
        polar_encoder_pretrained: bool = True,
        correction_type: str = "resnet",  # 'resnet' or 'light'
        residual_scale: float = 0.1,
        polar_input_size: tuple = (256, 512),
        use_film: bool = True,
        device: str = "auto",
    ):
        super().__init__()
        self.context_dim = context_dim
        self.residual_scale = residual_scale
        self.polar_input_size = polar_input_size
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Frozen DDColor
        self.ddcolor = DDColorWrapper(
            model_path=ddcolor_weights_path,
            model_code_path=ddcolor_code_path,
            input_size=256,
            device=self.device,
        )

        # 2. Polar Context Encoder
        self.polar_encoder = PolarContextEncoder(
            context_dim=context_dim,
            pretrained=polar_encoder_pretrained,
            freeze_backbone=False,  # We want to train this
        )

        # 3. Residual Correction Module
        if correction_type == "light":
            self.correction = LightCorrectionModule(
                context_dim=context_dim,
                residual_scale=residual_scale,
            )
        else:
            self.correction = ResidualCorrectionModule(
                context_dim=context_dim,
                residual_scale=residual_scale,
                use_film=use_film,
            )

        # Move to device
        self.to(self.device)

    def forward(
        self,
        gray_rgb: torch.Tensor,
        polar_img: torch.Tensor,
    ) -> dict:
        """
        Forward pass.

        Args:
            gray_rgb: Grayscale image in RGB format (B, 3, H, W), range [0, 1]
            polar_img: Polar satellite view (B, 3, H_p, W_p), range [0, 1]

        Returns:
            Dictionary with:
                - 'base_rgb': DDColor base output
                - 'context_vector': Context from polar encoder
                -final_rgb': Final corrected output
                'delta_color': Color correction delta
        """
        # 1. Get base colorization from DDColor (frozen)
        with torch.no_grad():
            base_rgb = self.ddcolor.colorize(gray_rgb)

        # 2. Extract context from polar image
        context_vector = self.polar_encoder(polar_img)

        # 3. Apply residual correction
        result = self.correction(base_rgb, context_vector)

        return {
            'base_rgb': base_rgb,
            'context_vector': context_vector,
            'final_rgb': result['final_rgb'],
            'delta_color': result['delta_color'],
        }

    def get_trainable_parameters(self):
        """Get list of trainable parameters (excluding DDColor)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_parameters(self) -> dict:
        """Return parameter statistics."""
        ddcolor_total, ddcolor_frozen = self.ddcolor.get_num_parameters()

        trainable_params = []
        frozen_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.numel()))
            else:
                frozen_params.append((name, param.numel()))

        total_trainable = sum(p[1] for p in trainable_params)
        total_frozen = sum(p[1] for p in frozen_params)

        return {
            'ddcolor_total': ddcolor_total,
            'ddcolor_frozen': ddcolor_frozen,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'total_trainable': total_trainable,
            'total_frozen': total_frozen,
            'total': total_trainable + total_frozen,
        }

    def verify_ddcolor_frozen(self) -> bool:
        """Verify DDColor weights are frozen."""
        return self.ddcolor.verify_frozen()
