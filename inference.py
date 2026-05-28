"""Inference for four-patch satellite-guided street-view colorization."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model import SatelliteGuidedDDColor
from src.polar_transform import create_polar_from_satellite
from src.utils import load_config


def load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_panorama(street_rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """Normalize an input panorama to four horizontal 256x256 grayscale patches."""
    resized = cv2.resize(street_rgb, (1024, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    patches = np.stack([gray_rgb[:, i * 256:(i + 1) * 256] for i in range(4)])
    tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float() / 255.0
    return gray_rgb, tensor


@torch.inference_mode()
def colorize_panorama(
    model: SatelliteGuidedDDColor,
    street_rgb: np.ndarray,
    satellite_rgb: np.ndarray,
    device: torch.device,
    polar_size: tuple[int, int] = (256, 512),
) -> dict:
    gray_rgb, gray_patches = prepare_panorama(street_rgb)
    polar_rgb = create_polar_from_satellite(satellite_rgb, output_size=polar_size)
    polar = torch.from_numpy(polar_rgb).permute(2, 0, 1).float().div(255.0)
    polar = polar.unsqueeze(0).repeat(4, 1, 1, 1)
    output = model(gray_patches.to(device), polar.to(device))

    def merge(name: str) -> np.ndarray:
        patches = output[name].detach().cpu().permute(0, 2, 3, 1).numpy()
        return np.clip(np.concatenate(list(patches), axis=1), 0, 1)

    return {
        "gray": gray_rgb,
        "polar": polar_rgb,
        "base": merge("base_rgb"),
        "final": merge("final_rgb"),
    }


def save_rgb_float(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (np.clip(image, 0, 1) * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Inference with Satellite-Guided DDColor")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--street_view", required=True)
    parser.add_argument("--satellite", required=True)
    parser.add_argument("--output", default="outputs/inference/output.jpg")
    parser.add_argument("--show_base", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatelliteGuidedDDColor(
        ddcolor_weights_path=config["ddcolor"]["weights_path"],
        ddcolor_code_path=config["ddcolor"].get("code_path"),
        context_dim=config["model"]["context_dim"],
        polar_encoder_pretrained=config["model"]["polar_encoder_pretrained"],
        correction_type=config["model"]["correction_type"],
        residual_scale=config["model"]["residual_scale"],
        device=str(device),
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    result = colorize_panorama(
        model,
        load_rgb(args.street_view),
        load_rgb(args.satellite),
        device,
        tuple(config["model"]["polar_input_size"]),
    )
    output_path = Path(args.output)
    save_rgb_float(result["final"], output_path)
    if args.show_base:
        save_rgb_float(result["base"], output_path.with_name(output_path.stem + "_base" + output_path.suffix))
    polar_path = output_path.with_name(output_path.stem + "_polar" + output_path.suffix)
    cv2.imwrite(str(polar_path), cv2.cvtColor(result["polar"], cv2.COLOR_RGB2BGR))
    print(f"Final result: {output_path}")
    print(f"Polar context: {polar_path}")


if __name__ == "__main__":
    main()
