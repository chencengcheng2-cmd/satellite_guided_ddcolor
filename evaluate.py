"""
Evaluate model on test set.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CVUSADataset
from src.model import SatelliteGuidedDDColor
from src.metrics import MetricsCalculator
from src.utils import load_config


def collate_fn(batch):
    """Custom collate function."""
    gray = torch.stack([
        torch.from_numpy(b['gray']).unsqueeze(0).repeat(3, 1, 1).float() for b in batch
    ])
    rgb = torch.stack([torch.from_numpy(b['rgb']).permute(2, 0, 1).float() for b in batch])

    if batch[0]['polar'] is not None:
        polar = torch.stack([torch.from_numpy(b['polar']).permute(2, 0, 1).float() for b in batch])
    else:
        polar = None

    return {
        'gray': gray,
        'rgb': rgb,
        'polar': polar,
        'file_id': [b['file_id'] for b in batch],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Evaluate Satellite-Guided DDColor')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results.json')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = SatelliteGuidedDDColor(
        ddcolor_weights_path=config['ddcolor']['weights_path'],
        ddcolor_code_path=config.get('ddcolor', {}).get('code_path'),
        context_dim=config['model']['context_dim'],
        polar_encoder_pretrained=config['model']['polar_encoder_pretrained'],
        correction_type=config['model']['correction_type'],
        residual_scale=config['model']['residual_scale'],
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()

    # Load dataset
    print(f"Loading {args.split} set...")
    dataset = CVUSADataset(
        config['dataset']['root'],
        split=args.split,
        load_polar=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No paired samples are available in split '{args.split}'. "
            "Use --split val until the test files are repaired or regenerated."
        )

    if args.num_samples:
        dataset = torch.utils.data.Subset(dataset, list(range(args.num_samples)))

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Metrics calculator
    base_calc = MetricsCalculator(device=device)
    final_calc = MetricsCalculator(device=device)

    # Results storage
    results = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(dataset),
        'base_metrics': {},
        'final_metrics': {},
    }

    # Evaluate
    print("\nEvaluating...")

    for batch in tqdm(dataloader):
        gray = batch['gray'].to(device)
        rgb_gt = batch['rgb'].to(device)
        polar = batch['polar'].to(device)

        # Resize polar if needed
        polar_h, polar_w = polar.shape[2:]
        target_h, target_w = config['model']['polar_input_size']
        if polar_h != target_h or polar_w != target_w:
            import torch.nn.functional as F
            polar = F.interpolate(polar, size=(target_h, target_w), mode='bilinear')

        # Forward
        output = model(gray, polar)

        # Compute metrics for base
        base_metrics = base_calc.compute_batch(output['base_rgb'], rgb_gt, accumulate_fid=True)

        # Compute metrics for final
        final_metrics = final_calc.compute_batch(output['final_rgb'], rgb_gt, accumulate_fid=True)

        # Accumulate
        for k, v in base_metrics.items():
            if v is not None and v != float('inf'):
                results['base_metrics'].setdefault(k, []).append(v)

        for k, v in final_metrics.items():
            if v is not None and v != float('inf'):
                results['final_metrics'].setdefault(k, []).append(v)

    # Compute averages
    for key in results['final_metrics']:
        results['final_metrics'][key] = float(np.mean(results['final_metrics'][key]))
    for key in results['base_metrics']:
        results['base_metrics'][key] = float(np.mean(results['base_metrics'][key]))

    base_fid = base_calc.get_fid()
    final_fid = final_calc.get_fid()
    if base_fid is not None:
        results['base_metrics']['fid'] = float(base_fid)
    if final_fid is not None:
        results['final_metrics']['fid'] = float(final_fid)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")
    print()

    print("DDColor Base:")
    for k, v in results['base_metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\nSatellite-Guided (Final):")
    for k, v in results['final_metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\nImprovement:")
    for k in results['base_metrics']:
        base = results['base_metrics'].get(k, 0)
        final = results['final_metrics'].get(k, 0)
        if k in ('lpips', 'fid'):
            improv = base - final  # Lower is better for LPIPS
        else:
            improv = final - base
        print(f"  {k}: {improv:+.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
