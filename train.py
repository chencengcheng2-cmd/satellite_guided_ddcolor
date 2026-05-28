"""
Training script for Satellite-Guided DDColor.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import CVUSADataset, create_dataset_report
from src.model import SatelliteGuidedDDColor
from src.trainer import Trainer
from src.utils import load_config, set_seed, create_timestamp_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train Satellite-Guided DDColor')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no_ddcolor', action='store_true',
                        help='Skip DDColor and use dummy model for testing')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run a short smoke test')
    parser.add_argument('--quick_train', action='store_true',
                        help='Run two train and two validation batches to verify the full pipeline')
    return parser.parse_args()


def collate_fn(batch):
    """Custom collate function for the dataset."""
    gray = torch.stack([
        torch.from_numpy(b['gray']).unsqueeze(2).repeat(1, 1, 3).permute(2, 0, 1).float()
        for b in batch
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


def smoke_test(config):
    """Run a smoke test to verify everything works."""
    print("\n" + "=" * 50)
    print("Running smoke test...")
    print("=" * 50)

    # Test dataset
    print("\n1. Testing dataset...")
    dataset_root = config['dataset']['root']
    report = create_dataset_report(dataset_root, "outputs/cache/smoke_dataset_report.json")
    print(f"Dataset report saved to outputs/cache/smoke_dataset_report.json")

    train_set = CVUSADataset(dataset_root, split="train", load_polar=True)
    val_set = CVUSADataset(dataset_root, split="val", load_polar=True)

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    # Test loading
    sample = train_set[0]
    polar_shape = sample['polar'].shape if sample['polar'] is not None else 'N/A'
    print(f"Sample shapes:")
    print(f"  RGB: {sample['rgb'].shape}")
    print(f"  Gray: {sample['gray'].shape}")
    print(f"  Polar: {polar_shape}")

    # Test dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Gray: {batch['gray'].shape}")
    print(f"  RGB: {batch['rgb'].shape}")
    print(f"  Polar: {batch['polar'].shape}")

    print("\n" + "=" * 50)
    print("Smoke test passed!")
    print("=" * 50)


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.resume:
        config['resume_from'] = args.resume
    if args.smoke_test:
        config['epochs'] = 2
        config['log_interval'] = 1
    if args.quick_train:
        config['training']['epochs'] = 1
        config['training']['max_train_batches'] = 2
        config['training']['max_val_batches'] = 2
        config['training']['log_interval'] = 1
        config['loss']['use_perceptual'] = False

    # Set seed
    set_seed(config.get('training', {}).get('seed', 42))

    # Run smoke test first
    smoke_test(config)

    if args.smoke_test and not args.quick_train:
        print("\n" + "=" * 50)
        print("SMOKE TEST COMPLETE")
        print("=" * 50)
        return

    # Check DDColor weights
    ddcolor_weights = config.get('ddcolor', {}).get('weights_path')
    if not ddcolor_weights or not os.path.exists(ddcolor_weights):
        print(f"\nError: DDColor weights not found at {ddcolor_weights}")
        print("Please download ddcolor_paper_tiny weights and update config.yaml")
        return

    # Create dataloaders
    dataset_root = config['dataset']['root']
    train_set = CVUSADataset(dataset_root, split="train", load_polar=True)
    val_set = CVUSADataset(dataset_root, split="val", load_polar=True)

    train_loader = DataLoader(
        train_set,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Create model
    print("\n" + "=" * 50)
    print("Loading model...")
    print("=" * 50)

    try:
        model = SatelliteGuidedDDColor(
            ddcolor_weights_path=config['ddcolor']['weights_path'],
            ddcolor_code_path=config.get('ddcolor', {}).get('code_path'),
            context_dim=config['model']['context_dim'],
            polar_encoder_pretrained=config['model']['polar_encoder_pretrained'],
            correction_type=config['model']['correction_type'],
            residual_scale=config['model']['residual_scale'],
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create trainer
    training_config = {
        **config['training'],
        **config['loss'],
        'output_base_dir': config['paths']['output_base_dir'],
        'checkpoint_base_dir': config['paths']['checkpoint_base_dir'],
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        exp_name=args.exp_name,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    from pathlib import Path
    main()
