"""
Training loop and utilities.
"""

import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from contextlib import nullcontext

from .loss import TotalLoss
from .metrics import MetricsCalculator
from .utils import (
    save_checkpoint, load_checkpoint, count_trainable_parameters,
    AverageMeter, set_seed, get_output_dir, get_checkpoint_dir, get_device,
)


class Trainer:
    """Trainer for Satellite-Guided DDColor."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        exp_name: str = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup directories
        if exp_name:
            self.exp_name = exp_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_name = f"exp_{timestamp}"

        self.output_dir = get_output_dir(self.exp_name, config.get('output_base_dir', 'outputs'))
        self.checkpoint_dir = get_checkpoint_dir(self.exp_name, config.get('checkpoint_base_dir', 'checkpoints'))

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir / "visualizations", exist_ok=True)
        os.makedirs(self.output_dir / "logs", exist_ok=True)

        # Setup device
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Setup loss
        self.criterion = TotalLoss(
            l1_weight=config.get('l1_weight', 1.0),
            perceptual_weight=config.get('perceptual_weight', 0.5),
            residual_weight=config.get('residual_weight', 0.1),
            use_perceptual=config.get('use_perceptual', True),
        ).to(self.device)

        # Setup optimizer (only trainable parameters)
        trainable_params = model.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # Verify DDColor is not in optimizer
        ddcolor_params = set(id(p) for p in model.ddcolor.parameters())
        optimizer_params = set(id(p) for group in self.optimizer.param_groups for p in group['params'])
        if ddcolor_params & optimizer_params:
            raise RuntimeError("Optimizer contains DDColor parameters! DDColor must be frozen.")

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 30),
            eta_min=config.get('min_lr', 1e-6),
        )

        # Mixed precision
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Metrics
        self.metrics_calc = MetricsCalculator(
            device=str(self.device),
            enable_lpips=config.get('validate_lpips', False),
            enable_fid=config.get('validate_fid', False),
        )

        # Training state
        self.current_epoch = 0
        self.best_psnr = 0
        self.global_step = 0
        self.start_epoch = 0

        # Load checkpoint if specified
        if config.get('resume_from'):
            self.load_checkpoint(config['resume_from'])

        # Set seed
        set_seed(config.get('seed', 42))

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        """Print model parameter information."""
        stats = self.model.get_num_parameters()

        print("=" * 50)
        print("Model Parameters")
        print("=" * 50)
        print(f"DDColor total:      {stats['ddcolor_total']:,}")
        print(f"DDColor frozen:     {stats['ddcolor_frozen']:,}")
        print(f"Trainable:          {stats['total_trainable']:,}")
        print(f"Frozen:             {stats['total_frozen']:,}")
        print(f"Total:              {stats['total']:,}")
        print(f"DDColor frozen:     {self.model.verify_ddcolor_frozen()}")
        print("=" * 50)

        print("\nTrainable modules:")
        for name, num in stats['trainable_params']:
            print(f"  {name}: {num:,}")

        print("=" * 50)

    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            gray_rgb = batch['gray'].to(self.device)
            rgb_gt = batch['rgb'].to(self.device)
            polar = batch['polar'].to(self.device)

            # Forward pass
            amp_context = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_context:
                output = self.model(gray_rgb, polar)
                loss_dict = self.criterion(output['final_rgb'], rgb_gt, output['delta_color'])
                loss = loss_dict['total']

            # Check for NaN
            if not torch.isfinite(loss):
                print(f"NaN loss at step {self.global_step}!")
                print(f"Loss dict: {loss_dict}")
                continue

            # Backward pass
            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            losses.update(loss.item(), gray_rgb.size(0))

            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                print(
                    f"Epoch [{self.current_epoch}/{self.config['epochs']}][{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {losses.avg:.4f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            self.global_step += 1
            if self.config.get('max_train_batches') and batch_idx + 1 >= self.config['max_train_batches']:
                break

        return {'total_loss': losses.avg}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """Validate the model."""
        self.model.eval()

        psnr_values = []
        ssim_values = []
        lpips_values = []

        if self.metrics_calc.fid:
            self.metrics_calc.fid.reset()

        for batch_idx, batch in enumerate(self.val_loader):
            gray_rgb = batch['gray'].to(self.device)
            rgb_gt = batch['rgb'].to(self.device)
            polar = batch['polar'].to(self.device)

            # Forward
            amp_context = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_context:
                output = self.model(gray_rgb, polar)

            # Compute metrics
            metrics = self.metrics_calc.compute_batch(output['final_rgb'], rgb_gt)

            if metrics['psnr'] != float('inf'):
                psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
            if metrics['lpips'] is not None:
                lpips_values.append(metrics['lpips'])
            if self.config.get('max_val_batches') and batch_idx + 1 >= self.config['max_val_batches']:
                break

        # Compute average metrics
        results = {
            'psnr': np.mean(psnr_values) if psnr_values else 0,
            'ssim': np.mean(ssim_values) if ssim_values else 0,
            'lpips': np.mean(lpips_values) if lpips_values else None,
        }

        # Get FID
        fid_val = self.metrics_calc.get_fid()
        if fid_val is not None:
            results['fid'] = fid_val

        print(f"Validation - PSNR: {results['psnr']:.2f}, SSIM: {results['ssim']:.4f}")
        if results.get('lpips') is not None:
            print(f"Validation - LPIPS: {results['lpips']:.4f}")
        if results.get('fid') is not None:
            print(f"Validation - FID: {results['fid']:.2f}")

        return results

    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{self.current_epoch}.pth"

        metrics = {
            'best_psnr': self.best_psnr,
            'global_step': self.global_step,
        }

        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.current_epoch,
            metrics,
            str(checkpoint_path),
            is_best=is_best,
        )

        # Also save latest
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.current_epoch,
            metrics,
            str(self.checkpoint_dir / "latest.pth"),
            is_best=False,
        )

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        epoch, metrics = load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        self.start_epoch = epoch + 1
        self.current_epoch = epoch
        self.best_psnr = metrics.get('best_psnr', 0)
        self.global_step = metrics.get('global_step', 0)
        print(f"Resumed from epoch {epoch}, best PSNR: {self.best_psnr:.2f}")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        print(f"Outputs will be saved to: {self.output_dir}")
        print()

        start_time = time.time()

        for epoch in range(self.start_epoch, self.config['epochs']):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate(epoch)

                # Update best PSNR
                if val_metrics['psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['psnr']
                    print(f"New best PSNR: {self.best_psnr:.2f}")
                    is_best = True
                else:
                    is_best = False

                # Save checkpoint
                if (epoch + 1) % self.config.get('save_interval', 1) == 0:
                    self.save_checkpoint(is_best=is_best)

            # Update learning rate
            self.scheduler.step()

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 3600:.2f} hours")
        print(f"Best PSNR: {self.best_psnr:.2f}")
