"""
PyTorch-based training script for DWTS elimination model.
Supports GPU acceleration with advanced training features:
- Early stopping
- Cosine annealing learning rate
- Real-time logging
- Gradient clipping
- Model checkpointing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import numpy as np

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task1.config import load_config
from task1.types import RuleParams


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 100, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


class TrainingLogger:
    """Real-time training logger with JSON and text output."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_log = log_dir / "train.log"
        self.json_log = log_dir / "train_metrics.jsonl"
        self.history = {"elbo": [], "log_lik": [], "kl": [], "lr": [], "time": []}
        
        # Clear previous logs
        self.text_log.write_text("", encoding="utf-8")
        self.json_log.write_text("", encoding="utf-8")
        
    def log(self, msg: str):
        """Log text message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg, flush=True)
        with open(self.text_log, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")
    
    def log_metrics(self, epoch: int, elbo: float, log_lik: float, kl: float, 
                    lr: float, elapsed: float, extra: dict = None):
        """Log training metrics."""
        self.history["elbo"].append(elbo)
        self.history["log_lik"].append(log_lik)
        self.history["kl"].append(kl)
        self.history["lr"].append(lr)
        self.history["time"].append(elapsed)
        
        # Write to JSONL for real-time monitoring
        metrics = {
            "epoch": epoch,
            "elbo": elbo,
            "log_lik": log_lik,
            "kl": kl,
            "lr": lr,
            "time": elapsed
        }
        if extra:
            metrics.update(extra)
            
        with open(self.json_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def save_history(self, path: Path):
        """Save complete training history."""
        np.savez(path, **{k: np.array(v) for k, v in self.history.items()})


def main():
    parser = argparse.ArgumentParser(description="Train DWTS model with PyTorch (GPU supported)")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--data", type=str, default=None, help="Path to data CSV")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.data:
        cfg.paths.data_csv = args.data
    if args.out:
        cfg.paths.output_dir = args.out

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Import data processing modules
    from task1.io.load_csv import load_csv
    from task1.data.reshape import wide_to_long
    from task1.data.build_masks import build_masks
    from task1.data.features import build_features
    from task1.data.tensors import build_tensors, build_week_obs_list
    from task1.model.torch_model import build_torch_model

    # Setup output directory
    out_dir = Path(cfg.paths.output_dir) / "task1" / "run_torch"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = TrainingLogger(out_dir)
    
    logger.log(f"PyTorch DWTS Training")
    logger.log(f"Device: {device}")
    logger.log("=" * 60)

    # Load and preprocess data
    logger.log("Loading data...")
    df = load_csv(cfg.paths.data_csv)
    logger.log(f"Loaded {len(df)} rows")

    logger.log("Preprocessing...")
    long_df = wide_to_long(df, max_week=cfg.data.max_week)
    long_df, valid_df = build_masks(long_df, withdrawal_policy=cfg.data.withdrawal_policy, 
                                     multi_elim_policy=cfg.data.multi_elim_policy)
    feats_df, feature_cols = build_features(long_df)
    tensors = build_tensors(long_df, valid_df, feats_df, feature_cols, 
                           cfg.rules.season_rules, cfg.data.max_week)
    week_obs = build_week_obs_list(tensors)
    
    logger.log(f"Seasons: {len(tensors.seasons)}, Max contestants: {tensors.N_max}, Max weeks: {tensors.T_max}")
    logger.log(f"Features: {len(feature_cols)}")
    logger.log(f"Valid observations: {len([o for o in week_obs if o.valid])}")

    # Build model
    logger.log("Building model...")
    model = build_torch_model(tensors, cfg, device=device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model parameters: {n_params:,}")

    # Setup rule parameters
    params = RuleParams(
        epsilon=cfg.hyper.epsilon,
        delta=cfg.hyper.delta,
        tau=cfg.hyper.tau,
        kappa=cfg.hyper.kappa,
        kappa_r=cfg.hyper.kappa_r,
        alpha=cfg.hyper.alpha,
        kappa_b=cfg.hyper.kappa_b,
        eta1=cfg.hyper.eta1,
        eta2=cfg.hyper.eta2,
        lambda_ent=cfg.hyper.lambda_ent,
        bottom2_base=cfg.rules.bottom2_base,
        loglik_mode=cfg.rules.loglik_mode,
    )

    season_idx = {s: i for i, s in enumerate(tensors.seasons)}

    # Training parameters from config or args
    torch_cfg = getattr(cfg, 'torch', None)
    
    epochs = args.epochs or (torch_cfg.epochs if torch_cfg else None) or getattr(cfg.inference, "vi_iters", 5000)
    lr = args.lr or (torch_cfg.lr if torch_cfg else None) or 0.01
    lr_min = (torch_cfg.lr_min if torch_cfg else None) or 1e-5
    warmup_epochs = (torch_cfg.warmup_epochs if torch_cfg else None) or 100
    patience = args.patience or (torch_cfg.patience if torch_cfg else None) or 500
    min_delta = (torch_cfg.min_delta if torch_cfg else None) or 1.0
    grad_clip = (torch_cfg.grad_clip if torch_cfg else None) or 10.0
    n_samples = (torch_cfg.n_samples if torch_cfg else None) or 1
    log_interval = (torch_cfg.log_interval if torch_cfg else None) or 50
    save_interval = (torch_cfg.save_interval if torch_cfg else None) or 500
    
    use_early_stop = not args.no_early_stop
    
    logger.log(f"\nTraining Configuration:")
    logger.log(f"  Max epochs: {epochs}")
    logger.log(f"  Learning rate: {lr} -> {lr_min} (cosine annealing)")
    logger.log(f"  Warmup epochs: {warmup_epochs}")
    logger.log(f"  Early stopping: {'enabled' if use_early_stop else 'disabled'} (patience={patience})")
    logger.log(f"  Gradient clipping: {grad_clip}")
    logger.log(f"  MC samples: {n_samples}")
    logger.log(f"  Log interval: {log_interval}")

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts
    # T_0: first restart period, T_mult: multiplier for subsequent periods
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=2, eta_min=lr_min)
    
    # Early stopping
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')

    # Training loop
    logger.log(f"\nStarting training...")
    logger.log("=" * 60)
    
    best_elbo = float("-inf")
    start_time = time.time()
    
    # Moving average for smoothed logging
    elbo_ema = None
    ema_alpha = 0.1
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Compute ELBO
        elbo, log_lik, kl = model.elbo(week_obs, season_idx, params, n_samples=n_samples)
        
        # Maximize ELBO (minimize -ELBO)
        loss = -elbo
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        
        # Update EMA
        elbo_val = elbo.item()
        if elbo_ema is None:
            elbo_ema = elbo_val
        else:
            elbo_ema = ema_alpha * elbo_val + (1 - ema_alpha) * elbo_ema
        
        # Record metrics
        logger.log_metrics(epoch, elbo_val, log_lik.item(), kl.item(), current_lr, elapsed,
                          extra={"grad_norm": grad_norm.item(), "elbo_ema": elbo_ema})
        
        # Update best model
        if elbo_val > best_elbo:
            best_elbo = elbo_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'elbo': best_elbo,
            }, out_dir / "best_model.pt")
        
        # Logging
        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            logger.log(
                f"Epoch {epoch:5d}/{epochs} | ELBO={elbo_val:9.2f} (EMA={elbo_ema:9.2f}) | "
                f"LogLik={log_lik.item():9.2f} | KL={kl.item():7.2f} | "
                f"LR={current_lr:.2e} | GradNorm={grad_norm.item():.2f} | Time={elapsed:.0f}s"
            )
        
        # Periodic checkpoint
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'elbo': elbo_val,
            }, out_dir / f"checkpoint_epoch{epoch}.pt")
            logger.log(f"  -> Checkpoint saved at epoch {epoch}")
        
        # Early stopping check (use EMA for stability)
        if use_early_stop and epoch > warmup_epochs:
            if early_stopper(elbo_ema, epoch):
                logger.log(f"\nEarly stopping triggered at epoch {epoch}")
                logger.log(f"Best ELBO was {early_stopper.best_score:.2f} at epoch {early_stopper.best_epoch}")
                break
    
    total_time = time.time() - start_time
    logger.log("=" * 60)
    logger.log(f"Training completed!")
    logger.log(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.log(f"  Final epoch: {epoch}")
    logger.log(f"  Best ELBO: {best_elbo:.2f}")

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'elbo': elbo_val,
    }, out_dir / "final_model.pt")
    
    # Save training history
    logger.save_history(out_dir / "history.npz")
    
    # Load best model for inference
    logger.log("\nLoading best model for inference...")
    checkpoint = torch.load(out_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract posterior means
    logger.log("Extracting posterior means...")
    model.eval()
    with torch.no_grad():
        theta_mean = model.get_posterior_mean_theta().cpu().numpy()
    
    np.save(out_dir / "theta_mean.npy", theta_mean)
    
    # Save posterior parameters
    posterior_params = {
        'beta_mu': model.beta_mu.detach().cpu().numpy().tolist(),
        'beta_std': np.exp(model.beta_logstd.detach().cpu().numpy()).tolist(),
        'gamma_mu': model.gamma_mu.detach().cpu().numpy().tolist(),
        'gamma_std': np.exp(model.gamma_logstd.detach().cpu().numpy()).tolist(),
    }
    with open(out_dir / "posterior_params.json", "w", encoding="utf-8") as f:
        json.dump(posterior_params, f, indent=2)
    
    # Save config
    cfg.save(str(out_dir / "config.resolved.json"))
    
    # Summary
    logger.log(f"\nResults saved to: {out_dir}")
    logger.log("Files:")
    logger.log("  - best_model.pt (best checkpoint)")
    logger.log("  - final_model.pt (final checkpoint)")
    logger.log("  - history.npz (training history)")
    logger.log("  - train_metrics.jsonl (real-time metrics)")
    logger.log("  - theta_mean.npy (posterior mean)")
    logger.log("  - posterior_params.json (key parameters)")
    logger.log("\nDone!")


if __name__ == "__main__":
    main()
