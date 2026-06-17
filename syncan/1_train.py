"""
Train TranAD on preprocessed SynCAN data.

Trains a baseline model with deliberately conservative defaults so you can
see the pipeline working before optimizing. The grid sweep
(syncan/4_grid_sweep.py) finds the best configuration and retrains.

Usage:
    uv run python syncan/1_train.py
    uv run python syncan/1_train.py --window-size 200 --epochs 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TranADConfig
from src.train import train_full
from src.utils import auto_device

BASELINE_DEFAULTS = {
    "window_size": 100,
    "epochs": 5,
    "batch_size": 384,
    "lr": 0.001,
    "d_feedforward": 8,
    "n_layers": 1,
    "scheduler_gamma": 0.95,
    "loss_weighting": "exponential_decay",
    "scoring_mode": "averaged",
    "early_stopping_patience": 0,
    "val_split": 0.1,
    "max_epochs": 5,
}


def main():
    parser = argparse.ArgumentParser(description="Train TranAD baseline on SynCAN")
    parser.add_argument("--data-dir", type=str, default="data/syncan/processed")
    parser.add_argument("--output-dir", type=str, default="models/syncan/initial")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-feedforward", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--full", action="store_true",
                        help="Use 100%% of training data (default: 10%% subsample)")

    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir

    n_layers = args.n_layers or BASELINE_DEFAULTS["n_layers"]
    config = TranADConfig(
        n_features=20,
        n_heads=10,
        window_size=args.window_size or BASELINE_DEFAULTS["window_size"],
        epochs=args.epochs or BASELINE_DEFAULTS["epochs"],
        batch_size=args.batch_size or BASELINE_DEFAULTS["batch_size"],
        lr=args.lr or BASELINE_DEFAULTS["lr"],
        d_feedforward=args.d_feedforward or BASELINE_DEFAULTS["d_feedforward"],
        loss_weighting=BASELINE_DEFAULTS["loss_weighting"],
        scoring_mode=BASELINE_DEFAULTS["scoring_mode"],
        scheduler_gamma=BASELINE_DEFAULTS["scheduler_gamma"],
        early_stopping_patience=(
            args.early_stopping_patience or BASELINE_DEFAULTS["early_stopping_patience"]
        ),
        val_split=args.val_split or BASELINE_DEFAULTS["val_split"],
        max_epochs=args.max_epochs or BASELINE_DEFAULTS["max_epochs"],
        n_encoder_layers=n_layers,
        n_decoder_layers=n_layers,
    )

    subsample_frac = 1.0 if args.full else 0.1

    device = auto_device(args.device)
    if config.dtype == "float64" and device.type == "mps":
        print("Warning: float64 may not be fully supported on MPS, falling back to CPU")
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading training data from {data_dir}...")
    train_path = data_dir / "train_signals.npy"
    if not train_path.exists():
        print(f"Error: {train_path} not found. Run syncan/0_verify_setup.py first.")
        sys.exit(1)

    train_data = np.load(train_path)
    print(f"Training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    print(f"Config: window_size={config.window_size}, epochs={config.epochs}, "
          f"batch_size={config.batch_size}, lr={config.lr}")
    print(f"  n_layers={n_layers}, loss_weighting={config.loss_weighting}, "
          f"scoring_mode={config.scoring_mode}")
    print(f"  early_stopping_patience={config.early_stopping_patience}")
    print(f"  subsample_fraction={subsample_frac} "
          f"({'full dataset' if args.full else '10% subsample'})")

    model, final_epoch, epoch_loss = train_full(
        config, train_data, device, seed=args.seed,
        subsample_fraction=subsample_frac,
    )

    ckpt_dir = output_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.ckpt"
    torch.save(
        {
            "epoch": final_epoch - 1,
            "model_state_dict": model.state_dict(),
            "config": config,
            "final_loss": epoch_loss,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")
    print(f"Final loss: {epoch_loss:.6f}")


if __name__ == "__main__":
    main()
