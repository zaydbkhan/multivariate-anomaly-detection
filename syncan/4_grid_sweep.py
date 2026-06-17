"""
Hyperparameter grid sweep for TranAD on SynCAN.

Runs a grid search over parameter combinations, trains a model for each,
evaluates across all 5 attack types with POT, and saves results to a CSV.
After the sweep completes, the winning configuration is retrained end-to-end
and saved to models/syncan/best/ so the initial baseline artifacts are never
touched.

Usage:
    uv run python syncan/4_grid_sweep.py --quick
    uv run python syncan/4_grid_sweep.py
"""

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TranADConfig, TranADNet
from src.scorer import POTParams, calibrate_threshold, evaluate, score_batch
from src.syncan_registry import SynCANRegistry
from src.train import EarlyStopping, train_epoch, validate_epoch, train_full
from src.utils import auto_device, convert_to_windows, subsample_data

SYNCAN_PROCESSED = PROJECT_ROOT / "data" / "syncan" / "processed"
ATTACK_TYPES = ["plateau", "continuous", "playback", "suppress", "flooding"]

QUICK_CONFIGS = [
    {"window_size": 100, "lr": 0.001,  "n_layers": 1, "n_heads": 10, "d_feedforward": 8},
    {"window_size": 100, "lr": 0.001,  "n_layers": 2, "n_heads": 10, "d_feedforward": 8},
    {"window_size": 100, "lr": 0.001,  "n_layers": 3, "n_heads": 10, "d_feedforward": 8},
    {"window_size": 100, "lr": 0.001,  "n_layers": 2, "n_heads": 10, "d_feedforward": 16},
    {"window_size": 100, "lr": 0.001,  "n_layers": 2, "n_heads": 10, "d_feedforward": 32},
    {"window_size": 100, "lr": 0.001,  "n_layers": 2, "n_heads": 5,  "d_feedforward": 8},
    {"window_size": 100, "lr": 0.0001, "n_layers": 2, "n_heads": 10, "d_feedforward": 8},
    {"window_size": 140, "lr": 0.001,  "n_layers": 1, "n_heads": 10, "d_feedforward": 8},
    {"window_size": 140, "lr": 0.001,  "n_layers": 2, "n_heads": 10, "d_feedforward": 8},
    {"window_size": 140, "lr": 0.001,  "n_layers": 3, "n_heads": 10, "d_feedforward": 8},
    {"window_size": 140, "lr": 0.001,  "n_layers": 2, "n_heads": 10, "d_feedforward": 16},
    {"window_size": 140, "lr": 0.001,  "n_layers": 2, "n_heads": 10, "d_feedforward": 32},
    {"window_size": 140, "lr": 0.001,  "n_layers": 2, "n_heads": 5,  "d_feedforward": 8},
    {"window_size": 140, "lr": 0.0001, "n_layers": 2, "n_heads": 10, "d_feedforward": 8},
    {"window_size": 100, "lr": 0.001,  "n_layers": 2, "n_heads": 5,  "d_feedforward": 16},
    {"window_size": 140, "lr": 0.001,  "n_layers": 3, "n_heads": 5,  "d_feedforward": 16},
]

FULL_GRID = {
    "window_size": [100, 140],
    "lr": [0.0001, 0.001],
    "n_layers": [1, 2, 3],
    "n_heads": [5, 10],
    "d_feedforward": [8, 16, 32],
}

CSV_COLUMNS = [
    "trial", "window_size", "lr", "n_layers", "n_heads", "d_feedforward",
    "avg_f1", "avg_precision", "avg_recall",
    "plateau_f1", "continuous_f1", "playback_f1", "suppress_f1", "flooding_f1",
    "epochs_trained", "train_time_s", "final_train_loss", "status",
]


def build_config(params: dict, max_epochs: int, early_stopping_patience: int = 0) -> TranADConfig:
    return TranADConfig(
        n_features=20,
        n_heads=params["n_heads"],
        window_size=params["window_size"],
        d_feedforward=params["d_feedforward"],
        batch_size=256,
        use_layer_norm=False,
        dtype="float32",
        lr=params["lr"],
        loss_weighting="exponential_decay",
        adversarial_loss=False,
        scoring_mode="averaged",
        scheduler_gamma=0.95,
        early_stopping_patience=early_stopping_patience,
        val_split=0.1,
        max_epochs=max_epochs,
        n_encoder_layers=params["n_layers"],
        n_decoder_layers=params["n_layers"],
    )


def evaluate_attacks(
    model: TranADNet,
    config: TranADConfig,
    train_scores: np.ndarray,
    device: torch.device,
    score_batch_size: int = 5000,
) -> dict:
    """Evaluate model on all 5 attack types and return per-attack metrics."""
    attack_results = {}
    for attack in ATTACK_TYPES:
        test_sig = np.load(SYNCAN_PROCESSED / f"test_{attack}_signals.npy")
        test_lbl = np.load(SYNCAN_PROCESSED / f"test_{attack}_labels.npy")

        test_scores = score_batch(
            model, test_sig,
            window_size=config.window_size,
            device=device,
            scoring_mode=config.scoring_mode,
            batch_size=score_batch_size,
        )

        cal = calibrate_threshold(
            train_scores=train_scores,
            test_scores=test_scores,
            labels=test_lbl,
            method="pot",
            pot_params=POTParams(q=1e-3, level=0.99, scale=1.0),
        )

        metrics = evaluate(test_scores, test_lbl, cal["threshold"])
        attack_results[attack] = metrics

    return attack_results


def run_trial(
    config: TranADConfig,
    train_data: np.ndarray,
    device: torch.device,
    score_batch_size: int = 5000,
    subsample_fraction: float = 0.1,
) -> dict:
    """Train a model and evaluate on all 5 attack types.

    Args:
        config: TranAD hyperparameters.
        train_data: numpy array of shape (N, n_features).
        device: torch device.
        score_batch_size: batch size for scoring.
        subsample_fraction: fraction of rows to use for training
            (applied before windowing). 1.0 = full dataset.

    Returns dict with per-attack metrics, avg F1, training stats.
    """
    data_for_training = subsample_data(train_data, subsample_fraction)

    torch_dtype = torch.float64 if config.dtype == "float64" else torch.float32
    train_tensor = torch.from_numpy(data_for_training).to(torch_dtype)
    windows = convert_to_windows(train_tensor, config.window_size)

    use_early_stopping = config.early_stopping_patience > 0
    if use_early_stopping:
        n_total = windows.shape[0]
        n_val = int(n_total * config.val_split)
        n_train = n_total - n_val
        train_windows = windows[:n_train]
        val_windows = windows[n_train:]
        train_loader = DataLoader(TensorDataset(train_windows), batch_size=config.batch_size)
        val_loader = DataLoader(TensorDataset(val_windows), batch_size=config.batch_size)
        stopper = EarlyStopping(patience=config.early_stopping_patience)
        total_epochs = config.max_epochs
    else:
        total_epochs = config.max_epochs
        train_loader = DataLoader(TensorDataset(windows), batch_size=config.batch_size)
        val_loader = None
        stopper = None

    model = TranADNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    loss_fn = nn.MSELoss(reduction="none")

    start_time = time.time()
    final_epoch = 0
    final_loss = 0.0
    for epoch in range(total_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, epoch, config, device)
        scheduler.step()
        final_epoch = epoch + 1
        final_loss = train_loss

        if use_early_stopping and val_loader is not None and stopper is not None:
            val_loss = validate_epoch(model, val_loader, loss_fn, epoch, config, device)
            if stopper.step(val_loss, model):
                stopper.restore_best(model)
                break
    train_time = time.time() - start_time

    train_scores = score_batch(
        model, train_data, config.window_size, device, config.scoring_mode,
        batch_size=score_batch_size,
    )

    attack_results = evaluate_attacks(model, config, train_scores, device, score_batch_size)

    f1s = [m["f1"] for m in attack_results.values()]
    precs = [m["precision"] for m in attack_results.values()]
    recs = [m["recall"] for m in attack_results.values()]

    results = {
        "per_attack": {a: attack_results[a] for a in ATTACK_TYPES},
        "avg_f1": float(np.mean(f1s)),
        "avg_precision": float(np.mean(precs)),
        "avg_recall": float(np.mean(recs)),
        "epochs_trained": final_epoch,
        "train_time_s": round(train_time, 1),
        "final_train_loss": final_loss,
    }
    return results


def retrain_best(
    best_params: dict,
    train_data: np.ndarray,
    device: torch.device,
    output_dir: Path,
    retrain_epochs: int,
    seed: int,
    score_batch_size: int = 5000,
) -> dict:
    """Retrain the winning config end-to-end and save to output_dir."""
    print("\n" + "=" * 70)
    print("RETRAINING BEST CONFIG -> saving to disk")
    print("=" * 70)
    print(f"Config: {', '.join(f'{k}={v}' for k, v in sorted(best_params.items()))}")
    print(f"Destination: {output_dir}")
    print(f"Max epochs: {retrain_epochs}")
    print()

    config = TranADConfig(
        n_features=20,
        n_heads=best_params["n_heads"],
        window_size=best_params["window_size"],
        d_feedforward=best_params["d_feedforward"],
        batch_size=256,
        use_layer_norm=False,
        dtype="float32",
        lr=best_params["lr"],
        loss_weighting="exponential_decay",
        adversarial_loss=False,
        scoring_mode="averaged",
        scheduler_gamma=0.95,
        early_stopping_patience=5,
        val_split=0.1,
        max_epochs=retrain_epochs,
        n_encoder_layers=best_params["n_layers"],
        n_decoder_layers=best_params["n_layers"],
    )

    model, final_epoch, final_loss = train_full(config, train_data, device, seed=seed)

    registry = SynCANRegistry(base_dir=output_dir)
    registry.save_model(model, config, final_loss, epoch=final_epoch - 1)

    train_scores = score_batch(
        model, train_data, config.window_size, device, config.scoring_mode,
        batch_size=score_batch_size,
    )
    attack_results = evaluate_attacks(model, config, train_scores, device, score_batch_size)

    f1s = [m["f1"] for m in attack_results.values()]
    precs = [m["precision"] for m in attack_results.values()]
    recs = [m["recall"] for m in attack_results.values()]

    save_results = {
        "avg_f1": float(np.mean(f1s)),
        "avg_precision": float(np.mean(precs)),
        "avg_recall": float(np.mean(recs)),
        "per_attack": {a: {k: float(v) for k, v in attack_results[a].items()} for a in ATTACK_TYPES},
    }
    registry.save_eval_results(save_results)

    print(f"\nBest-config results:")
    for attack in ATTACK_TYPES:
        m = attack_results[attack]
        print(f"  {attack:<12s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
    print(f"  {'AVERAGE':<12s} F1={save_results['avg_f1']:.4f}  "
          f"P={save_results['avg_precision']:.4f}  R={save_results['avg_recall']:.4f}")
    print("=" * 70)

    return save_results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter grid sweep for SynCAN TranAD")
    parser.add_argument("--data-dir", type=str, default="data/syncan/processed")
    parser.add_argument("--results-dir", type=str, default="results/syncan")
    parser.add_argument("--output-dir", type=str, default="models/syncan/best",
                        help="Where to save the retrained best model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="Use 100%% of training data for sweep trials (default: 10%% subsample)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all trial configs and exit without training")
    parser.add_argument("--max-sweep-epochs", type=int, default=5,
                        help="Epochs per sweep trial (no early stopping)")
    parser.add_argument("--retrain-epochs", type=int, default=30,
                        help="Max epochs for the final retrain (uses early stopping)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-batch-size", type=int, default=5000,
                        help="Batch size for scoring (0 = process all at once)")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    results_dir = PROJECT_ROOT / args.results_dir
    output_dir = PROJECT_ROOT / args.output_dir
    device = auto_device(args.device)
    print(f"Device: {device}")

    train_signals = np.load(data_dir / "train_signals.npy")
    print(f"Training data: {train_signals.shape}")

    if args.quick:
        combos = QUICK_CONFIGS
        grid_type = "quick"
    else:
        grid = FULL_GRID
        param_names = sorted(grid.keys())
        combos = [
            dict(zip(param_names, values))
            for values in itertools.product(*(grid[k] for k in param_names))
        ]
        grid_type = "full"
    print(f"Grid: {grid_type}, {len(combos)} configurations")

    subsample_frac = 1.0 if args.full else 0.1
    print(f"Sweep trials will use {'full dataset' if args.full else '10% subsample'}")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN — no training will be performed")
        print("=" * 70)
        for i, params in enumerate(combos):
            trial_num = i + 1
            config = build_config(params, args.max_sweep_epochs, early_stopping_patience=0)
            print(f"\n[{trial_num}/{len(combos)}]")
            for k, v in sorted(params.items()):
                print(f"  {k}: {v}")
            print(f"  (epochs={args.max_sweep_epochs}, early_stopping_patience=0, "
                  f"subsample={subsample_frac})")
        print("\n" + "=" * 70)
        print("Dry run complete — no models were trained")
        print("=" * 70)
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"sweep_syncan_{grid_type}.csv"

    completed_trials: set[int] = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok":
                    completed_trials.add(int(row["trial"]))
        print(f"Resuming: {len(completed_trials)} trials already completed")

    write_header = not results_path.exists() or not args.resume
    if write_header:
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)

    best_avg_f1 = -1.0
    best_params: dict = {}

    for i, params in enumerate(combos):
        trial_num = i + 1

        if trial_num in completed_trials:
            continue

        trial_device = device

        params_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        print(f"\n[{trial_num}/{len(combos)}] {params_str}")

        try:
            config = build_config(params, args.max_sweep_epochs, early_stopping_patience=0)
            results = run_trial(
                config, train_signals, trial_device,
                args.score_batch_size, subsample_fraction=subsample_frac,
            )

            avg_f1 = results["avg_f1"]
            print(f"  Avg F1={avg_f1:.4f}  P={results['avg_precision']:.4f}  "
                  f"R={results['avg_recall']:.4f}  epochs={results['epochs_trained']}  "
                  f"time={results['train_time_s']}s")
            for attack in ATTACK_TYPES:
                m = results["per_attack"][attack]
                print(f"    {attack:<12s} F1={m['f1']:.4f}")

            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_params = params.copy()
                print("  *** New best avg F1! ***")

            row = [
                trial_num, params["window_size"], params["lr"],
                params["n_layers"], params["n_heads"], params["d_feedforward"],
                f"{results['avg_f1']:.6f}", f"{results['avg_precision']:.6f}",
                f"{results['avg_recall']:.6f}",
                f"{results['per_attack']['plateau']['f1']:.6f}",
                f"{results['per_attack']['continuous']['f1']:.6f}",
                f"{results['per_attack']['playback']['f1']:.6f}",
                f"{results['per_attack']['suppress']['f1']:.6f}",
                f"{results['per_attack']['flooding']['f1']:.6f}",
                results["epochs_trained"], results["train_time_s"],
                f"{results['final_train_loss']:.6f}", "ok",
            ]
        except Exception as e:
            print(f"  FAILED: {e}")
            param_names = sorted(params.keys())
            row = [trial_num] + [params.get(k, "") for k in param_names]
            row += [""] * (len(CSV_COLUMNS) - len(param_names) - 1)
            row[-1] = f"error: {e}"

        with open(results_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    print("\n" + "=" * 70)
    print(f"SWEEP COMPLETE: {len(combos)} trials")
    print(f"Results saved to: {results_path}")
    print(f"\nBest avg F1: {best_avg_f1:.4f}")
    if best_params:
        print("Best params:")
        for k, v in sorted(best_params.items()):
            print(f"  {k}: {v}")
    print("=" * 70)

    if best_params:
        retrain_best(
            best_params=best_params,
            train_data=train_signals,
            device=device,
            output_dir=output_dir,
            retrain_epochs=args.retrain_epochs,
            seed=args.seed,
            score_batch_size=args.score_batch_size,
        )


if __name__ == "__main__":
    main()
