"""
Evaluate TranAD on all SynCAN attack types.

Loads a trained checkpoint, runs batch inference on all 5 attack test sets,
calibrates anomaly thresholds (POT + percentile), and prints a per-attack
metrics table.

Usage:
    uv run python syncan/2_evaluate.py
    uv run python syncan/2_evaluate.py --model-dir models/syncan/initial
    uv run python syncan/2_evaluate.py --method percentile
    uv run python syncan/2_evaluate.py --from-saved
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.syncan_registry import SynCANRegistry
from src.scorer import POTParams, calibrate_threshold, evaluate, score_batch
from src.utils import auto_device

ATTACK_TYPES = ["plateau", "continuous", "playback", "suppress", "flooding"]
DATA_DIR = PROJECT_ROOT / "data" / "syncan" / "processed"


def load_data() -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load training signals and all attack test data."""
    train_sig = np.load(DATA_DIR / "train_signals.npy")
    test_data = {}
    for attack in ATTACK_TYPES:
        sig = np.load(DATA_DIR / f"test_{attack}_signals.npy")
        lbl = np.load(DATA_DIR / f"test_{attack}_labels.npy")
        test_data[attack] = (sig, lbl)
    return train_sig, test_data


def evaluate_attack(
    model,
    config,
    train_scores: np.ndarray,
    test_sig: np.ndarray,
    test_lbl: np.ndarray,
    device,
    method: str,
    score_batch_size: int,
) -> dict:
    """Score a single attack and return metrics."""
    test_scores = score_batch(
        model, test_sig,
        window_size=config.window_size,
        device=device,
        scoring_mode=config.scoring_mode,
        batch_size=score_batch_size,
    )

    pot_params = POTParams(q=1e-5, level=0.999, scale=1.0)
    cal = calibrate_threshold(
        train_scores=train_scores,
        test_scores=test_scores,
        labels=test_lbl,
        method=method,
        pot_params=pot_params,
        percentile=99.0,
    )

    metrics = evaluate(test_scores, test_lbl, cal["threshold"])
    metrics["threshold"] = float(cal["threshold"])
    return metrics


def evaluate_model(
    model_dir: Path,
    device,
    method: str,
    score_batch_size: int,
) -> dict | None:
    """Load model, evaluate all attacks, print and save results."""
    registry = SynCANRegistry(base_dir=model_dir)
    model, config = registry.get_model(device=device)
    print(f"Loaded model: {config.n_features} features, window_size={config.window_size}, "
          f"scoring_mode={config.scoring_mode}")

    train_sig, test_data = load_data()
    print(f"Training data: {train_sig.shape}")

    train_scores = score_batch(
        model, train_sig,
        window_size=config.window_size,
        device=device,
        scoring_mode=config.scoring_mode,
        batch_size=score_batch_size,
    )
    print(f"Train scores: shape={train_scores.shape}, "
          f"mean={train_scores.mean():.6f}, max={train_scores.max():.6f}")

    results = {}
    for attack in ATTACK_TYPES:
        sig, lbl = test_data[attack]
        m = evaluate_attack(model, config, train_scores, sig, lbl, device, method, score_batch_size)
        results[attack] = m
        n_anom = int(lbl.sum())
        print(f"  {attack:<12s} F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  AUC={m['roc_auc']:.4f}  "
              f"thresh={m['threshold']:.6f}  anomalies={n_anom}")

    f1s = [results[a]["f1"] for a in ATTACK_TYPES]
    precs = [results[a]["precision"] for a in ATTACK_TYPES]
    recs = [results[a]["recall"] for a in ATTACK_TYPES]

    print()
    print("=" * 65)
    print(f"Summary (method={method})")
    print("=" * 65)
    print(f"{'Attack':<12s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'AUC':>8s} {'Thresh':>10s}")
    print("-" * 65)
    for attack in ATTACK_TYPES:
        m = results[attack]
        print(f"{attack:<12s} {m['f1']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['roc_auc']:>8.4f} {m['threshold']:>10.6f}")
    print("-" * 65)
    print(f"{'AVERAGE':<12s} {np.mean(f1s):>8.4f} {np.mean(precs):>8.4f} "
          f"{np.mean(recs):>8.4f}")
    print("=" * 65)

    save_data = {
        "method": method,
        "per_attack": {
            a: {k: float(v) if hasattr(v, "item") else v for k, v in results[a].items()}
            for a in ATTACK_TYPES
        },
        "avg_f1": float(np.mean(f1s)),
        "avg_precision": float(np.mean(precs)),
        "avg_recall": float(np.mean(recs)),
    }

    registry.save_eval_results(save_data)
    print(f"\nResults saved to {registry.eval_path}")

    return save_data


def show_from_saved(model_dir: Path) -> None:
    """Read saved eval_results.json and print summary table."""
    registry = SynCANRegistry(base_dir=model_dir)
    saved = registry.get_scorer_state()
    eval_path = registry.eval_path

    if not eval_path.exists():
        print(f"No saved results found at {eval_path}")
        return

    with open(eval_path) as f:
        results = json.load(f)

    method = results.get("method", "?")
    per_attack = results.get("per_attack", {})

    print()
    print("=" * 65)
    print(f"SAVED RESULTS (method={method})")
    print("=" * 65)
    print(f"{'Attack':<12s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s}")
    print("-" * 65)

    for attack in ATTACK_TYPES:
        if attack not in per_attack:
            continue
        m = per_attack[attack]
        print(f"{attack:<12s} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}")
    print("-" * 65)
    print(f"{'AVERAGE':<12s} {results.get('avg_f1', 0):>8.4f} "
          f"{results.get('avg_precision', 0):>8.4f} "
          f"{results.get('avg_recall', 0):>8.4f}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TranAD on SynCAN")
    parser.add_argument("--model-dir", type=str, default="models/syncan/best")
    parser.add_argument("--method", type=str, default="pot",
                        choices=["pot", "percentile"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--from-saved", action="store_true",
                        help="Show saved results without re-scoring")
    parser.add_argument("--score-batch-size", type=int, default=5000)
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir
    device = auto_device(args.device)
    print(f"Device: {device}")
    print(f"Model dir: {model_dir}")

    if args.from_saved:
        show_from_saved(model_dir)
    else:
        evaluate_model(model_dir, device, args.method, args.score_batch_size)


if __name__ == "__main__":
    main()
