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
from src.scorer import (
    POTParams,
    adjust_predicts,
    build_segment_summaries,
    calibrate_threshold,
    compute_feature_baselines,
    diagnose,
    diagnose_with_elevation,
    evaluate,
    score_batch,
)
from src.utils import auto_device

ATTACK_TYPES = ["plateau", "continuous", "playback", "suppress", "flooding"]
DATA_DIR = PROJECT_ROOT / "data" / "syncan" / "processed"


def _json_safe(o):
    """Convert numpy types to Python scalars for JSON serialization."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


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
    baselines: np.ndarray,
    test_sig: np.ndarray,
    test_lbl: np.ndarray,
    device,
    method: str,
    score_batch_size: int,
) -> dict:
    """Score a single attack and return metrics with attribution."""
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

    diag = diagnose(test_scores, test_lbl)
    metrics.update(diag)

    diag_elev = diagnose_with_elevation(test_scores, test_lbl, baselines)
    metrics.update(diag_elev)

    return metrics, test_scores


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

    signal_labels = np.load(DATA_DIR / "signal_columns.npy", allow_pickle=True).tolist()

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

    baselines = compute_feature_baselines(train_scores)
    print(f"Feature baselines: shape={baselines.shape}")

    results = {}
    all_test_scores: list[np.ndarray] = []
    all_test_labels: list[np.ndarray] = []
    for attack in ATTACK_TYPES:
        sig, lbl = test_data[attack]
        m, test_scores = evaluate_attack(
            model, config, train_scores, baselines,
            sig, lbl, device, method, score_batch_size,
        )
        results[attack] = m
        all_test_scores.append(test_scores)
        all_test_labels.append(lbl)
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

    # Feature attribution across all attack segments
    print("\n" + "=" * 65)
    print("ROOT CAUSE ATTRIBUTION")
    print("=" * 65)
    concat_scores = np.concatenate(all_test_scores, axis=0)
    concat_labels = np.concatenate(all_test_labels, axis=0)
    score_1d = np.mean(concat_scores, axis=1)
    labels_1d = concat_labels.astype(float)
    threshold = np.mean([results[a]["threshold"] for a in ATTACK_TYPES])

    raw_predictions = (score_1d > threshold).astype(float)
    adjusted_predictions = adjust_predicts(score_1d, labels_1d, threshold)

    summary_segments = build_segment_summaries(
        concat_scores, adjusted_predictions, baselines,
        feature_labels=signal_labels,
    )
    print(f"  Detected {len(summary_segments)} anomalous segments across all attacks")

    attack_sizes = {a: len(all_test_scores[i]) for i, a in enumerate(ATTACK_TYPES)}
    attack_offsets: dict[str, int] = {}
    cum = 0
    for a in ATTACK_TYPES:
        attack_offsets[a] = cum
        cum += attack_sizes[a]

    attack_attribution: dict[str, dict] = {}
    for attack in ATTACK_TYPES:
        offset = attack_offsets[attack]
        end = offset + attack_sizes[attack]
        attack_segments = [
            s for s in summary_segments
            if offset <= s["segment_start"] < end
        ]
        top_channels: set[str] = set()
        for seg in attack_segments[:3]:
            for d in seg.get("attributed_dimensions", [])[:3]:
                top_channels.add(d["label"])
        attack_attribution[attack] = {
            "n_segments": len(attack_segments),
            "top_signals": sorted(top_channels)[:5],
        }
        if attack_segments:
            print(f"\n  {attack}:")
            top_list = sorted(top_channels)[:5]
            print(f"    Top contributed signals: {', '.join(top_list) if top_list else '(none)'}")
            for seg in attack_segments[:3]:
                top_dims = [d["label"] for d in seg.get("attributed_dimensions", [])[:3]]
                print(f"    Segment [{seg['segment_start']}-{seg['segment_end']}]: "
                      f"peak={seg['peak_score']:.4f}, top={top_dims}")

    save_data = {
        "method": method,
        "per_attack": {
            a: {k: float(v) if hasattr(v, "item") else v for k, v in results[a].items()}
            for a in ATTACK_TYPES
        },
        "avg_f1": float(np.mean(f1s)),
        "avg_precision": float(np.mean(precs)),
        "avg_recall": float(np.mean(recs)),
        "attribution": attack_attribution,
    }

    registry.save_eval_results(save_data)
    print(f"\nResults saved to {registry.eval_path}")

    attribution_path = model_dir / "attribution_results.json"
    with open(attribution_path, "w") as f:
        json.dump(summary_segments, f, indent=2, default=_json_safe)
    print(f"Attribution saved to {attribution_path}")

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
