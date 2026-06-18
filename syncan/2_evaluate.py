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
from src.eval_syncan import (
    compute_recall_progression,
    compute_tnr,
    evaluate_intervals,
    extract_attack_intervals,
    sample_normal_intervals,
)
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

    pot_params = POTParams(q=1e-3, level=0.99, scale=1.0)
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

    if test_lbl.ndim == 2:
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

    # ---- score normal test data for TNR ----
    print("Scoring normal test data for TNR estimation...")
    normal_sig = np.load(DATA_DIR / "test_normal_signals.npy")
    normal_scores = score_batch(
        model, normal_sig,
        window_size=config.window_size,
        device=device,
        scoring_mode=config.scoring_mode,
        batch_size=score_batch_size,
    )
    normal_score_1d = np.mean(normal_scores, axis=1)
    print(f"  normal_score_1d shape={normal_score_1d.shape}")

    # Sample normal intervals matching attack distribution
    all_attack_lengths = []
    for attack in ATTACK_TYPES:
        lbl = test_data[attack][1]
        intervals = extract_attack_intervals(lbl)
        all_attack_lengths.extend([e - s for s, e in intervals])
    median_duration = int(np.median(all_attack_lengths)) if all_attack_lengths else 200
    n_normal = min(len(all_attack_lengths), 50) if all_attack_lengths else 40
    normal_intervals = sample_normal_intervals(
        len(normal_sig), n_normal, median_duration, seed=42
    )
    print(f"  Sampled {len(normal_intervals)} normal intervals of {median_duration} steps")

    results = {}
    all_test_scores: list[np.ndarray] = []
    all_test_labels: list[np.ndarray] = []
    interval_metrics: dict[str, dict] = {}
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

        # ---- interval detection (>= Q% of interval flagged) ----
        score_1d = np.mean(test_scores, axis=1)
        threshold = float(m["threshold"])
        attack_intervals = extract_attack_intervals(lbl)
        if attack_intervals:
            iv = evaluate_intervals(score_1d, threshold, attack_intervals)
            tnr_result = compute_tnr(normal_score_1d, threshold, normal_intervals)
            interval_metrics[attack] = {"attack": iv, "tnr": tnr_result}
            r25 = iv["interval_recall"].get("0.25", 0.0)
            t25 = tnr_result["tnr"].get("0.25", 1.0)
            print(f"  Interval: R@25%={r25:.4f}  TNR@25%={t25:.4f}  "
                  f"({iv['total_intervals']} attack intervals)")
            # Recall progression within interval (fraction-of-duration)
            rp = compute_recall_progression(score_1d, lbl, threshold, attack_intervals)
            interval_metrics[attack]["recall_progression"] = rp
            print(f"  Recall progression: 25%={rp.get('0.25', 0):.4f} "
                  f"50%={rp.get('0.50', 0):.4f} 75%={rp.get('0.75', 0):.4f} "
                  f"100%={rp.get('1.00', 0):.4f}")
        else:
            interval_metrics[attack] = {"attack": None, "tnr": None}
            print(f"  Interval: no attack intervals found")

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

    # ---- interval detection summary ----
    q_labels = ["0.01", "0.02", "0.05", "0.10", "0.25", "0.50", "0.90"]
    has_interval = any(
        interval_metrics.get(a, {}).get("attack") is not None
        for a in ATTACK_TYPES
    )
    if has_interval:
        print()
        print("=" * 82)
        print("Interval Detection (attack detected if >= Q% of interval flagged)")
        print("=" * 82)
        header = f"{'Attack':<12s}"
        header += "".join(f"{f'R@{q}':>8s}" for q in q_labels)
        header += f" {'Flagged%':>9s} {'Ints':>5s}"
        print(header)
        print("-" * 82)

        tnr_vals = {q: [] for q in q_labels}
        for attack in ATTACK_TYPES:
            entry = interval_metrics.get(attack, {})
            iv = entry.get("attack")
            if iv is None:
                continue
            vals = "".join(f"{iv['interval_recall'].get(q, 0):>8.4f}" for q in q_labels)
            print(f"{attack:<12s}{vals}{iv['avg_flagged_fraction']:>9.4f}{iv['total_intervals']:>5d}")
            tnr_entry = entry.get("tnr")
            if tnr_entry:
                for q in q_labels:
                    tnr_vals[q].append(tnr_entry["tnr"].get(q, 1.0))

        if any(tnr_vals[q] for q in q_labels):
            print("-" * 82)
            avg_tnr = "".join(
                f"{np.mean(tnr_vals[q]):>8.4f}" if tnr_vals[q] else f"{'':>8s}"
                for q in q_labels
            )
            print(f"{'TNR (avg)':<12s}{avg_tnr}{'':>9s}{'':>5s}")
            print(f"  (TNR from {len(normal_intervals)} normal intervals of {median_duration} steps)")
        print("=" * 82)

        # ---- recall progression summary ----
        print()
        print("=" * 55)
        print("Recall Progression (cumulative fraction of interval)")
        print("=" * 55)
        rp_header = f"{'Attack':<12s} {'25%':>8s} {'50%':>8s} {'75%':>8s} {'100%':>8s}"
        print(rp_header)
        print("-" * 55)
        for attack in ATTACK_TYPES:
            entry = interval_metrics.get(attack, {})
            iv = entry.get("attack")
            rp = entry.get("recall_progression", {})
            if iv is None:
                continue
            print(
                f"{attack:<12s} "
                f"{rp.get('0.25', 0):>8.4f} {rp.get('0.50', 0):>8.4f} "
                f"{rp.get('0.75', 0):>8.4f} {rp.get('1.00', 0):>8.4f}"
            )
        print("=" * 55)

    # Feature attribution — per-attack to avoid cross-boundary point-adjustment
    print("\n" + "=" * 65)
    print("ROOT CAUSE ATTRIBUTION")
    print("=" * 65)
    mean_threshold = float(np.mean([results[a]["threshold"] for a in ATTACK_TYPES]))

    attack_attribution: dict[str, dict] = {}
    all_summary_segments: list[dict] = []
    for attack in ATTACK_TYPES:
        test_scores = all_test_scores[ATTACK_TYPES.index(attack)]
        test_labels = all_test_labels[ATTACK_TYPES.index(attack)]

        score_1d = np.mean(test_scores, axis=1)
        labels_1d = test_labels.astype(float)
        threshold = results[attack]["threshold"]

        raw_preds = (score_1d > threshold).astype(float)
        adj_preds = adjust_predicts(score_1d, labels_1d, threshold)

        segs = build_segment_summaries(
            test_scores, adj_preds, baselines,
            feature_labels=signal_labels,
        )

        top_channels: set[str] = set()
        for seg in segs[:3]:
            for d in seg.get("attributed_dimensions", [])[:3]:
                top_channels.add(d["label"])

        attack_attribution[attack] = {
            "n_segments": len(segs),
            "top_signals": sorted(top_channels)[:5],
        }
        all_summary_segments.extend(segs)

        if segs:
            print(f"\n  {attack} ({len(segs)} segments):")
            top_list = sorted(top_channels)[:5]
            print(f"    Top contributed signals: {', '.join(top_list) if top_list else '(none)'}")
            for seg in segs[:3]:
                top_dims = [d["label"] for d in seg.get("attributed_dimensions", [])[:3]]
                print(f"    Segment [{seg['segment_start']}-{seg['segment_end']}]: "
                      f"peak={seg['peak_score']:.4f}, top={top_dims}")

    print(f"\n  Total segments across all attacks: {len(all_summary_segments)}")

    save_data = {
        "method": method,
        "per_attack": {
            a: {
                **{k: float(v) if hasattr(v, "item") else v for k, v in results[a].items()},
                "interval_metrics": interval_metrics.get(a, {}),
            }
            for a in ATTACK_TYPES
        },
        "avg_f1": float(np.mean(f1s)),
        "avg_precision": float(np.mean(precs)),
        "avg_recall": float(np.mean(recs)),
        "attribution": attack_attribution,
    }
    if has_interval:
        save_data["normal_interval_duration"] = median_duration
        save_data["n_normal_intervals"] = len(normal_intervals)

    registry.save_eval_results(save_data)
    print(f"\nResults saved to {registry.eval_path}")

    attribution_path = model_dir / "attribution_results.json"
    with open(attribution_path, "w") as f:
        json.dump(all_summary_segments, f, indent=2, default=_json_safe)
    print(f"Attribution saved to {attribution_path}")

    registry.save_scorer_state({
        "threshold": mean_threshold,
        "feature_baselines": baselines,
        "method": method,
        "details": {"q": 1e-3, "level": 0.99, "scale": 1.0},
    })
    print(f"Scorer state saved to {registry.scorer_path}")

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

    # ---- interval detection ----
    q_labels = ["0.01", "0.02", "0.05", "0.10", "0.25", "0.50", "0.90"]
    has_interval = any(
        per_attack.get(a, {}).get("interval_metrics", {}).get("attack")
        for a in ATTACK_TYPES
    )
    if has_interval:
        print()
        print("=" * 82)
        print("Interval Detection (attack detected if >= Q% of interval flagged)")
        print("=" * 82)
        header = f"{'Attack':<12s}"
        header += "".join(f"{f'R@{q}':>8s}" for q in q_labels)
        header += f" {'Flagged%':>9s} {'Ints':>5s}"
        print(header)
        print("-" * 82)
        for attack in ATTACK_TYPES:
            entry = per_attack.get(attack, {}).get("interval_metrics", {}).get("attack")
            if entry is None:
                continue
            vals = "".join(f"{entry['interval_recall'].get(q, 0):>8.4f}" for q in q_labels)
            print(f"{attack:<12s}{vals}{entry['avg_flagged_fraction']:>9.4f}{entry['total_intervals']:>5d}")
        print("=" * 82)

        # ---- recall progression ----
        print()
        print("=" * 55)
        print("Recall Progression (cumulative fraction of interval)")
        print("=" * 55)
        rp_header = f"{'Attack':<12s} {'25%':>8s} {'50%':>8s} {'75%':>8s} {'100%':>8s}"
        print(rp_header)
        print("-" * 55)
        for attack in ATTACK_TYPES:
            entry = per_attack.get(attack, {}).get("interval_metrics", {}).get("recall_progression", {})
            if not entry:
                continue
            print(
                f"{attack:<12s} "
                f"{entry.get('0.25', 0):>8.4f} {entry.get('0.50', 0):>8.4f} "
                f"{entry.get('0.75', 0):>8.4f} {entry.get('1.00', 0):>8.4f}"
            )
        print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TranAD on SynCAN")
    parser.add_argument("--model-dir", type=str, default="models/syncan/best")
    parser.add_argument("--method", type=str, default="pot",
                        choices=["pot", "percentile", "f1_max"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--from-saved", action="store_true",
                        help="Show saved results without re-scoring")
    parser.add_argument("--score-batch-size", type=int, default=2000)
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
