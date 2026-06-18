"""
Synthetic coordinated multi-target attack evaluation for TranAD on SynCAN.

Generates 3 coordinated attack scenarios using correlated signal groups,
evaluates with the best available model, and reports results.

Attack philosophy: subtle, realistic coordinated attacks. Signals are frozen
or drifted to individually-plausible values — the anomaly is in the broken
correlation between signals, not in any single signal being extreme.

Usage:
    uv run python syncan/5_coordinated_attack.py
    uv run python syncan/5_coordinated_attack.py --model-dir models/syncan/initial
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_syncan import (
    compute_tnr,
    evaluate_intervals,
    sample_normal_intervals,
)
from src.scorer import (
    POTParams,
    adjust_predicts,
    build_segment_summaries,
    calibrate_threshold,
    compute_feature_baselines,
    evaluate,
    score_batch,
)
from src.syncan_registry import SynCANRegistry
from src.utils import auto_device

PROCESSED = PROJECT_ROOT / "data" / "syncan" / "processed"


def parse_device_id(signal_name: str) -> str:
    return signal_name.split("_")[0]


def compute_cross_id_pairs(
    corr_matrix: np.ndarray,
    signal_labels: list[str],
) -> list[tuple[int, int, float]]:
    n = len(signal_labels)
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if parse_device_id(signal_labels[i]) != parse_device_id(signal_labels[j]):
                pairs.append((i, j, abs(corr_matrix[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def build_plateau_groups(
    pairs: list[tuple[int, int, float]],
    signal_labels: list[str],
    num_groups: int = 5,
) -> list[list[int]]:
    used: set[int] = set()
    groups: list[list[int]] = []
    for i, j, _ in pairs:
        if len(groups) >= num_groups:
            break
        if i in used or j in used:
            continue
        group = [i, j]
        used.add(i)
        used.add(j)
        for base in [i, j]:
            if len(group) >= 4:
                break
            existing_ids = {parse_device_id(signal_labels[d]) for d in group}
            for p_i, p_j, _ in pairs:
                candidate = -1
                if p_i == base and p_j not in used:
                    if parse_device_id(signal_labels[p_j]) not in existing_ids:
                        candidate = p_j
                elif p_j == base and p_i not in used:
                    if parse_device_id(signal_labels[p_i]) not in existing_ids:
                        candidate = p_i
                if candidate >= 0:
                    group.append(candidate)
                    used.add(candidate)
                    break
        groups.append(group[:4])
    return groups


def generate_attack_positions(
    total_length: int,
    num_attacks: int,
    duration: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    positions: list[tuple[int, int]] = []
    current = int(rng.integers(2000, 5000))
    for _ in range(num_attacks):
        end = current + duration
        if end > total_length - 2000:
            break
        positions.append((current, end))
        gap = int(rng.integers(500, 2001))
        current = end + gap
    return positions


def build_labels(
    total_length: int,
    positions: list[tuple[int, int]],
) -> np.ndarray:
    labels = np.zeros(total_length, dtype=np.float64)
    for start, end in positions:
        labels[start:end] = 1.0
    return labels


def generate_coordinated_plateau(
    normal_signals: np.ndarray,
    positions: list[tuple[int, int]],
    groups: list[list[int]],
    scenario_seed: int,
) -> np.ndarray:
    sig = normal_signals.copy()
    for atk_idx, (start, end) in enumerate(positions):
        group = groups[atk_idx % len(groups)]
        rng = np.random.default_rng(scenario_seed + atk_idx * 10)
        for dim in group:
            freeze_val = rng.uniform(-0.5, 0.5)
            sig[start:end, dim] = freeze_val
    return sig


def generate_coordinated_mixed(
    normal_signals: np.ndarray,
    positions: list[tuple[int, int]],
    pairs: list[tuple[int, int]],
    scenario_seed: int,
) -> np.ndarray:
    sig = normal_signals.copy()
    for atk_idx, (start, end) in enumerate(positions):
        i, j = pairs[atk_idx % len(pairs)]
        dur = end - start
        rng = np.random.default_rng(scenario_seed + atk_idx * 10)
        sig[start:end, i] = rng.uniform(-0.5, 0.5)
        ramp = np.linspace(0, 2.0, dur)
        sig[start:end, j] += ramp
    return sig


def generate_coordinated_suppress_plateau(
    normal_signals: np.ndarray,
    positions: list[tuple[int, int]],
    pairs: list[tuple[int, int]],
    scenario_seed: int,
) -> np.ndarray:
    sig = normal_signals.copy()
    for atk_idx, (start, end) in enumerate(positions):
        i, j = pairs[atk_idx % len(pairs)]
        rng = np.random.default_rng(scenario_seed + atk_idx * 10)
        sig[start:end, i] = 0.0
        sig[start:end, j] = rng.uniform(-0.5, 0.5)
    return sig


def compute_recall_progression(
    score_1d: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    positions: list[tuple[int, int]],
    checkpoints: list[int],
) -> dict:
    cp_recalls: dict[int, list[float]] = {cp: [] for cp in checkpoints}
    for start, end in positions:
        dur = end - start
        for cp in checkpoints:
            if cp > dur:
                continue
            window_end = start + cp
            slice_preds = (score_1d[start:window_end] > threshold).astype(float)
            slice_labels = labels[start:window_end]
            tp = ((slice_preds == 1) & (slice_labels == 1)).sum()
            fn = ((slice_preds == 0) & (slice_labels == 1)).sum()
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            cp_recalls[cp].append(recall)
    return {
        str(cp): float(np.mean(cp_recalls[cp])) if cp_recalls[cp] else 0.0
        for cp in checkpoints if cp_recalls[cp]
    }


SCENARIOS = [
    ("coordinated_plateau", generate_coordinated_plateau),
    ("coordinated_mixed", generate_coordinated_mixed),
    ("coordinated_suppress_plateau", generate_coordinated_suppress_plateau),
]

SHORT_NAMES = {
    "coordinated_plateau": "cplateau",
    "coordinated_mixed": "cmixed",
    "coordinated_suppress_plateau": "csuppress_plateau",
}


def main():
    parser = argparse.ArgumentParser(
        description="Coordinated multi-target attack evaluation for TranAD on SynCAN"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model directory (default: auto-pick best/ then initial/)"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--score-batch-size", type=int, default=2000)
    parser.add_argument("--num-attacks", type=int, default=20)
    parser.add_argument("--attack-duration", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--two-tailed", action="store_true",
                        help="Also evaluate with two-tailed z-score attribution")
    parser.add_argument("--method", type=str, default="pot",
                        choices=["pot", "percentile", "f1_max"],
                        help="Threshold calibration method")
    args = parser.parse_args()

    device = auto_device(args.device)
    print(f"Device: {device}")

    # ---- model selection ----
    if args.model_dir is not None:
        model_dir = Path(args.model_dir)
    else:
        best = PROJECT_ROOT / "models" / "syncan" / "best"
        initial = PROJECT_ROOT / "models" / "syncan" / "initial"
        if best.exists():
            model_dir = best
            print(f"Using model: {model_dir}")
        elif initial.exists():
            model_dir = initial
            print(f"Using model: {model_dir}")
        else:
            print("Error: no model found at models/syncan/best/ or models/syncan/initial/")
            print("Run syncan/1_train.py first.")
            sys.exit(1)

    # ---- load signal labels ----
    signal_labels = np.load(
        PROCESSED / "signal_columns.npy", allow_pickle=True
    ).tolist()

    # ---- correlation ----
    print("Computing Pearson correlation matrix on training data...")
    train_all = np.load(PROCESSED / "train_signals.npy")
    corr = np.corrcoef(train_all, rowvar=False)
    del train_all

    cross_id_pairs = compute_cross_id_pairs(corr, signal_labels)

    print(f"\nTop 5 cross-ID correlated signal pairs:")
    print(f"{'Rank':<5s} {'Signal A':<15s} {'Signal B':<15s} {'|r|':<8s} {'r':<8s}")
    print("-" * 53)
    for rank, (i, j, abs_r) in enumerate(cross_id_pairs[:5], 1):
        print(
            f"{rank:<5d} {signal_labels[i]:<15s} {signal_labels[j]:<15s} "
            f"{abs_r:<8.4f} {corr[i, j]:<8.4f}"
        )

    print(f"\nTop 10 cross-ID pairs used for mixed/suppress-plateau attacks:")
    top_pairs = [(i, j) for i, j, _ in cross_id_pairs[:10]]
    for p_idx, (i, j) in enumerate(top_pairs):
        print(
            f"  Pair {p_idx + 1:>2d}: {signal_labels[i]:<15s} <-> "
            f"{signal_labels[j]:<15s}  (|r|={corr[i, j]:.4f})"
        )

    print(f"\nPlateau attack groups (4 correlated signals, same-ID filter):")
    plateau_groups = build_plateau_groups(cross_id_pairs, signal_labels, num_groups=5)
    for g_idx, group in enumerate(plateau_groups):
        names = [f"{signal_labels[d]} (dim {d})" for d in group]
        print(f"  Group {g_idx + 1}: {', '.join(names)}")
    print(f"  ({len(plateau_groups)} groups, cycling for {args.num_attacks} attacks)")

    del corr, cross_id_pairs

    # ---- attack generation ----
    print(f"\nLoading normal test data...")
    normal_signals = np.load(PROCESSED / "test_normal_signals.npy")
    total_len = len(normal_signals)

    t0 = time.time()
    for scenario_idx, (scenario_name, generator_fn) in enumerate(SCENARIOS):
        scenario_seed = args.seed + scenario_idx
        print(f"Generating {scenario_name} (seed={scenario_seed})...")

        positions = generate_attack_positions(
            total_len, args.num_attacks, args.attack_duration, scenario_seed
        )
        if positions:
            print(
                f"  {len(positions)} attacks, first at offset={positions[0][0]}, "
                f"last at offset={positions[-1][0]}"
            )
        else:
            print(f"  No attacks generated (total_len={total_len} too short)")

        if scenario_name == "coordinated_plateau":
            sig = generator_fn(normal_signals, positions, plateau_groups, scenario_seed)
        else:
            sig = generator_fn(normal_signals, positions, top_pairs, scenario_seed)

        labels = build_labels(total_len, positions)
        np.save(PROCESSED / f"test_{scenario_name}_signals.npy", sig)
        np.save(PROCESSED / f"test_{scenario_name}_labels.npy", labels)

        actual_anomalies = int(labels.sum())
        perc_anomalous = 100.0 * actual_anomalies / total_len
        print(f"  Anomalous timesteps: {actual_anomalies} ({perc_anomalous:.2f}%)")
        print(f"  Durations: min={min(e-s for s,e in positions)}, "
              f"max={max(e-s for s,e in positions)}")

    gen_time = time.time() - t0
    print(f"Generation complete ({gen_time:.1f}s)")

    del normal_signals
    if device.type == "cuda":
        gc.collect()

    # ---- evaluation ----
    print(f"\n{'=' * 70}")
    print(f"EVALUATION")
    print(f"{'=' * 70}")
    print(f"Loading model from {model_dir}...")

    registry = SynCANRegistry(base_dir=model_dir)
    model, config = registry.get_model(device=device)
    print(
        f"Model: window_size={config.window_size}, "
        f"scoring_mode={config.scoring_mode}"
    )

    print(f"Scoring training data for threshold calibration...")
    t0 = time.time()
    train_data = np.load(PROCESSED / "train_signals.npy")
    train_scores = score_batch(
        model, train_data,
        window_size=config.window_size,
        device=device,
        scoring_mode=config.scoring_mode,
        batch_size=args.score_batch_size,
    )
    del train_data
    print(f"  Done ({time.time() - t0:.1f}s) — train_scores shape={train_scores.shape}")

    baselines = compute_feature_baselines(train_scores)
    print(f"  Feature baselines: shape={baselines.shape}")
    train_mean = np.mean(train_scores, axis=0)
    train_std = np.std(train_scores, axis=0)
    train_std = np.maximum(train_std, 1e-8)

    # ---- score normal data for TNR ----
    print("Scoring normal test data for TNR estimation...")
    normal_sig = np.load(PROCESSED / "test_normal_signals.npy")
    normal_scores = score_batch(
        model, normal_sig,
        window_size=config.window_size,
        device=device,
        scoring_mode=config.scoring_mode,
        batch_size=args.score_batch_size,
    )
    normal_score_1d = np.mean(normal_scores, axis=1)
    print(f"  normal_score_1d shape={normal_score_1d.shape}")
    normal_intervals = sample_normal_intervals(
        len(normal_sig), args.num_attacks, args.attack_duration, seed=42
    )
    print(f"  Sampled {len(normal_intervals)} normal intervals of {args.attack_duration} steps")

    results = {}
    recall_progressions = {}
    attribution_results: dict[str, dict] = {}
    q_metrics: dict[str, dict] = {}
    for scenario_idx, (scenario_name, _) in enumerate(SCENARIOS):
        scenario_seed = args.seed + scenario_idx
        print(f"\nEvaluating {scenario_name}...")
        sig_path = PROCESSED / f"test_{scenario_name}_signals.npy"
        lbl_path = PROCESSED / f"test_{scenario_name}_labels.npy"

        test_sig = np.load(sig_path)
        test_lbl = np.load(lbl_path).astype(float)

        t0 = time.time()
        test_scores = score_batch(
            model, test_sig,
            window_size=config.window_size,
            device=device,
            scoring_mode=config.scoring_mode,
            batch_size=args.score_batch_size,
        )
        score_1d = np.mean(test_scores, axis=1)
        print(f"  Scored ({time.time() - t0:.1f}s), score_1d shape={score_1d.shape}")

        cal = calibrate_threshold(
            train_scores=train_scores,
            test_scores=test_scores,
            labels=test_lbl,
            method=args.method,
            pot_params=POTParams(q=1e-3, level=0.99, scale=1.0),
            percentile=99.0,
        )

        threshold = float(cal["threshold"])
        metrics = evaluate(test_scores, test_lbl, threshold)
        metrics["threshold"] = threshold

        n_anom = int(test_lbl.sum())
        print(
            f"  Global: F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}  AUC={metrics['roc_auc']:.4f}  "
            f"thresh={threshold:.6f}  anomalies={n_anom}"
        )

        results[scenario_name] = metrics

        # ---- interval detection (>= Q% of interval flagged) ----
        positions = generate_attack_positions(
            len(test_lbl), args.num_attacks, args.attack_duration, scenario_seed
        )
        q_iv = evaluate_intervals(score_1d, threshold, positions)
        tnr_result = compute_tnr(normal_score_1d, threshold, normal_intervals)
        q_metrics[scenario_name] = {"attack": q_iv, "tnr": tnr_result}
        r25 = q_iv["interval_recall"].get("0.25", 0.0)
        t25 = tnr_result["tnr"].get("0.25", 1.0)
        print(f"  Interval detection: R@25%={r25:.4f}  TNR@25%={t25:.4f}  "
              f"({q_iv['total_intervals']} intervals)")

        cp = compute_recall_progression(
            score_1d, test_lbl, threshold, positions,
            checkpoints=[50, 100, 200, args.attack_duration],
        )
        recall_progressions[scenario_name] = cp
        print(f"  Recall progression (cumulative from attack onset):")
        for cp_name in sorted(cp.keys(), key=int):
            print(f"    Within {int(cp_name):>3d} timesteps: recall={cp[cp_name]:.4f}")

        # ---- attribution verification ----
        adj_preds = adjust_predicts(score_1d, test_lbl, threshold)
        segments = build_segment_summaries(
            test_scores, adj_preds, baselines,
            feature_labels=signal_labels,
        )

        if scenario_name == "coordinated_plateau":
            attacked_dims = {d for g in plateau_groups for d in g}
        else:
            attacked_dims = {d for pair in top_pairs for d in pair}

        dim_hits: dict[int, int] = {d: 0 for d in attacked_dims}
        for seg in segments:
            top3 = [d["dim"] for d in seg["attributed_dimensions"][:3]]
            for d in attacked_dims:
                if d in top3:
                    dim_hits[d] += 1

        n_segs = len(segments)
        print(f"  Attribution ({n_segs} segments, attacked signals in top-3):")
        for dim in sorted(attacked_dims):
            if n_segs > 0:
                hit_rate = 100.0 * dim_hits[dim] / n_segs
                bar = "#" * int(hit_rate / 5) + "." * (20 - int(hit_rate / 5))
                print(f"    {signal_labels[dim]:<15s} (dim {dim:>2d}): "
                      f"{dim_hits[dim]:>2d}/{n_segs} ({hit_rate:>5.1f}%)  [{bar}]")

        attribution_results[scenario_name] = {
            dim: {"label": signal_labels[dim], "hits": dim_hits[dim], "total_segments": n_segs}
            for dim in attacked_dims
        }

        # ---- two-tailed (z-score) attribution (--two-tailed flag) ----
        if args.two_tailed:
            z2_dim_hits: dict[int, int] = {d: 0 for d in attacked_dims}
            for seg in segments:
                s, e = seg["segment_start"], seg["segment_end"]
                seg_scores = test_scores[s:e + 1]
                z_scores = (seg_scores - train_mean) / train_std  # (T, F)
                mean_z = np.mean(np.abs(z_scores), axis=0)  # (F,)
                top3_z = np.argsort(mean_z)[::-1][:3]
                for d in attacked_dims:
                    if d in top3_z:
                        z2_dim_hits[d] += 1

            print(f"  Two-tailed z-score ({n_segs} segments, attacked signals in top-3):")
            for dim in sorted(attacked_dims):
                if n_segs > 0:
                    hit_rate = 100.0 * z2_dim_hits[dim] / n_segs
                    bar = "#" * int(hit_rate / 5) + "." * (20 - int(hit_rate / 5))
                    print(f"    {signal_labels[dim]:<15s} (dim {dim:>2d}): "
                          f"{z2_dim_hits[dim]:>2d}/{n_segs} ({hit_rate:>5.1f}%)  [{bar}]")

        del test_sig, test_lbl, test_scores, score_1d

    # ---- summary table ----
    q_labels = ["0.10", "0.25", "0.50", "0.90"]
    print(f"\n{'=' * 108}")
    print(f"Coordinated Attack Evaluation Summary (method={args.method})")
    print(f"{'=' * 108}")
    # Pointwise section
    header = f"{'Attack':<20s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} "
    header += f"{'AUC':>8s} {'Thresh':>10s} {'Anom':>8s}"
    print(header)
    print(f"{'-' * 76}")
    for scenario_name, _ in SCENARIOS:
        m = results[scenario_name]
        lbl = np.load(PROCESSED / f"test_{scenario_name}_labels.npy")
        n_anom = int(lbl.sum())
        print(
            f"{SHORT_NAMES[scenario_name]:<20s} {m['f1']:>8.4f} "
            f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
            f"{m['roc_auc']:>8.4f} {m['threshold']:>10.6f} {n_anom:>8d}"
        )
        del lbl
    avg_f1 = np.mean([results[s]["f1"] for s, _ in SCENARIOS])
    avg_prec = np.mean([results[s]["precision"] for s, _ in SCENARIOS])
    avg_rec = np.mean([results[s]["recall"] for s, _ in SCENARIOS])
    print(f"{'AVERAGE':<20s} {avg_f1:>8.4f} {avg_prec:>8.4f} {avg_rec:>8.4f}")
    print()

    # Interval detection section
    print(f"{'=' * 108}")
    print(f"Interval Detection (attack detected if >= Q% of interval flagged)")
    print(f"{'=' * 108}")
    header = f"{'Attack':<20s}"
    header += "".join(f"{f'R@{q}':>8s}" for q in q_labels)
    header += f" {'Flagged%':>9s} {'Ints':>5s} {'TNR@25%':>9s}"
    print(header)
    print(f"{'-' * 108}")
    tnr_vals = {q: [] for q in q_labels}
    for scenario_name, _ in SCENARIOS:
        cm = q_metrics[scenario_name]
        iv = cm["attack"]
        vals = "".join(f"{iv['interval_recall'].get(q, 0):>8.4f}" for q in q_labels)
        t25 = cm.get("tnr", {}).get("tnr", {}).get("0.25", 1.0)
        print(
            f"{SHORT_NAMES[scenario_name]:<20s}{vals}"
            f"{iv['avg_flagged_fraction']:>9.4f}{iv['total_intervals']:>5d}{t25:>9.4f}"
        )
        for q in q_labels:
            tnr_vals[q].append(cm.get("tnr", {}).get("tnr", {}).get(q, 1.0))
    avg_flag = np.mean([q_metrics[s]["attack"]["avg_flagged_fraction"] for s, _ in SCENARIOS])
    total_ints = sum(q_metrics[s]["attack"]["total_intervals"] for s, _ in SCENARIOS)
    print(f"{'AVERAGE':<20s}", end="")
    for q in q_labels:
        avg_r = np.mean([q_metrics[s]["attack"]["interval_recall"].get(q, 0) for s, _ in SCENARIOS])
        print(f"{avg_r:>8.4f}", end="")
    print(f"{avg_flag:>9.4f}{total_ints:>5d}{'':>9s}")
    print(f"{'TNR (avg)':<20s}", end="")
    for q in q_labels:
        avg_t = np.mean(tnr_vals[q])
        print(f"{avg_t:>8.4f}", end="")
    print(f"{'':>9s}{'':>5s}{'':>9s}")
    print(f"  (TNR from {len(normal_intervals)} normal intervals of {args.attack_duration} steps)")
    print(f"{'=' * 108}")

    # ---- recall progression table ----
    print(f"\nRecall progression (cumulative from attack onset):")
    print(f"{'Attack':<20s} {'t=50':>8s} {'t=100':>8s} {'t=200':>8s} {'t=400':>8s}")
    print(f"{'-' * 55}")
    for scenario_name, _ in SCENARIOS:
        cp = recall_progressions[scenario_name]
        print(
            f"{SHORT_NAMES[scenario_name]:<20s} "
            f"{cp.get('50', 0):>8.4f} {cp.get('100', 0):>8.4f} "
            f"{cp.get('200', 0):>8.4f} {cp.get(str(args.attack_duration), 0):>8.4f}"
        )
    print(f"{'-' * 55}")
    print("If early recall (t=50) is high → model detects via correlation breakdown")
    print("If recall climbs over time → detection relies on accumulating error")

    # ---- save results ----
    save_data = {
        "method": args.method,
        "avg_f1": float(np.mean([results[s]["f1"] for s, _ in SCENARIOS])),
        "avg_precision": float(np.mean([results[s]["precision"] for s, _ in SCENARIOS])),
        "avg_recall": float(np.mean([results[s]["recall"] for s, _ in SCENARIOS])),
    }
    for scenario_name, _ in SCENARIOS:
        m = results[scenario_name]
        cp = recall_progressions[scenario_name]
        attr = attribution_results.get(scenario_name, {})
        cm = q_metrics.get(scenario_name, {})
        save_data[scenario_name] = {
            "f1": float(m["f1"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "roc_auc": float(m["roc_auc"]),
            "threshold": float(m["threshold"]),
            "TP": int(m["TP"]),
            "TN": int(m["TN"]),
            "FP": int(m["FP"]),
            "FN": int(m["FN"]),
            "recall_progression": {k: float(v) for k, v in cp.items()},
            "interval_detection": cm.get("attack"),
            "tnr": cm.get("tnr"),
        }
        if attr:
            save_data[scenario_name]["attribution"] = attr

    json_path = model_dir / "eval_results_coordinated.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
