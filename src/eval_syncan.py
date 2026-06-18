"""
SynCAN-specific evaluation utilities: interval detection and sampling.

Provides CANet-style interval evaluation (Figure 6) and normal-interval
TNR estimation for both standard and coordinated attack evaluation.

All positions use exclusive-end convention: (start, end) with end exclusive,
matching numpy slice semantics and the existing 5_coordinated_attack.py.
"""

import numpy as np


def extract_attack_intervals(
    labels: np.ndarray,
    min_length: int = 2,
) -> list[tuple[int, int]]:
    """Find contiguous blocks of label=1 (attack intervals).

    Args:
        labels: 1D binary label array, shape (N,).
        min_length: Minimum interval length to include.

    Returns:
        List of (start, end) tuples with exclusive end.
    """
    binary = (labels > 0.5).astype(int)
    padded = np.concatenate([[0], binary, [0]])
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    intervals = []
    for s, e in zip(starts, ends):
        length = e - s
        if length >= min_length:
            intervals.append((int(s), int(e)))
    return intervals


def sample_normal_intervals(
    total_length: int,
    n_intervals: int,
    duration: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Sample intervals from normal data matching attack interval shape.

    All labels are assumed 0 (normal). Intervals are sampled uniformly
    from valid start positions.

    Args:
        total_length: Length of the data array.
        n_intervals: Number of intervals to sample.
        duration: Length of each interval (in timesteps).
        seed: Random seed for reproducibility.

    Returns:
        List of (start, end) tuples with exclusive end.
    """
    rng = np.random.default_rng(seed)
    max_start = total_length - duration
    if max_start < 0:
        return []

    max_possible = min(n_intervals, max_start + 1)
    starts = rng.choice(max_start + 1, size=max_possible, replace=False)
    starts.sort()
    return [(int(s), int(s + duration)) for s in starts]


def evaluate_intervals(
    score_1d: np.ndarray,
    threshold: float,
    positions: list[tuple[int, int]],
    q_values: list[float] | None = None,
) -> dict:
    """CANet-style interval evaluation at multiple Q thresholds.

    An attack interval is detected if >= Q% of its timesteps exceed the
    pointwise anomaly threshold.

    Args:
        score_1d: Aggregated anomaly scores, shape (N,).
        threshold: Pointwise anomaly threshold.
        positions: List of (start, end) attack intervals (exclusive end).
        q_values: Q thresholds to evaluate. Default [0.10, 0.25, 0.50, 0.90].

    Returns:
        dict with:
          - interval_recall: {q: float} for each Q
          - detected_intervals: {q: int} for each Q
          - total_intervals: int
          - avg_flagged_fraction: float
          - per_interval: list of {start, end, flagged_fraction}
    """
    if q_values is None:
        q_values = [0.10, 0.25, 0.50, 0.90]

    per_interval = []
    for start, end in positions:
        slice_ = score_1d[start:end]
        flagged = float(np.mean(slice_ > threshold)) if len(slice_) > 0 else 0.0
        per_interval.append({
            "start": int(start),
            "end": int(end),
            "flagged_fraction": flagged,
        })

    total = len(per_interval)
    if total == 0:
        return {
            "interval_recall": {f"{q:.2f}": 0.0 for q in q_values},
            "detected_intervals": {f"{q:.2f}": 0 for q in q_values},
            "total_intervals": 0,
            "avg_flagged_fraction": 0.0,
            "per_interval": [],
        }

    avg_flagged = float(np.mean([p["flagged_fraction"] for p in per_interval]))

    recall = {}
    detected = {}
    for q in q_values:
        n_detected = sum(1 for p in per_interval if p["flagged_fraction"] >= q)
        recall[f"{q:.2f}"] = n_detected / total
        detected[f"{q:.2f}"] = n_detected

    return {
        "interval_recall": recall,
        "detected_intervals": detected,
        "total_intervals": total,
        "avg_flagged_fraction": avg_flagged,
        "per_interval": per_interval,
    }


def compute_tnr(
    score_1d: np.ndarray,
    threshold: float,
    normal_intervals: list[tuple[int, int]],
    q_values: list[float] | None = None,
) -> dict:
    """Compute TNR/FPR on normal intervals at multiple Q thresholds.

    A normal interval is a false positive if >= Q% of its timesteps
    exceed the anomaly threshold. TNR is the fraction of normal
    intervals that are NOT false positives.

    Args:
        score_1d: Aggregated anomaly scores for normal data, shape (N,).
        threshold: Pointwise anomaly threshold.
        normal_intervals: List of (start, end) normal intervals (exclusive end).
        q_values: Q thresholds. Default [0.10, 0.25, 0.50, 0.90].

    Returns:
        dict with:
          - tnr: {q: float} for each Q
          - fpr: {q: float} for each Q
          - total_intervals: int
          - per_interval: list of {start, end, flagged_fraction, is_fp}
    """
    if q_values is None:
        q_values = [0.10, 0.25, 0.50, 0.90]

    per_interval = []
    for start, end in normal_intervals:
        slice_ = score_1d[start:end]
        flagged = float(np.mean(slice_ > threshold)) if len(slice_) > 0 else 0.0
        per_interval.append({
            "start": int(start),
            "end": int(end),
            "flagged_fraction": flagged,
        })

    total = len(per_interval)
    if total == 0:
        return {
            "tnr": {f"{q:.2f}": 0.0 for q in q_values},
            "fpr": {f"{q:.2f}": 0.0 for q in q_values},
            "total_intervals": 0,
            "per_interval": [],
        }

    tnr = {}
    fpr = {}
    for q in q_values:
        n_fp = sum(1 for p in per_interval if p["flagged_fraction"] >= q)
        fp_rate = n_fp / total
        fpr[f"{q:.2f}"] = fp_rate
        tnr[f"{q:.2f}"] = 1.0 - fp_rate

    # Add is_fp to per_interval at the finest Q (lowest = most sensitive)
    finest_q = min(q_values)
    for p in per_interval:
        p["is_fp"] = p["flagged_fraction"] >= finest_q

    return {
        "tnr": tnr,
        "fpr": fpr,
        "total_intervals": total,
        "per_interval": per_interval,
    }
