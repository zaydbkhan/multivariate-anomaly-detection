"""SynCAN Dataset Preprocessing.

SynCAN is a synthetic CAN bus dataset for vehicle intrusion detection.
https://github.com/etas/SynCAN

Format: long-format CSV with Label, Time (ms), ID, Signal1-4 columns.
10 CAN IDs (id1-id10) with 20 total signal channels:
  id1: 2, id2: 3, id3: 2, id4: 1, id5: 2,
  id6: 2, id7: 2, id8: 1, id9: 1, id10: 4

Training: 4 CSV files (train_1-4), all label=0 (normal driving only).
Test: 5 attack types + test_normal, each with binary label column.

Processing: forward-fill missing values per signal, resample to 15ms grid.
"""

import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from src.preprocess import normalize

logger = logging.getLogger(__name__)

BASE_URL = "https://github.com/etas/SynCAN/raw/master"

ID_SIGNAL_COUNTS = {
    "id1": 2, "id2": 3, "id3": 2, "id4": 1, "id5": 2,
    "id6": 2, "id7": 2, "id8": 1, "id9": 1, "id10": 4,
}

SIGNAL_COLUMNS = [
    f"id{n}_Signal{i}"
    for n in range(1, 11)
    for i in range(1, ID_SIGNAL_COUNTS[f"id{n}"] + 1)
]

# There are exactly 20 signal columns
assert len(SIGNAL_COLUMNS) == 20, f"Expected 20 signals, got {len(SIGNAL_COLUMNS)}"

TRAIN_FILES = ["train_1", "train_2", "train_3", "train_4"]
TEST_FILES = [
    "test_normal", "test_plateau", "test_continuous",
    "test_playback", "test_suppress", "test_flooding",
]
ATTACK_TYPES = ["plateau", "continuous", "playback", "suppress", "flooding"]

ALL_SIGNAL_VARS = ["Signal1", "Signal2", "Signal3", "Signal4"]
CSV_COLUMNS = ["Label", "Time", "ID", "Signal1", "Signal2", "Signal3", "Signal4"]

SIGNAL_VAR_RENAME = {f"Signal{i}_of_ID": f"Signal{i}" for i in range(1, 5)}


def download_syncan_dataset(data_dir: Path, force: bool = False) -> None:
    """Download SynCAN CSV zip files from GitHub.

    Args:
        data_dir: Output directory (e.g., data/syncan/raw).
        force: Re-download files even if they exist.
    """
    files = TRAIN_FILES + TEST_FILES
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SynCAN dataset: %d files", len(files))

    downloaded = 0
    skipped = 0
    for fname in tqdm(files, desc="Downloading SynCAN"):
        zip_path = data_dir / f"{fname}.zip"
        csv_path = data_dir / f"{fname}.csv"
        if csv_path.exists() and not force:
            skipped += 1
            continue
        url = f"{BASE_URL}/{fname}.zip"
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(data_dir)
            zip_path.unlink()
            downloaded += 1
        except Exception as e:
            logger.error("Failed to download %s: %s", url, e)

    logger.info("Download complete: %d downloaded, %d skipped", downloaded, skipped)


def _read_csv_with_normalized_columns(filepath: Path) -> pd.DataFrame:
    """Read SynCAN CSV and normalize column names."""
    reader = pd.read_csv(filepath, nrows=0)
    has_header = "Label" in reader.columns
    if has_header:
        df = pd.read_csv(filepath)
        df = df.rename(columns=SIGNAL_VAR_RENAME)
    else:
        df = pd.read_csv(filepath, header=None, names=CSV_COLUMNS)
    return df


def load_syncan_csv(
    filepath: Path, resolution_ms: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Load a SynCAN CSV file into aligned (N, 20) signal array and (N,) label array.

    Handles the long-to-wide conversion: raw CSV has one row per CAN message
    (each row has values for a single ID at a single timestamp). This function
    pivots to wide format where each row represents all 20 signal channels
    at one timestamp, with forward-fill for channels that don't update.

    Args:
        filepath: Path to .csv file.
        resolution_ms: Resample grid resolution in milliseconds (default 15ms).

    Returns:
        (signals, labels): (N, 20) float64 array, (N,) float64 label array.
    """
    df = _read_csv_with_normalized_columns(filepath)

    df["Time"] = (df["Time"] / resolution_ms).round() * resolution_ms

    id_vars = ["Time", "Label", "ID"]
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=ALL_SIGNAL_VARS,
        var_name="SignalVar",
        value_name="Value",
    )
    df_melted = df_melted.dropna(subset=["Value"])

    df_melted["Channel"] = df_melted["ID"] + "_" + df_melted["SignalVar"]

    valid_set = set(SIGNAL_COLUMNS)
    df_melted = df_melted[df_melted["Channel"].isin(valid_set)]

    wide = df_melted.pivot_table(
        index="Time", columns="Channel", values="Value", aggfunc="last"
    )

    for col in SIGNAL_COLUMNS:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide[SIGNAL_COLUMNS]

    labels_per_time = df.groupby("Time")["Label"].max()

    wide = wide.sort_index()
    wide = wide.ffill()
    labels_per_time = labels_per_time.reindex(wide.index).ffill()

    mask = wide.isna().any(axis=1)
    if mask.any():
        n_dropped = int(mask.sum())
        if n_dropped > 0:
            logger.debug("Dropped %d rows with NaN (start of time series)", n_dropped)
        wide = wide[~mask]
        labels_per_time = labels_per_time[~mask]

    return wide.values.astype(np.float64), labels_per_time.values.astype(np.float64)


def load_syncan_train(
    data_dir: Path, resolution_ms: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all training files into one array.

    Returns:
        (signals, labels) where signals.shape = (N_total, 20).
    """
    all_signals: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for fname in TRAIN_FILES:
        csv_path = data_dir / f"{fname}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Training file not found: {csv_path}")
        sig, lbl = load_syncan_csv(csv_path, resolution_ms)
        all_signals.append(sig)
        all_labels.append(lbl)

    return np.concatenate(all_signals, axis=0), np.concatenate(all_labels, axis=0)


def load_syncan_test(
    data_dir: Path, attack_type: str, resolution_ms: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Load a single SynCAN test file by attack type.

    Args:
        data_dir: Directory containing CSV files.
        attack_type: One of "normal", "plateau", "continuous",
                     "playback", "suppress", "flooding".
        resolution_ms: Resample grid resolution.

    Returns:
        (signals, labels) where signals.shape = (N, 20).
    """
    fname = f"test_{attack_type}"
    csv_path = data_dir / f"{fname}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Test file not found: {csv_path}")
    return load_syncan_csv(csv_path, resolution_ms)


def preprocess_syncan(
    raw_dir: Path, output_dir: Path, resolution_ms: float = 15.0
) -> dict:
    """Full preprocessing: load, normalize, and save .npy files.

    Saves:
        train_signals.npy, train_labels.npy (all normal)
        test_normal_signals.npy, test_normal_labels.npy
        test_{attack}_signals.npy, test_{attack}_labels.npy  (for each attack)
        norm_params.npy  (min/max from training data)
        signal_columns.npy  (the 20 column names)

    Returns:
        dict with shapes of saved arrays.
    """
    logger.info("Loading training data...")
    train_sig, train_lbl = load_syncan_train(raw_dir, resolution_ms)
    logger.info("  Train: signals=%s, labels=%s, anomalies=%d",
                train_sig.shape, train_lbl.shape, int(train_lbl.sum()))

    train_norm, min_vals, max_vals = normalize(train_sig)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_signals.npy", train_norm)
    np.save(output_dir / "train_labels.npy", train_lbl)
    np.save(output_dir / "norm_params.npy", np.stack([min_vals, max_vals]))
    np.save(output_dir / "signal_columns.npy", np.array(SIGNAL_COLUMNS, dtype=object))

    shapes = {"train_signals": train_norm.shape, "train_labels": train_lbl.shape}

    for attack in ["normal"] + ATTACK_TYPES:
        sig, lbl = load_syncan_test(raw_dir, attack, resolution_ms)
        sig_norm, _, _ = normalize(sig, min_vals, max_vals)
        np.save(output_dir / f"test_{attack}_signals.npy", sig_norm)
        np.save(output_dir / f"test_{attack}_labels.npy", lbl)
        n_anom = int(lbl.sum())
        logger.info("  test_%s: signals=%s, labels=%s, anomalies=%d",
                    attack, sig_norm.shape, lbl.shape, n_anom)
        shapes[f"test_{attack}_signals"] = sig_norm.shape
        shapes[f"test_{attack}_labels"] = lbl.shape

    return shapes
