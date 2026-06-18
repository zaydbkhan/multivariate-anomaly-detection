"""
Verify Setup: Download SynCAN Data, Preprocess, and Check Artifacts.

Downloads the SynCAN dataset if missing, preprocesses it, and verifies
that all required data files exist.

Usage:
    uv run python syncan/0_verify_setup.py
    uv run python syncan/0_verify_setup.py --force-download
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess_syncan import (
    ATTACK_TYPES,
    preprocess_syncan,
    download_syncan_dataset,
)


def verify_imports() -> bool:
    """Check that all required packages are importable."""
    required = [
        "torch", "numpy", "pandas", "fastapi", "uvicorn", "pydantic",
        "sklearn", "scipy", "requests", "yaml", "tqdm", "rich",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  Missing packages: {', '.join(missing)}")
        print("  Run: uv sync")
        return False
    print(f"  All {len(required)} required packages importable")
    return True


def verify_data(
    raw_dir: Path, processed_dir: Path, force_download: bool = False
) -> bool:
    """Download and preprocess SynCAN data if needed, verify files exist."""

    # Check if processed data already exists
    expected_files = ["train_signals.npy", "train_labels.npy", "norm_params.npy", "signal_columns.npy"]
    for attack in ["normal"] + ATTACK_TYPES:
        expected_files.append(f"test_{attack}_signals.npy")
        expected_files.append(f"test_{attack}_labels.npy")

    all_exist = all((processed_dir / f).exists() for f in expected_files)

    if all_exist and not force_download:
        print(f"  Processed data: {len(expected_files)} files in {processed_dir}")
        return True

    # Download if raw data missing
    train_file = raw_dir / "train_1.csv"
    if not train_file.exists() or force_download:
        print("  Downloading SynCAN dataset...")
        download_syncan_dataset(raw_dir, force=force_download)
    else:
        n_files = len(list(raw_dir.glob("*.csv")))
        print(f"  Raw data: {n_files} CSV files in {raw_dir}")

    # Preprocess
    print("  Preprocessing SynCAN data...")
    try:
        shapes = preprocess_syncan(raw_dir, processed_dir)
        print(f"    Train: {shapes['train_signals']}")
        for attack in ["normal"] + ATTACK_TYPES:
            key = f"test_{attack}_signals"
            if key in shapes:
                print(f"    {key}: {shapes[key]}")
    except Exception as e:
        print(f"    Preprocessing FAILED: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify setup for SynCAN TranAD project"
    )
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download SynCAN data even if it exists")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "syncan" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "syncan" / "processed"

    print("=" * 60)
    print("SynCAN TranAD Anomaly Detection - Setup Verification")
    print("=" * 60)

    print("\n1. Checking dependencies...")
    deps_ok = verify_imports()

    print("\n2. Checking data...")
    data_ok = verify_data(raw_dir, processed_dir, args.force_download)

    print("\n" + "=" * 60)
    if deps_ok and data_ok:
        print("All checks passed! Ready to go.")
        print("\nNext steps:")
        print("  uv run python syncan/1_train.py")
    else:
        print("Some checks failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
