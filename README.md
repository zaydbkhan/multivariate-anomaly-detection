# Striim AI Prototype: TranAD for CAN Bus Intrusion Detection

This repository is a fork of the [Striim Labs TranAD multivariate anomaly detection prototype](https://github.com/striim-labs/multivariate-anomaly-detection), adapting it to intrusion detection on the vehicle Controller Area Network (CAN) bus using the [SynCAN dataset](https://github.com/etas/SynCAN) (Hanselmann et al., IEEE Access 2020).

**Motivation**
As automotives become increasingly complex, and with full autonomous vehicles on the horizon, it is increasingly important that we secure our vehicles against attacks on their internal communication systems. The most prevalent of these is the CAN bus, which handles low-level signal outputs between devices. While previous detection models (CANet) have been shown to effectively detect single-signal attacks, they are not designed to detect subtler coordinated attacks on multiple signals. We apply the TranAD prototype to the SynCAN dataset to address these cases.

**Overview**
We adapt the original TranAD workflow for SMD to SynCAN, synchonizing individual signals of different frequencies using forward-fill preprocessing. We account for SynCAN's specific structure by increasing window sizes, retuning anomaly threshold parameters, and introducing alternative grid sweep parameters. Finally, we synthetically generate and evaluate against multi-signal coordinated attack scenarios. The results show that TranAD detects coordinated attacks substantially more reliably than single-target attacks, with interval-level F1 of 0.92–0.99 vs 0.70 average on the standard SynCAN benchmark.

The modeling approach is based on: Tuli, S., Casale, G., & Jennings, N. R. (2022). "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data." *PVLDB*, 15(6), 1201-1214.

The SynCAN dataset was introduced in: Hanselmann, M., Strauss, T., Dormann, K., & Ulmer, H. (2020). "CANet: An Unsupervised Intrusion Detection System for High Dimensional CAN Bus Data." *IEEE Access*, 8, 58194-58205.

---

## Dataset

SynCAN is a CAN bus dataset released by ETAS (Robert Bosch GmbH) as a benchmark for unsupervised CAN intrusion detection. It contains 16.5 hours of normal driving data for training and 7.5 hours of evaluation data across six test files — one normal and five attack scenarios.

| Split | Duration | Samples | Description |
|---|---|---|---|
| Train | 16.5 hours | 4,140,288 | Normal driving only |
| Test — normal | ~1.25 hours | ~300,000 | No attacks |
| Test — plateau | ~1.25 hours | ~300,000 | Single signal frozen to constant value |
| Test — continuous | ~1.25 hours | ~300,000 | Single signal slowly drifting |
| Test — playback | ~1.25 hours | ~300,000 | Single signal replaced with recorded values |
| Test — suppress | ~1.25 hours | ~300,000 | Single ECU silenced |
| Test — flooding | ~1.25 hours | ~300,000 | Single ID injected at high frequency |

The dataset comprises 10 CAN IDs with 20 signals total. Different IDs broadcast at different frequencies, a structural property of CAN traffic that the original CANet architecture exploits directly. To apply this data to TranAD, we forward-fill values during preprocessing (use the last known value for each signal at each timestamp).

We additionally introduce 3 coordinated attack scenarios:

| Split | Samples | Description |
|---|---|---|
| Test — coordinated plateau | ~300,000 | 4 correlated signals simultaneously frozen to constant values |
| Test — coordinated mixed | ~300,000 | Plateau on some signals, continuous change on correlated counterparts |
| Test — coordinated suppress+plateau | ~300,000 | One signal suppressed, correlated signal simultaneously frozen |

These scenarios are synthetically injected into the normal test file. Signal groups are selected by computing cross-ID Pearson correlations on the training data and grouping the most correlated signals from different IDs. Each scenario injects 20 intervals of 400 timesteps (2.67% anomaly rate), with per-signal deviations chosen to remain individually within normal bounds while collectively inconsistent. Attacked signals are known exactly, enabling direct attribution accuracy evaluation not possible with the standard SynCAN test files.

---

## Results

### Single-target attacks

All results use the best-config model (`window_size=140`, `n_layers=2`, `n_heads=5`, `d_feedforward=8`, `exponential_decay`, `averaged`) evaluated with adjusted POT thresholding (`q=1e-3`).

| Attack | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| Plateau | 0.731 | 0.779 | 0.688 | 0.825 |
| Continuous change | 0.746 | 0.896 | 0.639 | 0.814 |
| Playback | 0.750 | 0.859 | 0.666 | 0.825 |
| Suppress | 0.595 | 0.995 | 0.424 | 0.712 |
| Flooding | 0.687 | 0.999 | 0.523 | 0.762 |
| **Average** | **0.702** | **0.906** | **0.588** | — |

For reference, the F1-max upper bound (oracle threshold per attack type):

| Attack | F1-max | AUC |
|---|---|---|
| Plateau | 0.731 | 0.825 |
| Continuous change | 0.751 | 0.818 |
| Playback | 0.757 | 0.830 |
| Suppress | 0.711 | 0.796 |
| Flooding | 0.729 | 0.793 |
| **Average** | **0.736** | — |

The gap between adjusted POT (0.702) and F1-max (0.736) indicate that the deployed threshold is slightly conservative. Across all attack types, precision is high (0.78–0.99) and recall is moderate (0.42–0.69), reflecting a threshold that minimizes false alarms at the cost of some missed detections.

### Coordinated multi-signal attacks

| Attack | Global F1 | Precision | Recall | AUC | Interval F1 | Interval Precision | Interval Recall |
|---|---|---|---|---|---|---|---|
| Coordinated plateau | 0.810 | 0.680 | 1.000 | 0.994 | **0.986** | 0.977 | 0.998 |
| Coordinated mixed | 0.758 | 0.610 | 1.000 | 0.991 | **0.921** | 0.883 | 0.968 |
| Coordinated suppress+plateau | 0.789 | 0.675 | 0.950 | 0.969 | **0.939** | 0.932 | 0.948 |
| **Average** | **0.786** | **0.655** | **0.983** | — | **0.949** | **0.931** | **0.971** |

F1-max upper bound (oracle threshold per attack type):

| Attack | Global F1 | Precision | Recall | AUC | Interval F1 | Interval Precision | Interval Recall |
|---|---|---|---|---|---|---|---|
| Coordinated plateau | 0.999 | 0.998 | 1.000 | 1.000 | 0.907 | 0.997 | 0.883 |
| Coordinated mixed | 0.999 | 0.998 | 1.000 | 1.000 | 0.747 | 0.996 | 0.620 |
| Coordinated suppress+plateau | 0.940 | 0.929 | 0.950 | 0.974 | 0.903 | 0.948 | 0.893 |
| **Average** | **0.979** | **0.975** | **0.983** | — | **0.853** | **0.980** | **0.799** |

**Global F1** is computed pointwise as usual. While global precision is dragged down by false positives, we suspect that a persistence filter would supress most of these results in a real deployment. The high recall indicates that most anomalous segments are flagged.

**Interval F1** redefines the detection unit from a timestep to an attack interval: an interval is detected if the model flags enough of it, and a false positive only counts if spurious detections cluster into something resembling a complete interval. This approximates what a production system with persistence filtering would see. **Interval F1 of 0.92–0.99 on coordinated attacks vs 0.70 average on single-target attacks is the primary result of this fork.**

The F1-max comparison reveals an important property of the deployed threshold. The oracle threshold for coordinated mixed is 0.042 — roughly 19× the POT threshold — because the pointwise optimal threshold only needs to clear the highest-scoring timesteps to maximize global F1. At that level, many attack intervals have no timesteps that score that high, and interval recall drops to 0.62. The deployed POT threshold at 0.002 catches the full range of elevated scores, achieving higher interval F1 than the oracle despite lower global F1. This is a concrete case where the deployed threshold is better calibrated for operational use than the pointwise-optimal alternative. The AUC values of 1.000 for plateau and mixed confirm that perfect score separation exists in the raw scores — the gap between global and interval F1 is a threshold and false alarm question, not a model capability question.

Plateau attacks are detected within 50 timesteps of onset at 98.1% interval recall, reflecting near-immediate detection of cross-signal correlation breakdown. Mixed attacks accumulate more slowly since the continuous change component starts within normal bounds.

| Attack | Recall at t=50 | t=100 | t=200 | t=400 |
|---|---|---|---|---|
| Coordinated plateau | 0.981 | 0.991 | 0.995 | 0.998 |
| Coordinated mixed | 0.818 | 0.882 | 0.936 | 0.968 |
| Coordinated suppress+plateau | 0.931 | 0.941 | 0.945 | 0.948 |
| **Average** | **0.910** | **0.938** | **0.959** | **0.971** |

### Root cause attribution

Two attribution methods are available. The standard **elevation ratio** compares each channel's reconstruction error against its training baseline. Channels being reconstructed much worse than normal are flagged as likely causes. However, for freeze-type attacks (plateau, suppress), the attacked signal can become easier to predict than normal (a frozen constant is trivially reconstructable) so elevation ratio under-attributes the targeted channels.

The **two-tailed z-score** method also flags channels that reconstruct significantly more cleanly than baseline, catching this pattern. In the coordinated plateau evaluation, two-tailed attribution correctly surfaces `id5_Signal2` as a top contributor in 62.5% of detected segments vs 1.8% for elevation ratio alone.

SynCAN does not publish which specific signals are attacked in each test file, so attribution accuracy on single-target attacks is not directly evaluated. For coordinated attacks, where the attacked signals are known, two-tailed z-score attribution improves on elevation ratio in some cases. However, neither method reliably identifies all attacked channels across scenarios, even when evaluating with f1_max. More work

---

## Detection methodology

### TranAD architecture

TranAD uses a transformer encoder-decoder with two-phase self-conditioning. Phase 1 produces a standard reconstruction of the input window across all 20 signal channels. The per-channel reconstruction errors from Phase 1 are concatenated with the original input and fed into Phase 2, which reconstructs again with explicit awareness of where Phase 1 struggled. Normal windows reconstruct well in both phases; anomalous windows accumulate compounding error across phases. The final anomaly score combines Phase 1 and Phase 2 errors (`scoring_mode=averaged`).

### Threshold calibration: POT on large datasets

The original TranAD prototype and the SMD baseline use POT (Peaks Over Threshold) to calibrate the anomaly threshold by fitting a Generalized Pareto Distribution to the tail of the training score distribution. With SMD's ~28K samples per machine this works well. SynCAN's training set is 4.1M samples — at this scale, POT's default false-positive-rate parameter (`q=1e-5`) encounters extreme tail values that skew the GPD fit upward, producing a threshold too conservative to be useful (average F1=0.37 for single target attacks on an older but similarly performing version of the model).

Adjusting to `q=1e-3` produces a threshold that independently converges with the 99th-percentile fixed threshold (both land at ~0.0025), validating the calibration.

| Threshold method | Average F1 | Notes |
|---|---|---|
| Default POT (`q=1e-5`) | 0.37 | Overcalibrated at 4.1M samples |
| Adjusted POT (`q=1e-3`) | 0.70 | Converges with 99th-percentile |
| 99th-percentile (fixed) | 0.71 | Independent validation |
| F1-max (oracle) | 0.74 | Upper bound; uses ground truth labels |

### Hyperparameter sweep

The original TranAD paper holds `n_heads` and `n_layers` fixed across experiments for consistent benchmarking against other published methods. This sweep treats them as tunable parameters, since there is no prior published TranAD result on SynCAN to compare against and the dataset's structure (20 signals, variable-frequency CAN IDs, short attack intervals) differs meaningfully from SMD.

The quick sweep (16 targeted configurations, `--quick`) varies `window_size`, `lr`, `n_layers`, `n_heads`, and `d_feedforward` with `loss_weighting` and `scoring_mode` fixed at `exponential_decay` and `averaged` — confirmed from the original prototype and stable on SynCAN. Key findings:

- `n_layers=2` consistently outperforms both 1 and 3. Three layers underperform on SynCAN, suggesting the dataset's correlation structure doesn't benefit from added depth.
- `n_heads=5` outperforms `n_heads=10` at `window_size=140`. With 20 channels and 10 heads, each head attends to 2 channels on average; reducing to 5 gives each head richer multi-channel context.
- `d_feedforward=32` collapses at `window_size=140` (F1=0.42). Larger feedforward capacity at longer windows overfits the 10% subsample.
- `window_size=140` marginally outperforms `window_size=100`; `window_size=60` underperforms unless paired with a larger network.

Winning configuration: `window_size=140`, `n_layers=2`, `n_heads=5`, `d_feedforward=8`, `lr=0.001`.

---

## Comparison with CANet

CANet (Hanselmann et al., 2020) is the architecture SynCAN was designed to benchmark and achieves approximately 0.95 average F1 on the single-target attack types. The two models are architecturally complementary rather than directly competing.

CANet uses per-ID LSTM subnetworks that update only when each ID's message arrives, making it well-suited to single-target, per-ECU attacks and naturally handling CAN's variable-frequency broadcast structure. TranAD operates across all 20 signals simultaneously, making it well-suited to coordinated attacks where the anomaly manifests as correlation breakdown across ECUs — a threat class the standard single-target SynCAN benchmark was not designed to evaluate.

The results reflect this division. On single-target attacks, CANet's per-ID architecture has a structural advantage: a suppressed ECU goes silent, which a per-ID model detects from the absence of messages, while TranAD's forward-filled preprocessing partially obscures the same signature. On coordinated attacks, TranAD's cross-signal reconstruction detects correlation breakdown that per-ID models cannot reliably identify. CANet's own authors note that suppress and flooding attacks are most efficiently handled by rule-based frequency monitoring, and recommend pairing any reconstruction-based model with such rules.

We theorize that a hybrid architecture combining both approaches — per-ID LSTM for localized single-ECU attacks, TranAD for correlated multi-ECU attacks, and frequency monitoring for suppress and flooding — covers the full attack surface more completely than either model alone.

---

## Limitations

- **Single-target sweep.** The hyperparameter sweep optimizes on the five standard SynCAN attack types, all of which are single-target. There is no guarantee that the configuration best suited for single-target detection is optimal for coordinated attack detection. A sweep that includes coordinated attack F1 as an evaluation criterion is a natural next step.
- **Synthetic coordinated attacks.** The coordinated attack evaluation uses injected attacks rather than recorded real-world multi-ECU compromises. The attack design reflects known CAN correlation structure, but real coordinated attacks may have different signatures.
- **Forward-fill preprocessing.** The synchronized matrix representation loses the inter-arrival timing that CANet uses. Frequency-based attacks (suppress, flooding) are more naturally addressed by monitoring message arrival rates, which TranAD does not do.
- **Synthetic dataset.** SynCAN's authors note it is "somewhat cleaner than in the real case." Performance on real vehicle CAN traffic is expected to be lower.

---

## Future work

The natural next step is applying the [Striim Labs LSTM autoencoder prototype](https://github.com/striim-labs/lstm-autoencoder-spark-kafka) to SynCAN in a CANet-style per-ID configuration — one LSTM per CAN ID, updating only when that ID broadcasts, with a shared latent vector for cross-ID reconstruction. This addresses the variable-frequency preprocessing limitation directly and is the architecture best matched to single-target per-ECU attacks.

Given TranAD's strength on coordinated attacks and a per-ID LSTM's strength on localized attacks, a hybrid architecture covers the full attack surface: per-ID LSTM for individual signal anomalies, TranAD for correlated multi-signal deviations, and rule-based frequency monitoring for suppress and flooding.

---

## Project Structure

```
multivariate-anomaly-detection/
│
├── syncan/                                # Numbered scripts -- the canonical workflow
│   ├── 0_verify_setup.py                  # Download SynCAN, verify environment
│   ├── 1_train.py                         # Train baseline or best config
│   ├── 2_evaluate.py                      # Evaluate model, print per-attack metrics
│   ├── 3_streaming_app.py                 # FastAPI real-time scoring API (Docker)
│   ├── 4_grid_sweep.py                    # Hyperparameter sweep + retrain best config
│   └── 5_coordinated_attack.py            # Coordinated attack generation and evaluation
│
├── code/                                  # Original SMD workflow (unchanged)
│   ├── 0_verify_setup.py
│   ├── 1_train_model.py
│   ├── 2_evaluate_model.py
│   ├── 3_streaming_app.py
│   └── 4_grid_sweep.py
│
├── configs/                               # Hyperparameter configs for sweep and training
│
├── notebooks/                             # Interactive walkthroughs
│   ├── data_exploration.ipynb
│   └── model_design.ipynb
│
├── src/                                   # Reusable library code
│   ├── model.py                           # TranADConfig, TranADNet, transformer layers
│   ├── train.py                           # Shared training loop
│   ├── scorer.py                          # Scoring, POT thresholding, attribution
│   ├── preprocess.py                      # Data loading, normalization, windowing
│   ├── registry.py                        # Model loading and caching
│   ├── spot.py                            # SPOT algorithm (extreme value theory)
│   ├── schemas.py                         # Pydantic v2 request/response models
│   └── utils.py                           # Sliding window, device selection
│
├── data/smd/                              # Server Machine Dataset (raw and processed gitignored)
├── data/syncan/                           # SynCAN dataset (raw and processed gitignored)
│
├── models/tranad/                         # Prebuilt SMD reference artifacts
│   ├── machine-1-1/
│   ├── machine-2-1/
│   ├── machine-3-2/
│   └── machine-3-7/
│
├── samples/                               # Sample score requests for the REST API
├── striim/                                # Striim integration: TQL, Open Processor, build scripts
├── docker-compose.rest.yml
├── docker-compose.demo.yml
├── Dockerfile
├── pyproject.toml
├── STRIIM.md                              # Striim pipeline setup guide
└── TECHNICAL.md                           # Detailed technical reference
```

The scripts under `syncan/` are the first-class path for this fork. The original `code/` scripts are unchanged and reproduce the SMD baseline from the parent prototype.

---

## Prerequisites

- **Python 3.11+**
- **uv** (Python package manager):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Docker** (optional, for containerized deployment)
- **CUDA-capable GPU recommended.** CPU and Apple MPS are supported but significantly slower on the full 4.1M-sample SynCAN training set.

---

## Going through the code

### 1. Install dependencies and verify setup

```bash
git clone <repo-url>
cd multivariate-anomaly-detection
uv sync
uv run python syncan/0_verify_setup.py
```

`0_verify_setup.py` downloads SynCAN, preprocesses the training and test splits into normalized arrays, and prints a dataset summary.

> **Note on dataset size:** SynCAN contains 4.1 million training samples across 20 signal channels — roughly 150× a single SMD machine. Full-dataset training takes approximately 2 minutes per epoch on a modern GPU. The `--subsample` flag (default: 10%) is used for hyperparameter sweeps.

### 2. Train a baseline

```bash
uv run python syncan/1_train.py
```

Trains a baseline model using the best-known fixed configuration (`exponential_decay` loss weighting, `averaged` scoring mode, `lr=0.001`, `window_size=140`, `n_layers=2`, `n_heads=5`, `d_feedforward=8`, 5 epochs) on 10% of the training data. Checkpoint saved to `models/syncan/initial/`.

For a full-dataset baseline:

```bash
uv run python syncan/1_train.py --full
```

### 3. Evaluate the baseline

```bash
uv run python syncan/2_evaluate.py
```

Prints per-attack-type F1, precision, recall, AUC, and threshold using adjusted POT (`q=1e-3`). To evaluate with a fixed 99th-percentile threshold:

```bash
uv run python syncan/2_evaluate.py --method percentile
```

### 4. Run the grid sweep

```bash
uv run python syncan/4_grid_sweep.py --quick
```

Runs 16 configurations varying `window_size`, `lr`, `n_layers`, `n_heads`, and `d_feedforward`, each trained for 5 epochs on 10% subsampled data. After the sweep, the winning configuration is retrained on the full dataset with early stopping (patience=5, max 30 epochs) and saved to `models/syncan/best/`.

To verify trial configurations without training:

```bash
uv run python syncan/4_grid_sweep.py --dry-run
```

### 5. Evaluate the best model

```bash
uv run python syncan/2_evaluate.py --model-dir models/syncan/best
```

### 6. Run the coordinated attack evaluation

```bash
uv run python syncan/5_coordinated_attack.py --two-tailed
```

Computes the cross-ID Pearson correlation matrix, selects the most correlated signal groups, injects 20 coordinated attack intervals per scenario into the normal test file, and evaluates the best model against each. See [Coordinated attack results](#coordinated-multi-signal-attacks) below.

---

## Streaming deployment

The repo includes a FastAPI server for real-time CAN bus scoring. It accepts a window of 20-channel data and returns an anomaly score, binary flag, and ranked contributing signal channels.

```bash
docker compose -f docker-compose.rest.yml up --build
```

```bash
# Health check
curl http://localhost:8000/health

# Score a window of CAN data
curl -s -X POST http://localhost:8000/score \
    -H 'Content-Type: application/json' \
    -d @samples/score_request.json | python -m json.tool
```

Interactive API docs at `http://localhost:8000/docs`.

```bash
docker compose -f docker-compose.rest.yml down
```

---

## Workflow reference

| Step | Script | Purpose |
|---|---|---|
| 0 | `syncan/0_verify_setup.py` | Download SynCAN, preprocess, verify environment |
| 1 | `syncan/1_train.py` | Train baseline (10% subsample by default) |
| 2 | `syncan/2_evaluate.py` | Evaluate any saved model, print per-attack metrics |
| 3 | `syncan/3_streaming_app.py` | FastAPI REST server (Docker only) |
| 4 | `syncan/4_grid_sweep.py` | Hyperparameter sweep + full retrain of winner |
| 5 | `syncan/5_coordinated_attack.py` | Coordinated attack generation and evaluation |

---

## License

BSD-3-Clause. See [LICENSE](LICENSE).