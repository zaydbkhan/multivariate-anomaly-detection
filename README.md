# Striim AI Prototype: TranAD for CAN Bus Intrusion Detection

This repository is a fork of the [Striim Labs TranAD multivariate anomaly detection prototype](https://github.com/striim-labs/multivariate-anomaly-detection), adapting it to intrusion detection on the vehicle Controller Area Network (CAN) bus using the [SynCAN dataset](https://github.com/etas/SynCAN) (Hanselmann et al., IEEE Access 2020).

**Motivation**
As automobiles become increasingly complex, and with full autonomous vehicles on the horizon, it is increasingly important that we secure our vehicles against attacks on their internal communication systems. The most prevalent of these is the CAN bus, which handles low-level signal outputs between devices. While previous detection models (CANet) have been shown to effectively detect single-signal attacks, they are not designed to detect subtler coordinated attacks on multiple signals. We apply the TranAD prototype to the SynCAN dataset to address these cases.

**Overview**
We adapt the original TranAD workflow for SMD to SynCAN, synchronizing individual signals of different frequencies using forward-fill preprocessing. We account for SynCAN's specific structure by increasing window sizes, retuning anomaly threshold parameters, and introducing alternative grid sweep parameters. Finally, we synthetically generate and evaluate against multi-signal coordinated attack scenarios. 

The results show that TranAD detects coordinated attacks substantially more reliably than single-target attacks, with a pointwise F1 of 0.90 vs 0.70 in the single-target case. More notably, TranAD achieves extremely high recall on interval-level true positive and true negative labeling, which is more applicable to real-world use cases where human review is impossible. We conclude that the best method for detecting CAN bus intrusions is a hybrid approach between per-ID LSTM models, basic per-signal frequency modeling, and multivariate models like TranAD to account for both single-signal and multi-signal attacks.

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

These scenarios are synthetically injected into the normal test file. Signal groups are selected by computing cross-ID Pearson correlations on the training data and grouping the most correlated signals from different IDs. Each scenario injects 100 intervals of 50–500 timesteps each (~9% anomaly rate), with per-signal deviations chosen to remain individually within normal bounds while collectively inconsistent. Attacked signals are known exactly, enabling direct attribution accuracy evaluation not possible with the standard SynCAN test files.

---

## Results

All results use the best-config model (`window_size=140`, `n_layers=2`, `n_heads=5`, `d_feedforward=8`, `exponential_decay`, `averaged`) evaluated with adjusted POT thresholding (`q=1e-3`).

### Single-target attacks

TNR (True Negative Rate) is scored with 50 normal intervals of length 427 (median of SynCAN attack lengths).

**Pointwise detection**

| Attack | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| Plateau | 0.731 | 0.779 | 0.688 | 0.825 |
| Continuous change | 0.746 | 0.896 | 0.639 | 0.814 |
| Playback | 0.750 | 0.859 | 0.666 | 0.825 |
| Suppress | 0.595 | 0.995 | 0.424 | 0.712 |
| Flooding | 0.687 | 0.999 | 0.523 | 0.762 |
| **Average** | **0.702** | **0.906** | **0.588** | — |

**Interval detection** (an attack interval is detected if ≥ Q% of its timesteps exceed the pointwise threshold)

| Attack | R@0.01 | R@0.02 | R@0.05 | R@0.10 | R@0.25 | R@0.50 | R@0.90 | Flagged% | Ints |
|---|---|---|---|---|---|---|---|---|---|
| Plateau | 0.629 | 0.621 | 0.578 | 0.578 | 0.500 | 0.457 | 0.431 | 0.479 | 116 |
| Continuous change | 0.615 | 0.615 | 0.604 | 0.573 | 0.438 | 0.271 | 0.000 | 0.263 | 96 |
| Playback | 0.634 | 0.613 | 0.548 | 0.527 | 0.430 | 0.258 | 0.140 | 0.307 | 93 |
| Suppress | 0.361 | 0.353 | 0.303 | 0.254 | 0.115 | 0.016 | 0.000 | 0.076 | 122 |
| Flooding | 0.483 | 0.449 | 0.381 | 0.314 | 0.237 | 0.203 | 0.144 | 0.220 | 118 |
| **Average** | **0.544** | **0.530** | **0.483** | **0.449** | **0.344** | **0.241** | **0.143** | **0.269** | — |
| **TNR (avg)** | **0.980** | **0.988** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | | |

**Recall progression** (cumulative pointwise recall within the first fraction of each interval's duration)

| Attack | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| Plateau | 0.448 | 0.464 | 0.475 | 0.479 |
| Continuous change | 0.018 | 0.085 | 0.176 | 0.263 |
| Playback | 0.306 | 0.293 | 0.308 | 0.307 |
| Suppress | 0.061 | 0.062 | 0.067 | 0.076 |
| Flooding | 0.241 | 0.230 | 0.227 | 0.220 |

As a multivariate model, TranAD predictably struggles with single-target attacks. The model detects plateau and playback attacks most reliably, while suppress and flooding attack types are harder for a reconstruction-based model as the original CANet paper indicates. The TNR row shows near-perfect specificity across all thresholds, indicating that false alarms are rare in practice. However, the low flagged fractions (especially continuous, suppress) show that detection within each interval is sparse rather than sustained, meaning the model catches only scattered timesteps even when it technically flags enough of the interval.

### Coordinated multi-signal attacks

TNR is scored with 100 normal intervals of length 285 (median synthetic attack length).

**Pointwise detection**

| Attack | F1 | Precision | Recall | AUC | Anomalies |
|---|---|---|---|---|---|---|
| Coordinated plateau | 0.9344 | 0.8770 | 1.0000 | 0.9926 | 28,707 |
| Coordinated mixed | 0.8490 | 0.7377 | 1.0000 | 0.9829 | 26,275 |
| Coordinated suppress+plateau | 0.9026 | 0.8752 | 0.9318 | 0.9587 | 29,455 |
| **Average** | **0.8954** | **0.8299** | **0.9773** | — | — |

**Interval detection**

| Attack | R@0.01 | R@0.02 | R@0.05 | R@0.10 | R@0.25 | R@0.50 | R@0.90 | Flagged% | Ints |
|---|---|---|---|---|---|---|---|---|---|---|
| Coordinated plateau | 1.0000 | 1.0000 | 1.0000 | 0.9900 | 0.9900 | 0.9800 | 0.9600 | 0.9747 | 100 |
| Coordinated mixed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8900 | 0.9718 | 100 |
| Coordinated suppress+plateau | 0.9200 | 0.9200 | 0.9200 | 0.9200 | 0.9200 | 0.9200 | 0.8700 | 0.8993 | 100 |
| **Average** | **0.9733** | **0.9733** | **0.9733** | **0.9700** | **0.9700** | **0.9667** | **0.9067** | **0.9486** | 300 |
| **TNR (avg)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | | |

**Recall progression**

| Attack | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| Coordinated plateau | 0.963 | 0.969 | 0.973 | 0.975 |
| Coordinated mixed | 0.890 | 0.944 | 0.962 | 0.972 |
| Coordinated suppress+plateau | 0.893 | 0.903 | 0.901 | 0.899 |
| **Average** | **0.915** | **0.938** | **0.945** | **0.949** |

The model achieves **high pointwise detection and near-perfect interval detection** across all coordinated attack types, with an average R@25% of 0.97 and average flagged fractions of 0.90–0.98. The gap between the single-target and multi-target results demonstrates that **TranAD's cross-signal reconstruction is substantially more sensitive to correlation breakdown across multiple channels than to individual signal deviations — the attack class that CANet-style per-ID models are structurally less equipped to detect**.

The recall progression shows that **detection is near-immediate**: 25% through the interval, average recall is already 0.915. Plateau attacks are detected earliest (0.963 at 25%), consistent with instantaneous correlation breakdown when multiple signals freeze simultaneously. Mixed attacks start lower (0.890) but climb steadily, reflecting the continuous change component that begins within normal bounds and accumulates over time.

We note that TNR is also perfect at this interval length, indicating that **true negative rate is robust to interval length** and supporting the reliability of this model in an environment where false positives may be particularly damaging.

### Root cause attribution

Two attribution methods are available. The standard **elevation ratio** compares each channel's reconstruction error against its training baseline. Channels being reconstructed much worse than normal are flagged as likely causes. However, for freeze-type attacks (plateau, suppress), the attacked signal can become easier to predict than normal (a frozen constant is trivially reconstructable) so elevation ratio under-attributes the targeted channels. The **two-tailed z-score** method also flags channels that **reconstruct significantly more cleanly than baseline**, catching this pattern and improving attributions.

**Coordinated plateau** — top-3 hit rate per dim, by group

| Group | Signal | Elevation | Two-tailed |
|---|---|---|---|
| 1 | id1_Signal1 | 9/20 (45%) | 12/20 (60%) |
|   | id2_Signal2 | 7/20 (35%) | 11/20 (55%) |
|   | id10_Signal4 | 14/20 (70%) | 16/20 (80%) |
|   | id5_Signal1 | 16/20 (80%) | 17/20 (85%) |
| 2 | id2_Signal1 | 10/20 (50%) | 14/20 (70%) |
|   | id3_Signal2 | 20/20 (100%) | 20/20 (100%) |
|   | id7_Signal1 | 6/20 (30%) | 11/20 (55%) |
|   | id10_Signal3 | 0/20 (0%) | 0/20 (0%) |
| 3 | id2_Signal3 | 12/20 (60%) | 15/20 (75%) |
|   | id8_Signal1 | 8/20 (40%) | 11/20 (55%) |
|   | id10_Signal1 | 15/20 (75%) | 15/20 (75%) |
|   | id4_Signal1 | 5/20 (25%) | 6/20 (30%) |
| 4 | id5_Signal2 | 7/20 (35%) | 8/20 (40%) |
|   | id6_Signal1 | 6/20 (30%) | 6/20 (30%) |
|   | id9_Signal1 | 15/20 (75%) | 13/20 (65%) |
|   | id10_Signal2 | 12/20 (60%) | 15/20 (75%) |
| 5 | id1_Signal2 | 11/20 (55%) | 15/20 (75%) |
|   | id7_Signal2 | 12/20 (60%) | 14/20 (70%) |
|   | id6_Signal2 | 5/20 (25%) | 15/20 (75%) |
|   | id3_Signal1 | 5/20 (25%) | 9/20 (45%) |

**Coordinated mixed** — top-3 hit rate per dim, by pair

| Pair | Signal | Elevation | Two-tailed |
|---|---|---|---|
| id1_Signal1 / id2_Signal2 | id1_Signal1 | 10/10 (100%) | 10/10 (100%) |
|   | id2_Signal2 | 10/10 (100%) | 10/10 (100%) |
| id2_Signal1 / id3_Signal2 | id2_Signal1 | 8/20 (40%) | 19/20 (95%) |
|   | id3_Signal2 | 18/20 (90%) | 20/20 (100%) |
| id2_Signal3 / id8_Signal1 | id2_Signal3 | 10/10 (100%) | 10/10 (100%) |
|   | id8_Signal1 | 10/10 (100%) | 10/10 (100%) |
| id4_Signal1 / id5_Signal1 | id4_Signal1 | 2/10 (20%) | 7/10 (70%) |
|   | id5_Signal1 | 10/10 (100%) | 10/10 (100%) |
| id5_Signal2 / id6_Signal1 | id5_Signal2 | 17/20 (85%) | 20/20 (100%) |
|   | id6_Signal1 | 20/20 (100%) | 20/20 (100%) |
| id6_Signal1 / id10_Signal3 | id6_Signal1 | 20/20 (100%) | 20/20 (100%) |
|   | id10_Signal3 | 19/19 (100%) | 18/19 (94.7%) |
| id3_Signal2 / id7_Signal1 | id3_Signal2 | 18/20 (90%) | 20/20 (100%) |
|   | id7_Signal1 | 18/20 (90%) | 20/20 (100%) |
| id9_Signal1 / id10_Signal1 | id9_Signal1 | 10/10 (100%) | 10/10 (100%) |
|   | id10_Signal1 | 10/10 (100%) | 10/10 (100%) |

**Coordinated suppress+plateau** — top-3 hit rate per dim, by pair

| Pair | Signal | Elevation | Two-tailed |
|---|---|---|---|
| id1_Signal1 / id2_Signal2 | id1_Signal1 | 7/10 (70%) | 8/10 (80%) |
|   | id2_Signal2 | 5/10 (50%) | 6/10 (60%) |
| id2_Signal1 / id3_Signal2 | id2_Signal1 | 3/20 (15%) | 7/20 (35%) |
|   | id3_Signal2 | 20/20 (100%) | 20/20 (100%) |
| id2_Signal3 / id8_Signal1 | id2_Signal3 | 2/5 (40%) | 2/5 (40%) |
|   | id8_Signal1 | 5/5 (100%) | 5/5 (100%) |
| id4_Signal1 / id5_Signal1 | id4_Signal1 | 0/7 (0%) | 0/7 (0%) |
|   | id5_Signal1 | 7/7 (100%) | 7/7 (100%) |
| id5_Signal2 / id6_Signal1 | id5_Signal2 | 17/20 (85%) | 18/20 (90%) |
|   | id6_Signal1 | 7/20 (35%) | 13/20 (65%) |
| id5_Signal2 / id10_Signal3 | id5_Signal2 | 17/20 (85%) | 18/20 (90%) |
|   | id10_Signal3 | 5/20 (25%) | 12/20 (60%) |
| id3_Signal2 / id7_Signal1 | id3_Signal2 | 20/20 (100%) | 20/20 (100%) |
|   | id7_Signal1 | 6/20 (30%) | 16/20 (80%) |
| id9_Signal1 / id10_Signal1 | id9_Signal1 | 6/10 (60%) | 10/10 (100%) |
|   | id10_Signal1 | 7/10 (70%) | 8/10 (80%) |

With the two-tailed improvement, attributions are fairly strong but still imperfect. Notably, **the mixed signal attribution is extremely strong**, rather than the 4-signal coordinated plateau we might expect to come out on top. The results seem to indicate that **TranAD performs better when signals fail in different ways**, but concede that there is not yet enough evidence to support any interpretation. We expect that better attribution results could be obtained through more focused training — see [Limitations](#limitations) below.

---

## Comparison with CANet

CANet (Hanselmann et al., 2020) is the architecture SynCAN was designed to benchmark and achieves approximately 0.95 average F1 on the single-target attack types. **We believe that the two models are architecturally complementary rather than directly competing**.

CANet uses **per-ID LSTM subnetworks** that update only when each signal's message arrives, allowing it to naturally detect single-target attacks under CAN's variable-frequency broadcast structure. **TranAD operates across all 20 signals simultaneously**, making it well-suited to coordinated attacks where the anomaly manifests as correlation breakdown across ECUs — **a threat class the standard single-target SynCAN benchmark was not designed to evaluate**.

The results reflect this division. **On single-target attacks, CANet's per-ID architecture has a structural advantage**: a suppressed ECU goes silent, which a per-ID model detects from the absence of messages, while TranAD's forward-filled preprocessing partially obscures the same signature. On coordinated attacks, **TranAD's cross-signal reconstruction detects correlation breakdown that per-ID models cannot reliably identify**. Both models struggle on suppress and flooding attacks, which CANet's authors note are most efficiently handled by rule-based frequency monitoring.

We theorize that a **hybrid architecture combining all 3 approaches** — per-ID LSTM for localized single-ECU attacks, TranAD for correlated multi-ECU attacks, and frequency monitoring for suppress and flooding — covers the full attack surface more completely than either model alone.

---

## Limitations

- **Single-target sweep.** The hyperparameter sweep optimizes for the five standard SynCAN attack types as a real-world baseline, all of which are single-target. However, there is no guarantee that the configuration best suited for single-target detection is optimal for coordinated attack detection, or for interval detection. A sweep that includes coordinated attack F1 and/or interval detection as an evaluation criterion is a natural next step.
- **Synthetic coordinated attacks.** The coordinated attack evaluation uses injected attacks rather than recorded real-world multi-ECU compromises. The attack design reflects known CAN correlation structure, but real coordinated attacks may have different signatures.
- **Forward-fill preprocessing.** The synchronized matrix representation loses the inter-arrival timing that CANet uses. Frequency-based attacks (suppress, flooding) are more naturally addressed by monitoring message arrival rates, which TranAD does not do.
- **Synthetic dataset.** SynCAN's authors note it is "somewhat cleaner than in the real case." Performance on real vehicle CAN traffic is expected to be lower.

---

## Future work

The natural next step is applying the [Striim Labs LSTM autoencoder prototype](https://github.com/striim-labs/lstm-autoencoder-spark-kafka) to SynCAN in a CANet-style per-ID configuration — one LSTM per CAN ID, updating only when that ID broadcasts, with a shared latent vector for cross-ID reconstruction. This addresses the variable-frequency preprocessing limitation directly and is the architecture best matched to single-target per-ECU attacks.

Given TranAD's strength on coordinated attacks and a per-ID LSTM's strength on localized attacks, a hybrid architecture covers the full attack surface: per-ID LSTM for individual signal anomalies, TranAD for correlated multi-signal deviations, and rule-based frequency monitoring for suppress and flooding.

---

## Additional methodology

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

The quick sweep (16 targeted configurations, `--quick`) varies `window_size`, `lr`, `n_layers`, `n_heads`, and `d_feedforward` with `loss_weighting` and `scoring_mode` fixed at `exponential_decay` and `averaged` respectively, as supported by previous grid sweep tests on SynCAN. Key findings:

- `n_layers=2` consistently outperforms both 1 and 3. Three layers underperform on SynCAN, suggesting the dataset's correlation structure doesn't benefit from added depth.
- `n_heads=5` outperforms `n_heads=10` at `window_size=140`. With 20 channels and 10 heads, each head attends to 2 channels on average; reducing to 5 gives each head richer multi-channel context.
- `d_feedforward=32` collapses at `window_size=140` (F1=0.42). Larger feedforward capacity at longer windows overfits the 10% subsample.
- `window_size=140` marginally outperforms `window_size=100`; `window_size=60` underperforms unless paired with a larger network.

Winning configuration: `window_size=140`, `n_layers=2`, `n_heads=5`, `d_feedforward=8`, `lr=0.001`.

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
uv run python syncan/5_coordinated_attack.py
uv run python syncan/5_coordinated_attack.py --two-tailed
```

Computes the cross-ID Pearson correlation matrix, selects the most correlated signal groups, injects 100 coordinated attack intervals per scenario (50–500 timesteps each) into the normal test file, and evaluates the best model against each. The `--two-tailed` flag enables an additional z-score attribution analysis.
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