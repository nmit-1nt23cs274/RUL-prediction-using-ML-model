# C-MAPSS Turbofan Remaining Useful Life Prediction

## Project Summary

This project builds an end-to-end predictive-maintenance prototype for estimating the Remaining Useful Life (RUL) of turbofan engines from time-series telemetry. It uses a Temporal Convolutional Network (TCN) trained on the NASA C-MAPSS benchmark, evaluates predictions with RMSE and the asymmetric PHM08 score, ranks engines by urgency, and provides a Flask dashboard for inference and bandwidth-aware transmission scheduling.

The repository contains data-processing and training code, trained checkpoints and evaluation artifacts for all four C-MAPSS subsets, a small test suite, an exploratory notebook, and a web application.

## Objectives

- Estimate engine RUL in cycles from its recent operating and sensor history.
- Compare model performance across four subsets with different operating conditions and fault modes.
- Convert RUL predictions into maintenance/transmission priorities.
- Provide a browser-based demonstration for uploading telemetry and viewing fleet results.

## Dataset

The project uses the four NASA C-MAPSS turbofan-engine benchmark subsets present under `data/`:

| Subset | Operating conditions | Fault modes |
| --- | --- | --- |
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

Each subset directory contains a training file (`train_FD00x.txt`), a test file (`test_FD00x.txt`), and test ground-truth RUL values (`RUL_FD00x.txt`). Each telemetry row has 26 whitespace-separated numeric columns: engine/unit ID, cycle, three operating settings, and 21 sensor readings (`s1` through `s21`).

The training files contain complete run-to-failure trajectories. The test files contain partial trajectories; the separate RUL file provides the target life for the last observed cycle of each test engine.

## Methodology

### Preprocessing and labels

Implemented in `src/data_utils.py` and used by `src/train.py`:

1. Read whitespace-separated rows into a consistent named-column table.
2. Generate training labels per engine using capped, piecewise-linear RUL:
   `RUL = min(max_cycle_for_engine - current_cycle, 125)`.
3. Remove sensors whose standard deviation is at or below `1e-6` in the training data.
4. Add rolling 10-cycle mean and standard-deviation features for active sensors, grouped by engine. Rolling windows use `min_periods=1`; an undefined initial standard deviation is filled with zero.
5. Split training engines into training and validation groups, rather than randomly splitting individual windows. The default validation fraction is 15% and the split seed is 42.
6. Fit a custom min-max normalizer on training rows only, then use it to scale training, validation, and test features.
7. Convert each engine trajectory into sliding sequences. The target is the RUL at the final cycle in each sequence. Short sequences are padded at the front by repeating their first row.
8. Create one test sequence per test engine from its most recent cycles, then compare predictions with the RUL values in the subset's ground-truth file.

### TCN model

Implemented in `src/tcn_model.py` using PyTorch:

- Temporal 1D convolutions with causal alignment (right-side outputs are trimmed by `Chomp1d`).
- Dilations grow exponentially by block (`1, 2, 4, 8` with the default four blocks).
- Residual connections, weight normalization, ReLU activations, and dropout.
- A regression head maps the final time-step representation through a 32-unit hidden layer to one continuous RUL estimate.
- The input tensor has shape `(batch, time_steps, features)`; the model transposes it for the convolution stack and returns one estimate per sequence.

Default architecture settings are channels `[32, 32, 64, 64]`, kernel size `3`, and dropout `0.2`.

### Training

Implemented in `src/train.py`:

- Loss: mean squared error (MSE).
- Optimizer: Adam, with default learning rate `0.001` and weight decay `0.0001`.
- Gradient norm clipping: maximum norm `5`.
- Learning-rate scheduler: `ReduceLROnPlateau`, factor `0.5`, with patience derived from the early-stopping patience.
- Best checkpoint is selected using validation RMSE; early stopping uses a default patience of 12 epochs.
- Default maximum is 80 epochs, batch size 256, window length 30, and random seed 42. A run can stop before the maximum epoch count due to early stopping.
- CUDA is used when available; otherwise training runs on CPU.

The checkpoint stores model weights, feature names, the fitted normalizer, and the training arguments.

### Metrics

- **RMSE**: square root of the mean squared difference between true and predicted RUL. Lower is better.
- **PHM08 score**: sum of asymmetric per-engine penalties, with late predictions (predicted RUL greater than true RUL) penalized more heavily than early predictions. Lower is better.

The implemented convention is `d = true_RUL - predicted_RUL`; for `d < 0`, the penalty is `exp(-d/10) - 1`, otherwise it is `exp(d/13) - 1`.

## Current Saved Results

The following values are read from the metric files currently in `outputs/`. These are test-set results from the saved runs; the recorded epoch count is the number of epochs actually completed, which can be below the 80-epoch configured maximum.

| Dataset | RMSE (cycles) | PHM08 score | Window | Epochs completed |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 20.862268 | 993.231934 | 30 | 37 |
| FD002 | 34.289120 | 32092.060547 | 30 | 35 |
| FD003 | 23.545105 | 1548.350586 | 30 | 21 |
| FD004 | 36.327042 | 25302.433594 | 30 | 32 |

These are recorded experiment results, not a guarantee of identical values on every machine or retraining run. Comparisons are meaningful when using the same data split, preprocessing, target convention, and metric implementation.

## IIoT Priority Scheduling

`src/priority.py` converts predicted RUL to an urgency score:

`priority_score = 1 / (predicted_RUL + 0.001)`.

Engines with lower predicted RUL receive higher scores and are sorted first. The core priority module labels engines as critical (`RUL < 20`), warning (`20 <= RUL < 60`), or normal (`RUL >= 60`). The web inference layer displays four tiers: critical (`RUL < 20`), high (`20 <= RUL < 50`), medium (`50 <= RUL < 90`), and low (`RUL >= 90`). It marks the top N ranked engines for transmission according to the user-provided bandwidth limit and defers the others.

## Web Application

The Flask application in `webapp/app.py` provides:

- Upload and basic format validation of a C-MAPSS-style telemetry file (26 numeric, whitespace-separated columns).
- Selection of FD001, FD002, FD003, or FD004 and a transmission bandwidth limit.
- TCN inference when a matching model checkpoint is available.
- A dashboard with fleet RUL summaries, urgency-tier counts, charts, and a ranked transmission queue.
- CSV export of the displayed queue.

The interface is rendered with Flask/Jinja templates in `webapp/templates/`, styled by `webapp/static/style.css`, and uses Chart.js from a CDN for charts.

**Important:** if the selected dataset checkpoint is not found or cannot be loaded, the inference module generates deterministic simulated preview values. The dashboard identifies this as demo mode; those values are not model predictions and must not be used as experimental results.

## Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.x |
| Deep learning | PyTorch |
| Data processing | NumPy, Pandas |
| ML utilities | scikit-learn (engine-level train/validation split) |
| Plotting | Matplotlib; Seaborn is included in dependencies |
| Web backend | Flask |
| Templates | Jinja2 via Flask |
| Dashboard charts | Chart.js 4.4.4 via CDN |
| Tests | pytest |
| Other declared dependency | joblib |

Dependency minimums are listed in `requirements.txt`: PyTorch 2.2, NumPy 1.26, Pandas 2.0, scikit-learn 1.3, Matplotlib 3.8, Seaborn 0.13, joblib 1.3, Flask 3.0, and pytest 8.0. Python itself is not pinned to a specific minor version in the repository.

## Repository Layout

```text
cmapss-rul-tcn/
├── data/                       # FD001-FD004 train, test, and RUL files
├── notebooks/eda.ipynb         # Exploratory data analysis notebook
├── outputs/                    # Per-subset checkpoints, predictions, metrics, plots
├── src/
│   ├── data_utils.py            # Loading, labels, scaling, windows, metrics
│   ├── tcn_model.py             # TCN architecture
│   ├── train.py                 # Training and artifact generation
│   ├── evaluate.py              # Checkpoint evaluation and plot generation
│   └── priority.py              # RUL urgency scoring and ranking
├── tests/test_data_utils.py     # Data and priority utility tests
├── webapp/
│   ├── app.py                   # Flask routes and upload handling
│   ├── inference.py             # Prediction, fallback, ranking, dashboard data
│   ├── templates/               # Upload and results pages
│   └── static/                  # CSS and uploaded telemetry files
├── requirements.txt             # Python dependencies
├── README.md                    # Quick-start documentation
├── PROJECT_OVERVIEW.md          # Existing project overview
└── PROJECT_REPORT.md            # Consolidated project report
```

Each `outputs/FD00x/` folder currently contains a `.pt` model checkpoint, a predictions CSV, a metrics text file, and a training-curve PNG. FD001 also contains `confusion_matrix_styled.png`; RUL prediction is a regression task, so this extra plot should not be interpreted as a primary regression metric.

## Running the Project

Run commands from the repository root. Create and activate a virtual environment as appropriate for the local Python installation.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Train a subset

```bash
python src/train.py --dataset FD001 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
```

Replace `FD001` with `FD002`, `FD003`, or `FD004` to train another subset. Training writes the checkpoint, predictions, metrics, and training curve under `outputs/<DATASET>/`.

### Evaluate a saved checkpoint

```bash
python src/evaluate.py --checkpoint outputs/FD001/tcn_FD001.pt --dataset FD001 --data_dir data --window 30 --output outputs
```

### Run tests

```bash
python -m pytest -q
```

The repository's existing overview records a previous run with five passing tests. That is historical documentation; run the command above to verify the current checkout.

### Launch the dashboard

```bash
python webapp/app.py
```

Open `http://127.0.0.1:5000` in a browser and upload a test-format file that matches the selected dataset/checkpoint.

## Tests and Validation Scope

`tests/test_data_utils.py` contains five tests covering capped RUL labels, sequence construction, front-padding of short histories, asymmetric PHM08 behavior, and priority ranking. They exercise selected utilities, not the whole training-to-dashboard workflow.

## Files and Outputs

- `outputs/FD00x/tcn_FD00x.pt`: model checkpoint and preprocessing metadata.
- `outputs/FD00x/predictions_FD00x.csv`: engine ID, true RUL, predicted RUL, and prediction error.
- `outputs/FD00x/metrics_FD00x.txt`: dataset name, RMSE, PHM08 score, window, and completed epoch count.
- `outputs/FD00x/training_curve_FD00x.png`: training loss and validation RMSE across epochs.
- `outputs/FD00x/eval_FD00x.png`: evaluation visualization generated by `src/evaluate.py` when run.

## Limitations and Interpretation

- C-MAPSS is a simulated benchmark and does not by itself establish performance on real operational aircraft engines.
- RUL is estimated from benchmark sensor histories; the output is decision support, not a certified maintenance instruction.
- The provided split is at engine level for validation, while final metrics use the benchmark test set and its supplied RUL labels.
- PHM08 values are summed over test engines, so scores depend on subset size as well as prediction quality; compare them alongside RMSE and only under a consistent evaluation setup.
- Dashboard demo fallback output is synthetic and must be clearly distinguished from trained-model inference.
- The current tests focus on utilities; they do not comprehensively test training, checkpoint compatibility, upload security, or dashboard inference across all four subsets.

## Main Source Files

- `src/data_utils.py`: C-MAPSS parsing, target creation, low-information sensor removal, rolling features, normalization, windowing, RMSE, and PHM08 score.
- `src/tcn_model.py`: causal dilated TCN with residual temporal blocks and a regression head.
- `src/train.py`: preprocessing, training/validation, early stopping, checkpointing, test predictions, metrics, and training plots.
- `src/evaluate.py`: loading a checkpoint, predicting test-engine RUL, printing metrics, and saving an evaluation plot.
- `src/priority.py`: urgency score, three-level priority classification, and ranking.
- `webapp/app.py` and `webapp/inference.py`: upload interface and end-to-end dashboard inference/scheduling.
