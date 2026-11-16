# C-MAPSS RUL Prediction using TCN

## 1. Project Overview

This project develops a Remaining Useful Life (RUL) prediction system for turbofan engine degradation using a Temporal Convolutional Network (TCN). It is designed for a final-year industrial IoT (IIoT) and predictive maintenance project based on the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark.

The main goal is to estimate how many cycles remain before an engine fails, using multivariate sensor data from the engine health monitoring system. Accurate RUL prediction enables condition-based maintenance, reduces unexpected downtime, lowers maintenance costs, and supports IIoT-focused decision-making systems.

This project is built around the following key idea:

- Use raw engine telemetry and sensor signals
- Convert each engine trajectory into sliding-window time sequences
- Train a deep learning model that learns degradation patterns over time
- Predict RUL as a regression output
- Evaluate performance using RMSE and PHM08 scoring
- Integrate the prediction into an IIoT priority scheduler and web dashboard

---

## 2. Problem Statement

In industrial systems, engine failure can lead to severe operational and financial losses. Early prediction of remaining useful life is essential for planning maintenance actions before a critical failure occurs.

The C-MAPSS dataset simulates degradation in turbofan engines under different operating conditions and fault modes. Each engine unit is monitored over time, and the task is to estimate RUL at any given operating cycle based on the engine's sensor history.

This project solves that problem by:

- importing the C-MAPSS data files
- converting raw operational data into supervised time-series samples
- training a TCN to model temporal degradation patterns
- optimizing for robust maintenance-focused performance

---

## 3. Why This Project Matters

This project is relevant to IIoT and smart manufacturing because it connects artificial intelligence with real industrial maintenance decisions.

Benefits include:

- reduced sudden machine failure
- improved maintenance scheduling
- lower spare-part and downtime costs
- better decision support for maintenance engineering
- scalable predictive analytics for industrial assets

The final project is not just a model demo; it is a practical pipeline from raw industrial data to predictive maintenance decisions.

---

## 4. Dataset: NASA C-MAPSS

The project uses the NASA C-MAPSS turbofan engine dataset, which contains multiple subsets:

- FD001: one operating condition, one fault mode
- FD002: six operating conditions, one fault mode
- FD003: one operating condition, two fault modes
- FD004: six operating conditions, two fault modes

Each dataset contains:

- train_FD00x.txt
- test_FD00x.txt
- RUL_FD00x.txt

### Data structure

Each record contains engine-level sensor and operational parameters such as:

- unit ID
- cycle number
- operating settings (setting1, setting2, setting3)
- sensor readings (s1 to s21)

The project loads data using a standardized column schema:

- unit
- cycle
- setting1
- setting2
- setting3
- s1 ... s21

---

## 5. Core Concept: RUL Labeling

For training, the model uses piecewise-linear RUL labels. This is the standard C-MAPSS approach.

For each engine unit:

- find the maximum cycle number
- compute residual life at each cycle as max_cycle - current_cycle
- cap the label at 125 cycles

This is implemented as:

```python
RUL = min(max_cycle - cycle, 125)
```

This prevents very early-life samples from being overly penalized while still preserving degradation information.

---

## 6. Sequence Modeling and Windowing

The system transforms each engine's history into sliding-window sequences.

### Windowing idea

- each engine has many time steps
- a fixed-length window of the last N cycles is treated as one sample
- the label is the RUL at the end of that window

This allows the model to see time dependency and degradation trends instead of only isolated points.

### Example

If the window size is 30:

- each sample contains 30 consecutive cycles
- the model learns from the previous 30-cycle physical behavior
- the final cycle within that window provides the target RUL

The project defines window length as a configurable training parameter.

---

## 7. Model Architecture: Temporal Convolutional Network

The predictive model is a Temporal Convolutional Network (TCN), which is well suited for time-series forecasting.

### TCN design features

- causal convolutions
- dilated convolutions
- residual connections
- ReLU activations
- dropout regularization
- final regression head for continuous RUL output

### Architectural modules

The model is implemented in `src/tcn_model.py` and includes:

- `Chomp1d`: trims the convolution output to maintain causal alignment
- `TemporalBlock`: a residual TCN block using dilated convolution
- `TCN`: end-to-end sequence model with final regression layer

The model converts the input sequence from shape:

```python
(batch, time_steps, features)
```

into a 1D RUL prediction per sample.

---

## 8. Data Preprocessing Pipeline

The project includes a dedicated preprocessing and utility pipeline in `src/data_utils.py`.

### Key operations

- load raw text files into DataFrames
- normalize column names to the canonical C-MAPSS schema
- create training RUL labels
- drop low-information sensors with near-zero variance
- fit a min-max scaler on training data only
- create training and validation sequences by engine unit
- create test sequences using the last window per engine

### Normalization

A custom `Normalizer` class is used for feature scaling:

- fit only on training data
- avoid information leakage
- preserve consistent scaling across train and test sets

---

## 9. Training Pipeline

The main training logic is implemented in `src/train.py`.

### Train workflow

The script does the following:

1. loads the selected dataset (FD001/FD002/FD003/FD004)
2. parses the raw train/test files
3. creates RUL labels
4. removes non-informative sensor columns
5. splits train and validation sets by engine unit
6. fits the scaler on training data
7. creates sliding-window sequences
8. initializes the TCN model
9. trains with Adam optimizer and MSE loss
10. monitors validation RMSE
11. saves the best model checkpoint
12. saves predictions, metrics, and training curve

### Example training command

```bash
python src/train.py --dataset FD001 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
```

The same pipeline supports all four data subsets:

```bash
python src/train.py --dataset FD001 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
python src/train.py --dataset FD002 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
python src/train.py --dataset FD003 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
python src/train.py --dataset FD004 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
```

---

## 10. Evaluation Metrics

Two evaluation measures are used:

### RMSE

Root Mean Squared Error measures overall prediction error magnitude.

```python
RMSE = sqrt(mean((y_true - y_pred)^2))
```

### PHM08 score

The PHM08 score is a maintenance-oriented asymmetric metric. It penalizes late predictions more heavily than early predictions, which is realistic in maintenance applications.

This is important because:

- overestimating remaining life (predicting too much life left) is dangerous
- underestimating remaining life is less harmful than late failure detection

The project implements this using the asymmetric scoring formula:

```python
if d < 0:
    score = exp(-d/10) - 1
else:
    score = exp(d/13) - 1
```

where d = true_RUL - pred_RUL.

---

## 11. IIoT Priority Scheduler

The project includes an IIoT-oriented prioritization module in `src/priority.py`.

This module converts predicted RUL into an urgency score, allowing higher-priority transmission or maintenance attention for assets with low remaining life.

The general idea is:

- lower predicted RUL = higher urgency
- greater prediction risk = quicker maintenance action

This adds a real-world industrial decision layer to the model beyond pure regression output.

---

## 12. Web Application

A Flask-based web application is provided in `webapp/app.py`.

### Features

- upload a raw C-MAPSS file
- select dataset type (FD001–FD004)
- configure bandwidth / urgency parameters
- run inference with the trained model
- display results in a web dashboard

### Run the app

```bash
python webapp/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

This gives a lightweight demo interface suitable for presentation and evaluation.

---

## 13. Project Structure

```text
cmapss-rul-tcn/
├── data/
│   ├── FD001/
│   ├── FD002/
│   ├── FD003/
│   └── FD004/
├── src/
│   ├── data_utils.py
│   ├── tcn_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── priority.py
├── webapp/
│   ├── app.py
│   ├── inference.py
│   ├── templates/
│   │   ├── index.html
│   │   └── results.html
│   └── static/
│       ├── style.css
│       └── uploads/
├── outputs/
├── notebooks/
│   └── eda.ipynb
├── tests/
│   └── test_data_utils.py
├── requirements.txt
├── README.md
├── PROJECT_OVERVIEW.md
├── .gitignore
└── .venv/
```

---

## 14. Key Technologies and Dependencies

The project uses Python and PyTorch with supporting data-science libraries.

### Main libraries

- Python 3.x
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Flask
- Pytest

These libraries are used for:

- model development
- preprocessing and feature scaling
- metric computation
- plotting learning curves
- web deployment
- test validation

---

## 15. Main Files and Their Roles

### `src/data_utils.py`

Handles:

- raw CSV loading
- C-MAPSS column setup
- RUL augmentation
- sensor filtering
- scaling and normalization
- sequence creation
- RMSE and PHM08 calculation

### `src/tcn_model.py`

Contains the TCN architecture:

- 1D causal temporal blocks
- dilated convolutions
- residual blocks
- final regression head

### `src/train.py`

This is the training pipeline:

- dataset selection
- preprocessing
- validation split
- model training
- checkpoint saving
- metric export
- training curves

### `src/evaluate.py`

Loads the saved model and evaluates a checkpoint without retraining.

### `webapp/app.py`

Runs the Flask interface for user interaction and inference.

### `tests/test_data_utils.py`

Basic validation tests for preprocessing and utility functions.

---

## 16. Working Methodology

The pipeline follows a standard machine-learning workflow:

1. Data ingestion
2. preprocessing and feature engineering
3. label generation
4. feature scaling
5. window-based sequence conversion
6. TCN training
7. validation monitoring
8. checkpointing
9. metric evaluation
10. deployment in a web-friendly interface

This structure reflects a realistic end-to-end predictive maintenance solution.

---

## 17. Output Artifacts

The training stage generates outputs such as:

- model checkpoint
- predictions CSV
- metric file
- learning curve plot
- dataset-specific output folders

Example output structure:

```text
outputs/
└── FD001/
    ├── tcn_FD001.pt
    ├── predictions_FD001.csv
    ├── metrics_FD001.txt
    └── training_curve_FD001.png
```

---

## 18. Project Status and Validation

The project is in a working and validated state.

### Validated status

The latest project validation command completed successfully:

```bash
python -m pytest -q
```

Result:

```text
5 passed in 1.89s
```

This confirms the utility and core workflow are stable under the existing test suite.

The project also supports all four major C-MAPSS subsets (FD001–FD004) and is aligned with the intended final-year project scope.

---

## 19. Final Project Scope

This project is designed to represent a complete predictive maintenance system using deep learning for industrial machinery health monitoring.

The final system combines:

- time-series deep learning for RUL prediction
- industrial maintenance-oriented evaluation metrics
- operational condition analysis across multiple datasets
- IIoT priority scheduling logic
- a small interactive web dashboard for demonstration

This makes the project suitable for a final-year major project, technical report, presentation, or demo environment.

---

## 20. Conclusion

The C-MAPSS RUL TCN project demonstrates how deep learning can be used for predictive maintenance in industrial settings. By combining TCN modeling, sliding-window time-series conversion, asymmetric maintenance scoring, and an IIoT scheduling layer, the project delivers a realistic, end-to-end solution for remaining useful life prediction.

It is well structured for academic demonstration, experimentation, and extension to more advanced industrial prognostics scenarios.
