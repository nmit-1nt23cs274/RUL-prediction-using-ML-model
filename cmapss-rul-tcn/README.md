# C-MAPSS RUL Prediction using TCN

This project implements a Temporal Convolutional Network (TCN) for predicting Remaining Useful Life (RUL) from the NASA C-MAPSS turbofan engine degradation dataset.

The project is designed for a final-year IIoT major project and supports all four C-MAPSS subsets: FD001, FD002, FD003, and FD004. FD001 is the baseline case, while the full suite allows comparison across different operating conditions and fault modes.

## Dataset

The NASA C-MAPSS benchmark contains four subsets:

- FD001: one operating condition, one fault mode
- FD002: six operating conditions, one fault mode
- FD003: one operating condition, two fault modes
- FD004: six operating conditions, two fault modes

Each dataset includes:

- train_FD00x.txt
- test_FD00x.txt
- RUL_FD00x.txt

The raw files should be placed in the corresponding folder under `data/` before running training.

## Project structure

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
├── .gitignore
└── .
```

## RUL labeling

Training labels are created using the standard piecewise-linear degradation target:

- For each engine unit, compute the max cycle value.
- For each row at cycle `t`, set RUL = min(max_cycle - t, 125)
- Cap at 125 to avoid over-penalizing early-life behavior.

This is the standard C-MAPSS procedure used in PHM benchmark papers.

## Windowing

The model uses sliding windows of fixed length `window` over each engine trajectory:

- Each sample is a window of the last `window` cycles.
- The label is the RUL at the final cycle in that window.
- For short trajectories, the sequence is front-padded to maintain the required window length.

## Scoring

The PHM08 score is asymmetric:

- Early predictions are penalized less severely.
- Late predictions are penalized more severely.

This is implemented in `src/data_utils.py` with the standard formulation:

```python
if d < 0:
    score = exp(-d/10) - 1
else:
    score = exp(d/13) - 1
```

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train on any dataset using the same pipeline:

```bash
python src/train.py --dataset FD001 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
python src/train.py --dataset FD002 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
python src/train.py --dataset FD003 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
python src/train.py --dataset FD004 --data_dir data --out_dir outputs --window 30 --epochs 80 --batch_size 256 --lr 1e-3 --val_split 0.15 --seed 42
```

## Evaluation

```bash
python src/evaluate.py --checkpoint outputs/FD001/tcn_FD001.pt --dataset FD001 --data_dir data --window 30 --output outputs
```

## IIoT priority scheduler

The `priority.py` module converts predicted RUL into a transmission-priority score.

```python
priority_score = 1 / (predicted_rul + 1e-3)
```

This allows the system to rank engines by urgency, with lower RUL values receiving higher priority for data transmission.

## Outputs

The training pipeline saves:

- `outputs/FD001/tcn_FD001.pt`
- `outputs/FD001/predictions_FD001.csv`
- `outputs/FD001/metrics_FD001.txt`
- `outputs/FD001/training_curve_FD001.png`

## Web application

A lightweight Flask app is included at `webapp/app.py` for inference and result display.

Run it with:

```bash
python webapp/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

