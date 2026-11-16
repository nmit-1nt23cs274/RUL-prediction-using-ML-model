import numpy as np
import pandas as pd

from src.data_utils import add_train_rul, drop_low_info_sensors, make_sequences, phm08_score


def test_rul_capping_logic():
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2],
        "cycle": [1, 10, 2, 8],
        "setting1": [0, 0, 0, 0],
        "setting2": [0, 0, 0, 0],
        "setting3": [0, 0, 0, 0],
        "s1": [1, 1, 1, 1],
        "s2": [2, 2, 2, 2],
    })
    out = add_train_rul(df)
    expected = np.array([9, 0, 6, 0], dtype=np.float32)
    assert np.allclose(out["RUL"].values, expected)


def test_make_sequences_shape():
    df = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "setting1": [0, 0, 0, 0, 0],
        "setting2": [0, 0, 0, 0, 0],
        "setting3": [0, 0, 0, 0, 0],
        "s1": [1, 2, 3, 4, 5],
        "s2": [10, 20, 30, 40, 50],
        "RUL": [5, 4, 3, 2, 1],
    })
    X, y = make_sequences(df, ["s1", "s2"], window=3)
    assert X.shape[1:] == (3, 2)
    assert y.shape[0] == X.shape[0]


def test_short_sequence_front_padding():
    df = pd.DataFrame({
        "unit": [1, 1],
        "cycle": [1, 2],
        "setting1": [0, 0],
        "setting2": [0, 0],
        "setting3": [0, 0],
        "s1": [2, 3],
        "s2": [20, 30],
        "RUL": [10, 9],
    })
    X, y = make_sequences(df, ["s1", "s2"], window=3)
    assert X.shape == (1, 3, 2)
    assert np.allclose(X[0, 0], np.array([2, 20], dtype=np.float32))
    assert y.shape == (1,)


def test_phm08_late_worse_than_early():
    y_true = np.array([50.0])
    early_pred = np.array([40.0])
    late_pred = np.array([60.0])
    assert phm08_score(y_true, late_pred) > phm08_score(y_true, early_pred)


def test_priority_ranking_order():
    from src.priority import classify_priority, priority_score, rank_engines

    rul_dict = {1: 15.0, 2: 120.0, 3: 45.0}
    ranked = rank_engines(rul_dict)
    
    # Engine 1 (RUL 15) must have highest priority and be ranked #1
    assert ranked[0][0] == 1
    assert ranked[0][3] == "critical"

    # Engine 3 (RUL 45) must be ranked #2 and warning
    assert ranked[1][0] == 3
    assert ranked[1][3] == "warning"

    # Engine 2 (RUL 120) must be ranked #3 and normal
    assert ranked[2][0] == 2
    assert ranked[2][3] == "normal"

