from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import numpy as np

CRITICAL_RUL = 20
WARNING_RUL = 60
EPSILON = 1e-3


def priority_score(rul: float, epsilon: float = EPSILON) -> float:
    return 1.0 / (float(rul) + epsilon)


def classify_priority(rul: float, critical: float = CRITICAL_RUL, warning: float = WARNING_RUL) -> str:
    if rul < critical:
        return "critical"
    if rul < warning:
        return "warning"
    return "normal"


def rank_engines(rul_dict: Union[dict, Iterable[float], np.ndarray]) -> List[Tuple[object, float, float, str]]:
    if isinstance(rul_dict, dict):
        items = list(rul_dict.items())
    elif isinstance(rul_dict, np.ndarray):
        items = list(enumerate(rul_dict.tolist()))
    else:
        items = list(enumerate(list(rul_dict)))

    ranked = []
    for engine_id, rul in items:
        value = float(rul)
        scored = priority_score(value)
        bucket = classify_priority(value)
        ranked.append((engine_id, value, scored, bucket))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked
