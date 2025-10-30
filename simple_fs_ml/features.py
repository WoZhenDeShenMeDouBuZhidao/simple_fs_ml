
from __future__ import annotations
from typing import List, Dict, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def drop_excluded(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    to_drop = [c for c in exclude_cols if c in df.columns]
    return df.drop(columns=to_drop, errors="ignore")

def select_random(X: pd.DataFrame, y: pd.Series, task: str, ratio: float, random_state: int = 42) -> List[str]:
    rng = np.random.default_rng(random_state)
    candidates = list(X.columns)
    k = max(1, int(round(len(candidates) * ratio)))
    rng.shuffle(candidates)
    return candidates[:k]

def _safe_corr(x: pd.Series, y: np.ndarray) -> float:
    x = pd.to_numeric(x, errors="coerce")
    mask = x.notna() & ~pd.isna(y)
    if mask.sum() < 2 or x[mask].nunique() <= 1:
        return 0.0
    try:
        return float(np.abs(np.corrcoef(x[mask].values, y[mask].values)[0,1]))
    except Exception:
        return 0.0

def select_by_corr(X: pd.DataFrame, y: pd.Series, task: str, ratio: float, random_state: int = 42) -> List[str]:
    y_arr = y.values
    if task == "classification":
        le = LabelEncoder()
        y_arr = le.fit_transform(y.astype(str).values)

    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        return select_random(X, y, task, ratio, random_state=random_state)

    scores = {col: _safe_corr(X[col], y_arr) for col in numeric_cols}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    k = max(1, int(round(len(numeric_cols) * ratio)))
    selected = [col for col, s in ranked[:k]]
    return selected

FEATURE_SELECTORS: Dict[str, Callable[[pd.DataFrame, pd.Series, str, float, int], List[str]]] = {
    "random": select_random,
    "corr": select_by_corr,
}

def prepare_design_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    selected_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df[selected_features].copy()
    va = valid_df[selected_features].copy()

    cat_cols = list(tr.select_dtypes(include=["object", "category", "bool"]).columns)
    for c in cat_cols:
        if tr[c].dtype == "bool":
            tr[c] = tr[c].astype("object")
            va[c] = va[c].astype("object")

    num_cols = [c for c in tr.columns if c not in cat_cols]
    if num_cols:
        tr[num_cols] = tr[num_cols].apply(lambda s: s.fillna(s.median()), axis=0)
        va[num_cols] = va[num_cols].apply(lambda s: s.fillna(tr[s.name].median()), axis=0)

    for c in cat_cols:
        tr[c] = tr[c].astype("object").fillna("NA")
        va[c] = va[c].astype("object").fillna("NA")

    tr["_split_flag"] = 1
    va["_split_flag"] = 0
    comb = pd.concat([tr, va], axis=0, ignore_index=True)
    comb_oh = pd.get_dummies(comb, columns=cat_cols, dummy_na=False)

    tr_oh = comb_oh[comb_oh["_split_flag"] == 1].drop(columns=["_split_flag"])
    va_oh = comb_oh[comb_oh["_split_flag"] == 0].drop(columns=["_split_flag"])
    return tr_oh, va_oh
