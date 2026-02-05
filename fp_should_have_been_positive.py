#!/usr/bin/env python3
"""
fp_should_have_been_positive.py

Goal:
- You have a CSV of model mistakes containing only FP and FN rows.
- Train an ML model to learn what "true positives" look like (your FN rows are true label=1)
  vs what "true negatives" look like (your FP rows are true label=0).
- Then, among the FP rows, flag cases that the ML model thinks are actually label=1
  (i.e., "this FP likely should have been a positive").

Input expected (your file has these):
- error_type in {FP, FN}
- label (same as y_true)
- numeric feature columns (RSI/drop/hits/etc)
- file_date, symbol
- optionally p, y_pred (from your original model)

Outputs:
- suspect_fp_should_be_positive.csv (ranked)
- model_feature_importance.csv (global)
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Config:
    infile: str
    out_suspects: str
    out_importance: str
    threshold: float
    folds: int
    top_n: int
    include_base_p: bool


DROP_ALWAYS = {
    # identifiers / non-features
    "label", "y_true", "error_type",
    # original model outputs (excluded by default)
    "y_pred",
}
BASE_P_COL = "p"


def _parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="debug_mistakes_main.csv", help="Path to debug_mistakes_main.csv")
    ap.add_argument("--out-suspects", default="suspect_fp_should_be_positive.csv", help="Output CSV of flagged FPs")
    ap.add_argument("--out-importance", default="model_feature_importance.csv", help="Output CSV of feature importance")
    ap.add_argument("--threshold", type=float, default=0.80, help="Flag FP if OOF P(label=1) >= threshold")
    ap.add_argument("--folds", type=int, default=5, help="Stratified CV folds")
    ap.add_argument("--top-n", type=int, default=120, help="Max number of suspect FPs to write (ranked)")
    ap.add_argument("--include-base-p", action="store_true", help="Include your original model's p as a feature")
    args = ap.parse_args()

    return Config(
        infile=args.infile,
        out_suspects=args.out_suspects,
        out_importance=args.out_importance,
        threshold=args.threshold,
        folds=args.folds,
        top_n=args.top_n,
        include_base_p=args.include_base_p,
    )


def _prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Basic checks
    needed = {"error_type", "label", "symbol", "file_date"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # Keep only FP/FN (your file appears to already be only those)
    df = df[df["error_type"].isin(["FP", "FN"])].copy()

    # Ensure label is int
    df["label"] = df["label"].astype(int)

    # Parse date -> numeric feature (ordinal)
    df["file_date_dt"] = pd.to_datetime(df["file_date"], errors="coerce")
    df["file_date_ordinal"] = df["file_date_dt"].map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    return df


def _build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # LogisticRegression is a solid "AI" baseline:
    # - outputs probabilities
    # - works with sparse one-hot
    # - fast + reasonably interpretable
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=5000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def _select_feature_columns(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str]]:
    drop = set(DROP_ALWAYS)
    if not cfg.include_base_p:
        drop.add(BASE_P_COL)

    # We'll use:
    # - numeric engineered columns (RSI/drop/hits/etc) + file_date_ordinal
    # - categorical: symbol
    candidate_cols = [c for c in df.columns if c not in drop]
    categorical_cols = ["symbol"]
    # file_date_ordinal is numeric; file_date_dt isn't needed after ordinal
    drop_more = {"file_date", "file_date_dt"}
    candidate_cols = [c for c in candidate_cols if c not in drop_more]

    # Numeric = all remaining non-categorical, non-target
    numeric_cols = [c for c in candidate_cols if c not in categorical_cols]

    # Keep only columns that actually exist and aren’t all-null
    numeric_cols = [c for c in numeric_cols if c in df.columns and df[c].notna().any()]

    return numeric_cols, categorical_cols


def _oof_probs(df: pd.DataFrame, pipe: Pipeline, feature_cols: List[str], y: np.ndarray, folds: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.full(len(df), np.nan, dtype=float)

    for tr_idx, va_idx in skf.split(df[feature_cols], y):
        X_tr, X_va = df.iloc[tr_idx][feature_cols], df.iloc[va_idx][feature_cols]
        y_tr = y[tr_idx]

        pipe.fit(X_tr, y_tr)
        oof[va_idx] = pipe.predict_proba(X_va)[:, 1]

    return oof


def _global_feature_importance(pipe: Pipeline, df: pd.DataFrame, feature_cols: List[str], y: np.ndarray) -> pd.DataFrame:
    """
    For logistic regression:
    - Extract feature names after preprocessing
    - Use absolute coefficient magnitude as importance (global)
    """
    pipe.fit(df[feature_cols], y)

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # Get feature names from ColumnTransformer
    num_cols = pre.transformers_[0][2]
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = pre.transformers_[1][2]

    cat_feature_names = []
    if len(cat_cols) == 1 and cat_cols[0] == "symbol":
        cat_feature_names = [f"symbol={v}" for v in ohe.categories_[0].tolist()]

    feat_names = list(num_cols) + cat_feature_names
    coefs = clf.coef_.ravel()

    if len(coefs) != len(feat_names):
        # fallback if shapes don't line up for any reason
        return pd.DataFrame({"feature": [], "importance": []})

    imp = np.abs(coefs)
    out = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
    return out


def main() -> int:
    warnings.filterwarnings("ignore")

    cfg = _parse_args()
    df = pd.read_csv(cfg.infile)
    df = _prep_dataframe(df)

    numeric_cols, categorical_cols = _select_feature_columns(df, cfg)
    feature_cols = numeric_cols + categorical_cols

    if not numeric_cols:
        raise SystemExit("No numeric feature columns found after filtering.")
    if "symbol" not in categorical_cols:
        raise SystemExit("Expected 'symbol' as categorical feature.")

    pipe = _build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    y = df["label"].values.astype(int)

    # OOF probability that label should be 1 (i.e., "positive-like")
    oof_p1 = _oof_probs(df, pipe, feature_cols, y, folds=cfg.folds)
    df["audit_p1_oof"] = oof_p1

    # Quick sanity metrics
    auc = roc_auc_score(y, oof_p1)
    print(f"OOF AUC (FN vs FP separability): {auc:.4f}")

    from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

    # ---- ADD THIS AFTER AUC PRINT ----
    pred_label = (oof_p1 >= 0.5).astype(int)

    acc = accuracy_score(y, pred_label)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred_label, average="binary")

    print("\n=== OOF Performance (audit model) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")

    # Full report (optional but useful)
    print("\nDetailed report:")
    print(classification_report(y, pred_label))


    # Among FPs, flag if they look like label=1
    fp = df[df["error_type"] == "FP"].copy()
    suspects = fp[fp["audit_p1_oof"] >= cfg.threshold].copy()
    suspects = suspects.sort_values("audit_p1_oof", ascending=False).head(cfg.top_n)

    # Helpful columns to carry through
    carry = [
        "file_date", "symbol", "error_type", "label",
        "audit_p1_oof",
    ]
    # keep original model outputs if present
    for c in ["p", "y_pred", "feat_prob_drop10"]:
        if c in df.columns:
            carry.append(c)

    # plus core numeric signals if present
    for c in [
        "last_prev_rsi", "last_cur_rsi", "last_drop", "last_drop_pct_simple",
        "hits_7d", "hits_30d", "hits_60d",
        "days_since_first", "days_since_last",
        "drop_mean", "drop_std", "drop_max", "drop_min",
        "cur_rsi_slope", "drop_is_newmax_30d",
    ]:
        if c in df.columns:
            carry.append(c)

    carry = list(dict.fromkeys(carry))  # dedupe

    suspects[carry].to_csv(cfg.out_suspects, index=False)
    print(f"Wrote suspects: {cfg.out_suspects}  (rows={len(suspects)})")

    # Global importance
    imp = _global_feature_importance(pipe, df, feature_cols, y)
    imp.to_csv(cfg.out_importance, index=False)
    print(f"Wrote feature importance: {cfg.out_importance}  (rows={len(imp)})")

    # Also print a small preview
    if len(suspects):
        print("\nTop suspect FPs (likely should have been positive):")
        print(suspects[carry].head(10).to_string(index=False))
    else:
        print("\nNo FP rows exceeded threshold. Try lowering --threshold (e.g. 0.70).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

