
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import json
import numpy as np
import pandas as pd

from simple_fs_ml import data, features, models, utils

def cmd_split(args):
    df = data.load_csv(args.dataset)
    train_df, valid_df = data.split_df(df, train_ratio=args.train_ratio, random_state=args.seed)
    out_dir = os.path.dirname(args.dataset) if args.output_dir is None else args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, args.train_name)
    valid_path = os.path.join(out_dir, args.valid_name)
    data.save_csv(train_df, train_path)
    data.save_csv(valid_df, valid_path)
    print(f"Wrote: {train_path}")
    print(f"Wrote: {valid_path}")

def cmd_train(args):
    utils.set_seed(args.seed)

    if args.train is not None and args.valid is not None:
        train_df = data.load_csv(args.train)
        valid_df = data.load_csv(args.valid)
    else:
        df = data.load_csv(args.dataset)
        train_df, valid_df = data.split_df(df, train_ratio=args.train_ratio, random_state=args.seed)

    if args.exclude:
        train_df = features.drop_excluded(train_df, args.exclude)
        valid_df = features.drop_excluded(valid_df, args.exclude)

    if args.target not in train_df.columns:
        raise ValueError(f"Target '{args.target}' not found in training data.")

    X_cols = [c for c in train_df.columns if c != args.target]
    if len(X_cols) == 0:
        raise ValueError("No features remaining after exclusion.")

    selector = features.FEATURE_SELECTORS.get(args.method)
    if selector is None:
        raise ValueError(f"Unknown selection method: {args.method}. Options: {list(features.FEATURE_SELECTORS)}")
    if not (0.0 < args.ratio <= 1.0):
        raise ValueError("ratio must be in (0,1].")

    selected = selector(train_df[X_cols], train_df[args.target], args.task, args.ratio, random_state=args.seed)

    Xtr_df, Xva_df = features.prepare_design_matrices(train_df, valid_df, selected)
    Xtr = Xtr_df.values.astype(np.float32)
    Xva = Xva_df.values.astype(np.float32)

    metrics = models.train_and_eval(
        train_df=train_df,
        valid_df=valid_df,
        X_train=Xtr,
        X_valid=Xva,
        target=args.target,
        task=args.task,
        model_type=args.model,
    )

    record = {
        "params": {
            "dataset": args.dataset,
            "train": args.train,
            "valid": args.valid,
            "task": args.task,
            "target": args.target,
            "exclude": args.exclude,
            "method": args.method,
            "ratio": args.ratio,
            "seed": args.seed,
            "selected_features": selected,
            "model_type": args.model,
        },
        "metrics": metrics,
    }
    utils.write_ndjson(os.path.join(args.log_dir, "metrics.ndjson"), record)

    print("Selected features:", selected)
    print("Metrics:", json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Appended metrics to: {os.path.join(args.log_dir, 'metrics.ndjson')}")

def build_parser():
    p = argparse.ArgumentParser(description="Simple CSV ML with feature selection (PyTorch models).")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("split", help="Split a CSV into train/valid CSVs.")
    sp.add_argument("--dataset", type=str, required=True, help="Path to input CSV.")
    sp.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    sp.add_argument("--seed", type=int, default=42, help="Random seed.")
    sp.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as dataset).")
    sp.add_argument("--train-name", type=str, default="AmesHousing_train.csv", help="Train CSV file name.")
    sp.add_argument("--valid-name", type=str, default="AmesHousing_valid.csv", help="Valid CSV file name.")
    sp.set_defaults(func=cmd_split)

    tp = sub.add_parser("train", help="Train a PyTorch model with feature selection; log metrics.ndjson only.")
    tp.add_argument("--dataset", type=str, help="Path to full CSV (used if --train/--valid not given).")
    tp.add_argument("--train", type=str, help="Path to training CSV (optional).")
    tp.add_argument("--valid", type=str, help="Path to validation CSV (optional).")
    tp.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio if splitting on the fly.")
    tp.add_argument("--seed", type=int, default=42, help="Random seed.")

    tp.add_argument("--task", type=str, choices=["regression", "classification"], required=True, help="Task type.")
    tp.add_argument("--target", type=str, required=True, help="Target column name.")
    tp.add_argument("--exclude", type=str, nargs="*", default=[], help="Columns to exclude (e.g., Order PID).")

    tp.add_argument("--method", type=str, choices=list(features.FEATURE_SELECTORS.keys()), default="corr", help="Feature selection method.")
    tp.add_argument("--ratio", type=float, default=0.3, help="Fraction of features to keep (0,1].")

    tp.add_argument("--model", type=str, choices=["linear", "mlp"], default="linear", help="PyTorch model type.")

    tp.add_argument("--log-dir", type=str, default="logs", help="Directory to store metrics.ndjson only.")
    tp.set_defaults(func=cmd_train)
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
