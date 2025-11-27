import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("id")["cycle"].max().reset_index()
    max_cycles.columns = ["id", "max_cycle"]
    df = df.merge(max_cycles, on="id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    return df.drop(columns=["max_cycle"])


def add_binary_label(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    df["label"] = (df["RUL"] <= threshold).astype(int)
    return df


def select_features(df: pd.DataFrame, drop_cols: Optional[List[str]] = None):
    drop_cols = drop_cols or []
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols].copy(), feature_cols


def split_and_scale(X, y, val_size, random_state):
    scaler = StandardScaler()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        stratify=y,
        random_state=random_state
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "y_train": y_train.values,
        "y_val": y_val.values,
        "scaler": scaler,
        "X_train_df": X_train,
        "X_val_df": X_val,
    }
