import os
import pandas as pd
from typing import Tuple, List

CMAPSS_COLS: List[str] = [
    "id", "cycle",
    "os1", "os2", "os3",
] + [f"s{i}" for i in range(1, 22)]


def load_cmapss(fd: str, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_file = os.path.join(data_dir, f"train_{fd}.txt")
    test_file = os.path.join(data_dir, f"test_{fd}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{fd}.txt")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")

    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=CMAPSS_COLS)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=CMAPSS_COLS)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL"])

    return train_df, test_df, rul_df
