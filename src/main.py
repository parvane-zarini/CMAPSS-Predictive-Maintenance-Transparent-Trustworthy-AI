from config import Config
from data_loader import load_cmapss
from preprocessing import (
    add_rul, add_binary_label, select_features, split_and_scale
)
from model import train_rf, evaluate
from explainability import plot_feature_importance, plot_local_explanation
from trust_export import export_trust_csv
import os


def run():

    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"Using dataset: {cfg.FD}")

    # Load
    train_df, test_df, rul_df = load_cmapss(cfg.FD, cfg.DATA_DIR)

    # RUL + Label
    train_df = add_rul(train_df)
    test_df = add_rul(test_df)

    train_df = add_binary_label(train_df, cfg.RUL_THRESHOLD)
    test_df = add_binary_label(test_df, cfg.RUL_THRESHOLD)

    # Features
    drop_cols = ["id", "cycle", "RUL", "label"]
    X_train_df, feature_cols = select_features(train_df, drop_cols)
    y_train = train_df["label"]

    X_test_df, _ = select_features(test_df, drop_cols)
    y_test = test_df["label"]

    # Split + Scale
    split_data = split_and_scale(
        X_train_df, y_train,
        cfg.VAL_SIZE, cfg.RANDOM_STATE
    )

    X_train = split_data["X_train"]
    X_val = split_data["X_val"]
    y_train_arr = split_data["y_train"]
    y_val = split_data["y_val"]
    scaler = split_data["scaler"]

    X_test_scaled = scaler.transform(X_test_df)

    # Train
    model = train_rf(X_train, y_train_arr, cfg.RANDOM_STATE)

    # Evaluate
    evaluate(model, X_train, y_train_arr, "Train", cfg.OUTPUT_DIR)
    evaluate(model, X_val, y_val, "Validation", cfg.OUTPUT_DIR)
    evaluate(model, X_test_scaled, y_test.values, "Test", cfg.OUTPUT_DIR)

    # Explainability
    plot_feature_importance(model, feature_cols, cfg.OUTPUT_DIR)
    plot_local_explanation(
        model, X_test_scaled,
        feature_cols, X_test_df,
        cfg.OUTPUT_DIR, instance_index=0, top_k=10
    )

    # CSV for Trust Study
    export_trust_csv(
        model, scaler, test_df,
        feature_cols, X_test_scaled,
        cfg.OUTPUT_DIR, cfg.TRUST_SAMPLES
    )


if __name__ == "__main__":
    run()
