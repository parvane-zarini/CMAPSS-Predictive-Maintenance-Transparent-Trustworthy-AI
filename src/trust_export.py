import numpy as np
import pandas as pd
import os


def export_trust_csv(model, scaler, df_original, feature_cols,
                     X_scaled, output_dir, n_samples=50):

    os.makedirs(output_dir, exist_ok=True)

    if df_original.shape[0] > n_samples:
        df_sample = df_original.sample(n_samples, random_state=42)
        X_sample = X_scaled[df_sample.index]
    else:
        df_sample = df_original
        X_sample = X_scaled

    global_importance = model.feature_importances_
    proba = model.predict_proba(X_sample)[:, 1]

    top1, top2, top3 = [], [], []
    score1, score2, score3 = [], [], []

    for i in range(len(df_sample)):
        local_scores = np.abs(X_sample[i]) * global_importance
        idx = np.argsort(local_scores)[::-1][:3]

        top1.append(feature_cols[idx[0]])
        top2.append(feature_cols[idx[1]])
        top3.append(feature_cols[idx[2]])

        score1.append(local_scores[idx[0]])
        score2.append(local_scores[idx[1]])
        score3.append(local_scores[idx[2]])

    df_out = df_sample.copy()
    df_out["model_output"] = proba
    df_out["top1_feature"] = top1
    df_out["top2_feature"] = top2
    df_out["top3_feature"] = top3
    df_out["top1_score"] = score1
    df_out["top2_score"] = score2
    df_out["top3_score"] = score3

    df_out.to_csv(os.path.join(output_dir, "trust_study_samples_no_shap.csv"),
                  index=False)
