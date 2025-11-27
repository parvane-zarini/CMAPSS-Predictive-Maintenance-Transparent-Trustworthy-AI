import numpy as np
import matplotlib.pyplot as plt
import os


def plot_feature_importance(model, feature_names, output_dir):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(8, 8))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title("Global Feature Importance - Random Forest")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "rf_feature_importance.png"), dpi=300)
    plt.close()


def plot_local_explanation(model, X_scaled, feature_names, df_original,
                           output_dir, instance_index=0, top_k=10):

    x = X_scaled[instance_index]
    gi = model.feature_importances_
    local_scores = np.abs(x) * gi
    idx = np.argsort(local_scores)[-top_k:]

    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in idx], local_scores[idx])
    plt.title("Local Explanation - Important Sensors for One Engine")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "local_explanation_example.png"), dpi=300)
    plt.close()
