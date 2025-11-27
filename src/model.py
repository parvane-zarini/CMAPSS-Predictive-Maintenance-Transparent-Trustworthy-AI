import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os


def train_rf(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, set_name, output_dir):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    print(f"\n=== Classification Report ({set_name}) ===")
    print(classification_report(y, y_pred))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {set_name}")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"confusion_matrix_{set_name}.png"), dpi=300)
    plt.close()
