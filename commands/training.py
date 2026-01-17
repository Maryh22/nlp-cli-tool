import click
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

# =====================
# Output directories
# =====================
VIS_DIR = Path("outputs/visualizations")
REP_DIR = Path("outputs/reports")
MOD_DIR = Path("outputs/models")
for d in [VIS_DIR, REP_DIR, MOD_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(cm, out_path: Path, title: str):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, zero_division=0)
    return acc, p, r, f1, cm, rep


@click.command()
@click.option("--csv_path", required=True, type=str, help="CSV file with labels")
@click.option("--input_col", required=True, type=str, help="Embeddings .pkl file (tfidf_vectors.pkl)")
@click.option("--output_col", required=True, type=str, help="Label column name (e.g., label)")
@click.option("--test_size", default=0.2, type=float, show_default=True)
@click.option(
    "--models",
    multiple=True,
    type=click.Choice(["knn", "lr", "rf"]),
    required=True,
    help="Models to train: repeat --models for each (e.g., --models knn --models lr)",
)
def train(csv_path, input_col, output_col, test_size, models):
    """
    Train and evaluate classification models using precomputed embeddings.
    Prints progress so it's clear what is happening during execution.
    """

    # ---------------------
    # Load CSV / labels
    # ---------------------
    click.echo(" Loading CSV and labels...")
    df = pd.read_csv(csv_path)
    click.echo(f" Loaded CSV: {csv_path} | rows={len(df)}")

    if output_col not in df.columns:
        raise click.ClickException(
            f"Column '{output_col}' not found. Available columns: {list(df.columns)}"
        )

    y = df[output_col].values
    click.echo(f" Labels ready | classes={len(set(y))}")

    # ---------------------
    # Load embeddings
    # ---------------------
    click.echo(" Loading embeddings (TF-IDF vectors)...")
    with open(input_col, "rb") as f:
        obj = pickle.load(f)

    if "X" not in obj:
        raise click.ClickException("Embeddings file must contain key 'X'.")

    X = obj["X"]
    click.echo(f" Embeddings loaded | shape={X.shape}")

    # ---------------------
    # Split
    # ---------------------
    click.echo(" Splitting train/test...")
    strat = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=strat
    )
    click.echo(
        f" Split done | train={X_train.shape[0]} test={X_test.shape[0]} test_size={test_size}"
    )

    # ---------------------
    # Prepare report
    # ---------------------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = REP_DIR / f"training_report_{ts}.md"

    lines = []
    lines.append(f"## Training Report - {ts}\n\n")
    lines.append("### Dataset Info\n")
    lines.append(f"- CSV: {csv_path}\n")
    lines.append(f"- Total samples: {len(df)}\n")
    lines.append(f"- Train/Test split: {X_train.shape[0]}/{X_test.shape[0]}\n")
    lines.append(f"- Classes: {len(set(y))}\n")
    lines.append(f"- Features: {X.shape[1]}\n\n")
    lines.append("### Model Performance\n\n")

    results = {}  # model_name -> accuracy

    # =====================
    # KNN
    # =====================
    if "knn" in models:
        click.echo(" Training KNN...")
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)

        click.echo(" Evaluating KNN...")
        preds = knn.predict(X_test)

        acc, p, r, f1, cm, rep = compute_metrics(y_test, preds)
        cm_path = VIS_DIR / f"confusion_knn_{ts}.png"
        save_confusion_matrix(cm, cm_path, "Confusion Matrix - KNN")

        results["KNN"] = acc

        lines.append("#### KNN (default)\n")
        lines.append(f"- Accuracy: {acc:.4f}\n")
        lines.append(f"- Precision: {p:.4f}\n")
        lines.append(f"- Recall: {r:.4f}\n")
        lines.append(f"- F1-Score: {f1:.4f}\n")
        lines.append(f"- Confusion Matrix: {cm_path}\n\n")
        lines.append("```text\n" + rep + "\n```\n\n")

        click.echo(f" KNN done | acc={acc:.4f}, f1={f1:.4f}")
        click.echo(f"  Saved KNN confusion matrix: {cm_path}")

    # =====================
    # Logistic Regression
    # =====================
    if "lr" in models:
        click.echo(" Training Logistic Regression...")
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train, y_train)

        click.echo(" Evaluating Logistic Regression...")
        preds = lr.predict(X_test)

        acc, p, r, f1, cm, rep = compute_metrics(y_test, preds)
        cm_path = VIS_DIR / f"confusion_lr_{ts}.png"
        save_confusion_matrix(cm, cm_path, "Confusion Matrix - Logistic Regression")

        results["Logistic Regression"] = acc

        lines.append("#### Logistic Regression (default)\n")
        lines.append(f"- Accuracy: {acc:.4f}\n")
        lines.append(f"- Precision: {p:.4f}\n")
        lines.append(f"- Recall: {r:.4f}\n")
        lines.append(f"- F1-Score: {f1:.4f}\n")
        lines.append(f"- Confusion Matrix: {cm_path}\n\n")
        lines.append("```text\n" + rep + "\n```\n\n")

        click.echo(f" LR done | acc={acc:.4f}, f1={f1:.4f}")
        click.echo(f"  Saved LR confusion matrix: {cm_path}")

    # =====================
    # Random Forest
    # =====================
    if "rf" in models:
        click.echo(" Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)

        click.echo(" Evaluating Random Forest...")
        preds = rf.predict(X_test)

        acc, p, r, f1, cm, rep = compute_metrics(y_test, preds)
        cm_path = VIS_DIR / f"confusion_rf_{ts}.png"
        save_confusion_matrix(cm, cm_path, "Confusion Matrix - Random Forest")

        results["Random Forest"] = acc

        lines.append("#### Random Forest (n_estimators=200)\n")
        lines.append(f"- Accuracy: {acc:.4f}\n")
        lines.append(f"- Precision: {p:.4f}\n")
        lines.append(f"- Recall: {r:.4f}\n")
        lines.append(f"- F1-Score: {f1:.4f}\n")
        lines.append(f"- Confusion Matrix: {cm_path}\n\n")
        lines.append("```text\n" + rep + "\n```\n\n")

        click.echo(f" RF done | acc={acc:.4f}, f1={f1:.4f}")
        click.echo(f" Saved RF confusion matrix: {cm_path}")

    # ---------------------
    # Best model
    # ---------------------
    if len(results) == 0:
        raise click.ClickException("No models trained. Use --models knn and/or lr and/or rf.")

    best_model = max(results, key=results.get)
    lines.append("### Best Model \n")
    lines.append(f"- **{best_model}** (accuracy={results[best_model]:.4f})\n")

    report_path.write_text("".join(lines), encoding="utf-8")

    click.echo(f"Report saved: {report_path}")
    click.echo(f" Best model: {best_model}")
