import click
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


@click.group()
def embed():
    """Embedding commands."""
    pass


@embed.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV file")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--max_features", default=5000, type=int, show_default=True)
@click.option("--output", required=True, type=str, help="Output pickle file")
def tfidf(csv_path, text_col, max_features, output):
    """TF-IDF Embedding (sklearn)."""

    click.echo("[EMBED] Step: TF-IDF")
    click.echo(f"Input CSV: {csv_path}")
    click.echo(f"Text column: {text_col}")
    click.echo(f"Max features: {max_features}")

    df = pd.read_csv(csv_path)
    click.echo(f"CSV loaded | rows={len(df)}")

    if text_col not in df.columns:
        raise click.ClickException(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    texts = df[text_col].fillna("").astype(str).tolist()
    click.echo("Building TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)

    click.echo(f"TF-IDF matrix created | shape={X.shape}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "X": X,
                "vectorizer": vectorizer,
            },
            f,
        )

    click.echo(f"Saved embeddings: {out_path}")
