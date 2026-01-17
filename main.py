import click
from pathlib import Path

from commands.eda import eda
from commands.preprocessing import preprocess
from commands.embedding import embed
from commands.training import train

@click.group()
def cli():
    """Arabic NLP Classification CLI Tool."""
    pass

cli.add_command(eda)
cli.add_command(preprocess)
cli.add_command(embed)
cli.add_command(train)


@cli.command()
@click.option("--csv_path", required=True, type=str, help="Input CSV file")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--label_col", required=True, type=str, help="Label column name")
@click.option("--max_features", default=5000, type=int, show_default=True)
@click.option("--test_size", default=0.2, type=float, show_default=True)
@click.option(
    "--models",
    multiple=True,
    type=click.Choice(["knn", "lr", "rf"]),
    default=("knn", "lr", "rf"),
    show_default=True,
)
@click.option("--workdir", default="outputs/pipeline", show_default=True, help="Folder to save intermediate files")
def pipeline(csv_path, text_col, label_col, max_features, test_size, models, workdir):
    """
    Run full pipeline in one command:
    preprocess (remove -> stopwords -> replace) -> embed tfidf -> train
    """
    click.echo("[PIPELINE] Starting full pipeline...")

    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)

    step1 = w / "cleaned.csv"
    step2 = w / "no_stops.csv"
    step3 = w / "normalized.csv"
    emb_path = w / "tfidf_vectors.pkl"

    # 1) preprocess remove
    click.echo("[PIPELINE] Preprocess: remove")
    from commands.preprocessing import remove as cmd_remove
    cmd_remove.callback(csv_path=csv_path, text_col=text_col, output=str(step1))

    # 2) preprocess stopwords
    click.echo("[PIPELINE] Preprocess: stopwords")
    from commands.preprocessing import stopwords as cmd_stop
    cmd_stop.callback(csv_path=str(step1), text_col=text_col, output=str(step2))

    # 3) preprocess replace
    click.echo("[PIPELINE] Preprocess: replace")
    from commands.preprocessing import replace as cmd_replace
    cmd_replace.callback(csv_path=str(step2), text_col=text_col, output=str(step3))

    # 4) embed tfidf
    click.echo("[PIPELINE] Embedding: tfidf")
    from commands.embedding import tfidf as cmd_tfidf
    cmd_tfidf.callback(csv_path=str(step3), text_col=text_col, max_features=max_features, output=str(emb_path))

    # 5) train
    click.echo("[PIPELINE] Training...")
    from commands.training import train as cmd_train
    cmd_train.callback(
        csv_path=str(step3),
        input_col=str(emb_path),
        output_col=label_col,
        test_size=test_size,
        models=models,
    )

    click.echo("[PIPELINE] Done.")
    click.echo(f"[PIPELINE] Outputs saved under: {w}")


if __name__ == "__main__":
    cli()
