import click
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

VIS_DIR = Path("outputs") / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

@click.group()
def eda():
    """EDA commands."""
    pass


@eda.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV file")
@click.option("--label_col", required=True, type=str, help="Label column name")
@click.option(
    "--plot_type",
    type=click.Choice(["pie", "bar"]),
    default="pie",
    show_default=True,
)
def distribution(csv_path, label_col, plot_type):
    """View class distribution (pie/bar chart)."""

    click.echo(" [EDA] Loading CSV file...")
    df = pd.read_csv(csv_path)
    click.echo(f" CSV loaded | rows={len(df)}")

    if label_col not in df.columns:
        raise click.ClickException(
            f"Column '{label_col}' not found. Available columns: {list(df.columns)}"
        )

    click.echo(" Computing class distribution...")
    counts = df[label_col].value_counts(dropna=False)
    click.echo(f" Found {len(counts)} classes")
    click.echo(f" Plot type: {plot_type}")

    plt.figure()
    if plot_type == "bar":
        counts.plot(kind="bar")
        plt.title("Class Distribution (Bar)")
        plt.xlabel("Class")
        plt.ylabel("Count")
    else:
        counts.plot(kind="pie", autopct="%1.1f%%")
        plt.title("Class Distribution (Pie)")
        plt.ylabel("")

    out_path = VIS_DIR / f"class_distribution_{plot_type}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    click.echo(f" Saved visualization: {out_path}")
    click.echo(f" Classes: {len(counts)} | Total samples: {int(counts.sum())}")


@eda.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV file")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option(
    "--unit",
    type=click.Choice(["words", "chars"]),
    default="words",
    show_default=True,
)
def histogram(csv_path, text_col, unit):
    """Generate text length histogram (words/chars)."""

    click.echo(" [EDA] Loading CSV file...")
    df = pd.read_csv(csv_path)
    click.echo(f" CSV loaded | rows={len(df)}")

    if text_col not in df.columns:
        raise click.ClickException(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    click.echo(f"📏 Computing text length histogram | unit={unit}")
    texts = df[text_col].fillna("").astype(str)

    if unit == "chars":
        lengths = texts.apply(len)
        title = "Text Length Histogram (Characters)"
    else:
        lengths = texts.apply(lambda x: len(x.split()))
        title = "Text Length Histogram (Words)"

    plt.figure()
    plt.hist(lengths, bins=30)
    plt.title(title)
    plt.xlabel(unit)
    plt.ylabel("Frequency")

    out_path = VIS_DIR / f"text_length_hist_{unit}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    click.echo(f" Saved visualization: {out_path}")
    click.echo(
        f" Stats -> mean: {lengths.mean():.2f}, "
        f"median: {lengths.median():.2f}, "
        f"std: {lengths.std():.2f}, "
        f"min: {lengths.min()}, "
        f"max: {lengths.max()}"
    )
@eda.command(name="remove-outliers")
@click.option("--csv_path", required=True, type=str, help="Path to CSV file")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--method", type=click.Choice(["iqr"]), default="iqr", show_default=True)
@click.option("--unit", type=click.Choice(["words", "chars"]), default="words", show_default=True)
@click.option("--output", required=True, type=str, help="Output CSV file")
def remove_outliers(csv_path, text_col, method, unit, output):
    """Remove outliers based on text length using IQR."""
    click.echo("[EDA] Step: remove-outliers")
    click.echo(f"Input CSV: {csv_path}")
    click.echo(f"Text column: {text_col}")
    click.echo(f"Method: {method} | Unit: {unit}")

    df = pd.read_csv(csv_path)
    click.echo(f"CSV loaded | rows={len(df)}")

    if text_col not in df.columns:
        raise click.ClickException(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    texts = df[text_col].fillna("").astype(str)

    if unit == "chars":
        lengths = texts.apply(len)
    else:
        lengths = texts.apply(lambda x: len(x.split()))

    q1 = lengths.quantile(0.25)
    q3 = lengths.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (lengths >= lower) & (lengths <= upper)
    removed = int((~mask).sum())

    df_clean = df.loc[mask].copy()
    df_clean.to_csv(output, index=False)

    click.echo(f"Thresholds -> lower={lower:.2f}, upper={upper:.2f}")
    click.echo(f"Removed outliers: {removed}")
    click.echo(f"Saved cleaned CSV: {output} | rows={len(df_clean)}")

