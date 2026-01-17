import click
import pandas as pd
import re

# =========================
# Text cleaning helpers
# =========================
TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = re.compile(r"\u0640")
DIGITS = re.compile(r"[0-9٠-٩]")
URLS = re.compile(r"http\S+|www\S+")
SPECIAL_CHARS = re.compile(r"[^\w\s\u0600-\u06FF]")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URLS.sub(" ", text)
    text = TASHKEEL.sub("", text)
    text = TATWEEL.sub("", text)
    text = DIGITS.sub("", text)
    text = SPECIAL_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# Preprocess group
# =========================
@click.group()
def preprocess():
    """Preprocessing commands."""
    pass


# =========================
# remove command
# =========================
@preprocess.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV file")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--output", required=True, type=str, help="Output CSV file")
def remove(csv_path, text_col, output):
    """Remove unwanted Arabic-specific characters."""
    click.echo("[PREPROCESS] Step: remove")
    click.echo(f"Input CSV: {csv_path}")
    click.echo(f"Text column: {text_col}")

    df = pd.read_csv(csv_path)
    click.echo(f"CSV loaded | rows={len(df)}")

    if text_col not in df.columns:
        raise click.ClickException(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    before_len = df[text_col].fillna("").astype(str).str.len().mean()
    click.echo("Cleaning text (remove tashkeel/tatweel/digits/urls/special chars)...")

    df[text_col] = df[text_col].fillna("").astype(str).apply(clean_text)
    after_len = df[text_col].str.len().mean()

    df.to_csv(output, index=False)
    click.echo(f"Saved output: {output}")
    click.echo(f"Text length (avg chars) -> before: {before_len:.2f}, after: {after_len:.2f}")


# =========================
# stopwords command
# =========================
@preprocess.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV file")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--output", required=True, type=str, help="Output CSV file")
def stopwords(csv_path, text_col, output):
    """Remove Arabic stopwords."""
    click.echo("[PREPROCESS] Step: stopwords")
    click.echo(f"Input CSV: {csv_path}")
    click.echo(f"Text column: {text_col}")

    df = pd.read_csv(csv_path)
    click.echo(f"CSV loaded | rows={len(df)}")

    if text_col not in df.columns:
        raise click.ClickException(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    arabic_stopwords = {
        "في","على","من","إلى","عن","مع","هذا","هذه","ذلك","تلك",
        "كان","كانت","يكون","تكون","هو","هي","هم","هن",
        "ما","لم","لن","لا","نعم","كل","كما","لكن","أو","أي"
    }

    def remove_stops(text: str) -> str:
        tokens = text.split()
        tokens = [t for t in tokens if t not in arabic_stopwords]
        return " ".join(tokens)

    before_len = df[text_col].fillna("").astype(str).str.split().apply(len).mean()
    click.echo("Removing Arabic stopwords...")

    df[text_col] = df[text_col].fillna("").astype(str).apply(remove_stops)
    after_len = df[text_col].str.split().apply(len).mean()

    df.to_csv(output, index=False)
    click.echo(f"Saved output: {output}")
    click.echo(f"Text length (avg words) -> before: {before_len:.2f}, after: {after_len:.2f}")


# =========================
# replace (normalization) command
# =========================
@preprocess.command()
@click.option("--csv_path", required=True, type=str, help="Path to CSV file")
@click.option("--text_col", required=True, type=str, help="Text column name")
@click.option("--output", required=True, type=str, help="Output CSV file")
def replace(csv_path, text_col, output):
    """Normalize Arabic text (hamza, alef maqsoura, taa marbouta)."""
    click.echo("[PREPROCESS] Step: replace")
    click.echo(f"Input CSV: {csv_path}")
    click.echo(f"Text column: {text_col}")

    df = pd.read_csv(csv_path)
    click.echo(f"CSV loaded | rows={len(df)}")

    if text_col not in df.columns:
        raise click.ClickException(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    click.echo("Normalizing Arabic letters (alef/hamza variants, ى->ي, ة->ه, ...)")
    df[text_col] = df[text_col].fillna("").astype(str).apply(normalize_arabic)

    df.to_csv(output, index=False)
    click.echo(f"Saved output: {output}")
