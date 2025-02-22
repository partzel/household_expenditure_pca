from pathlib import Path
import typer
from loguru import logger
import pandas as pd
from household_expenditure_pca.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv"
):
    REPO_ID = r"CODE-BLOCK/Weekly_household_expenditure"

    logger.info(f"Fetching Hugging Face dataset '{REPO_ID}'...")

    df = pd.read_parquet(f"hf://datasets/{REPO_ID}/data/train-00000-of-00001.parquet")
    df.to_csv(f"{input_path}")

    logger.success("Downloading dataset complete.")
    logger.info(f"You can find the data set in {input_path}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
