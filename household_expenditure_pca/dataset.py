from pathlib import Path
import typer
from loguru import logger
import pandas as pd
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def fill_na_values(df):
    num_cols=df.select_dtypes(include="float64").columns
    cat_cols=df.select_dtypes(include="object").columns
    for col in num_cols:
        df.loc[:,col] = df.loc[:,col].fillna(df[col].mean())
    for col in cat_cols:
        mode_value = df[col].mode()[0]
        df.loc[:,col] = df.loc[:,col].fillna(mode_value)

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

    logger.info(f"Applying eda ...")
    fill_na_values(df)
    df.to_csv(f"{output_path}")
    logger.info(f"You can find the processed data set in {output_path}")

if __name__ == "__main__":
    app()
