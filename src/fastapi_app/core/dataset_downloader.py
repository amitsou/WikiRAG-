""" This file contains the DatasetDownloader class for downloading and saving Hugging Face datasets."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.utils.general_utils import load_config, timeit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config/config.yaml"


class DatasetDownloader:
    """Downloads and saves Hugging Face datasets in Parquet format."""

    def __init__(self, config_path=CONFIG_PATH) -> None:
        """Initializes the DatasetDownloader class with configurations."""
        self.config = load_config(str(config_path))
        self.dataset_name = self.config["dataset"]["name"]
        self.subset = self.config["dataset"]["subset"]
        self.save_filename = self.config["dataset"]["dataset_filename"]
        self.save_dir = PROJECT_ROOT / "data"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @timeit
    def download_and_save(self) -> None:
        """
        Downloads the **entire dataset** and saves it as a Parquet file.

        Returns: None
        """
        try:
            print(f"Downloading dataset: {self.dataset_name} ({self.subset})")
            dataset = load_dataset(self.dataset_name, self.subset, split="train")
            df = pd.DataFrame(dataset)
            save_path = self.save_dir / self.save_filename
            df.to_parquet(save_path, index=False)
            print(f"Dataset saved to {save_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
