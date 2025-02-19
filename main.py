""" Main script to run the dataset pipeline for RAG. """

import argparse

from src.fastapi_app.core.dataset_downloader import DatasetDownloader
from src.fastapi_app.core.qdrant_manager import QdrantManager


def parse_args():
    parser = argparse.ArgumentParser(description=("Run the dataset pipeline for RAG."))

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset.",
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest dataset into Qdrant.",
    )

    parser.add_argument(
        "--show_records",
        action="store_true",
        help="Show records in Qdrant.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.download:
        DatasetDownloader().download_and_save()

    if args.ingest:
        QdrantManager().ingest_data()

    if args.show_records:
        QdrantManager().show_records()


if __name__ == "__main__":
    main()
