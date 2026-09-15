"""
download_datasets.py

Downloads all three demo datasets via curl and extracts them into
demo_models/datasets/<name>/. Zip files are deleted after extraction.

Usage:
    python download_datasets.py
"""

import os
import zipfile
import subprocess

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

# Kaggle sources: downloaded as .zip, extracted, then zip is deleted
ZIP_SOURCES = [
    (
        "https://www.kaggle.com/api/v1/datasets/download/janiobachmann/bank-marketing-dataset",
        "bank_marketing",
        "bank-marketing-dataset.zip",
    ),
    (
        "https://www.kaggle.com/api/v1/datasets/download/yasserh/housing-prices-dataset",
        "house_prices",
        "housing-prices-dataset.zip",
    ),
]

# Direct file sources: downloaded as-is, no extraction needed
FILE_SOURCES = [
    (
        "https://raw.githubusercontent.com/Dawit-1621/BBC-News-Classification/main/Data/BBC%20News%20Train.csv",
        "bbc_news",
        "BBC_News_Train.csv",
    ),
]


def download_and_extract_zip(url, dest_name, zip_filename):
    dest_folder = os.path.join(DATASETS_DIR, dest_name)
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, zip_filename)

    print(f"\nDownloading -> {dest_folder}")
    print(f"  URL: {url}")
    subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)

    print(f"  Extracting {zip_filename} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_folder)

    os.remove(zip_path)
    print(f"  Deleted {zip_filename}")
    print(f"  Done: {dest_name}")


def download_file(url, dest_name, filename):
    dest_folder = os.path.join(DATASETS_DIR, dest_name)
    os.makedirs(dest_folder, exist_ok=True)
    file_path = os.path.join(dest_folder, filename)

    print(f"\nDownloading -> {dest_folder}")
    print(f"  URL: {url}")
    subprocess.run(["curl", "-L", "-o", file_path, url], check=True)

    print(f"  Done: {dest_name}")


def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    for url, dest_name, zip_filename in ZIP_SOURCES:
        download_and_extract_zip(url, dest_name, zip_filename)

    for url, dest_name, filename in FILE_SOURCES:
        download_file(url, dest_name, filename)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()