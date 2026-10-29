"""
GTZAN Dataset Downloader and Extractor.
GTZAN is the standard benchmark dataset for Music Genre Classification:
- 1,000 audio tracks (30 seconds each, 22050 Hz, 16-bit mono wav/au)
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- 100 tracks per genre
"""

import sys
import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, DATA_DIR

GTZAN_URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
BACKUP_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_gtzan(dest_dir: Path = RAW_DATA_DIR) -> bool:
    """
    Downloads genres.tar.gz and extracts tracks into dest_dir/<genre>/*.wav
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "genres.tar.gz"

    # Check if dataset already exists
    subfolders = [f for f in dest_dir.iterdir() if f.is_dir()]
    if len(subfolders) >= 10:
        total_files = sum(len(list(f.glob("*.*"))) for f in subfolders)
        if total_files >= 100:
            print(f"GTZAN dataset already detected in {dest_dir} ({total_files} audio files found).")
            return True

    print("=" * 65)
    print("GTZAN Music Genre Dataset Downloader")
    print("=" * 65)
    print("Size: ~1.2 GB (1,000 tracks across 10 genres)")

    if not archive_path.exists():
        urls = [GTZAN_URL, BACKUP_URL]
        success = False
        for url in urls:
            try:
                print(f"\nAttempting download from: {url}")
                with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading GTZAN") as t:
                    urllib.request.urlretrieve(url, filename=str(archive_path), reporthook=t.update_to)
                success = True
                break
            except Exception as e:
                print(f"Failed from {url}: {e}")

        if not success:
            print("\n" + "!" * 65)
            print("Automatic download could not reach the server.")
            print("Manual Download Instructions:")
            print("1. Download 'genres.tar.gz' from:")
            print("   https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
            print(f"2. Extract the 'genres' folder into: {dest_dir.resolve()}")
            print("!" * 65 + "\n")
            return False

    print("\nExtracting archive...")
    try:
        with tarfile.open(str(archive_path), "r:gz") as tar:
            tar.extractall(path=str(dest_dir.parent))
        
        # In case it extracts to data/genres, move contents to data/raw
        extracted_genres = dest_dir.parent / "genres"
        if extracted_genres.exists() and extracted_genres != dest_dir:
            for item in extracted_genres.iterdir():
                target_folder = dest_dir / item.name
                target_folder.mkdir(parents=True, exist_ok=True)
                for file in item.glob("*.*"):
                    file.rename(target_folder / file.name)
        
        print(f"Dataset successfully extracted to: {dest_dir}")
        return True
    except Exception as e:
        print(f"Extraction error: {e}")
        return False


if __name__ == "__main__":
    download_and_extract_gtzan()
