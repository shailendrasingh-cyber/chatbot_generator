# Run this in your Python environment before starting the app
import urllib.request
import zipfile
import os
from pathlib import Path

nltk_data_dir = Path("nltk_data")
nltk_data_dir.mkdir(exist_ok=True)

# Download punkt_tab manually
punkt_tab_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
zip_path = nltk_data_dir / "punkt_tab.zip"
urllib.request.urlretrieve(punkt_tab_url, zip_path)

# Extract to correct location
extract_path = nltk_data_dir / "tokenizers" / "punkt_tab"
extract_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path.parent)
os.remove(zip_path)