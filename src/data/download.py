import requests
import os

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_tinyshakespeare():
    os.makedirs(DATASET_DIR, exist_ok=True)

    output_path = os.path.join(DATASET_DIR, "raw.txt")

    if os.path.exists(output_path):
        print(f"Already exists: {output_path} — skipping download.")
        return

    print("Downloading TinyShakespeare...")
    response = requests.get(URL)
    response.raise_for_status()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Saved to {output_path}")
    print(f"Total characters: {len(response.text):,}")

if __name__ == "__main__":
    download_tinyshakespeare()