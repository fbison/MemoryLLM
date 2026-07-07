import os
import urllib.request
from pathlib import Path

FILES = {
    "squad/indices_squad_3.npy": "https://huggingface.co/datasets/YuWangX/KnowledgeRetention/resolve/main/squad/indices_squad_3.npy",
    "squad/dev-v2.0.json": "https://huggingface.co/datasets/YuWangX/KnowledgeRetention/resolve/main/squad/dev-v2.0.json",
    "squad/train-v2.0.json": "https://huggingface.co/datasets/YuWangX/KnowledgeRetention/resolve/main/squad/train-v2.0.json",
    "nq/indices_nq_4.npy": "https://huggingface.co/datasets/YuWangX/KnowledgeRetention/resolve/main/nq/indices_nq_4.npy",
    "nq/v1.0-simplified_nq-dev-all.jsonl": "https://huggingface.co/datasets/YuWangX/KnowledgeRetention/resolve/main/nq/v1.0-simplified_nq-dev-all.jsonl",
    "nq/v1.0-simplified_simplified-nq-train.jsonl": "https://huggingface.co/datasets/YuWangX/KnowledgeRetention/resolve/main/nq/v1.0-simplified_simplified-nq-train.jsonl",
}

def download_file(url, dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Complete: {dest_path} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return False

def main():
    print("="*60)
    print("M+ Paper - Pre-formatted Dataset Downloader")
    print("="*60)
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    success = True
    for relative_path, url in FILES.items():
        dest = data_dir / relative_path
        # Force download to overwrite any incorrect files previously downloaded
        if not download_file(url, dest):
            success = False
                
    if success:
        print("\nAll datasets downloaded and ready to use in ./data/!")
    else:
        print("\nSome datasets failed to download. Please check your internet connection.")

if __name__ == "__main__":
    main()
