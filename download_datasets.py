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

def download_file_with_resume(url, dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Get remote file size
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            remote_size = int(resp.getheader('Content-Length', 0))
    except Exception as e:
        print(f"  ✗ Failed to get remote file size for {url}: {e}")
        remote_size = 0

    local_size = dest_path.stat().st_size if dest_path.exists() else 0
    
    # 2. Check if already complete
    if remote_size > 0 and local_size == remote_size:
        print(f"✓ Complete: {dest_path} ({local_size / (1024*1024):.2f} MB)")
        return True
        
    if local_size > remote_size:
        print(f"⚠ Local file is larger than remote ({local_size} > {remote_size} bytes). Overwriting...")
        local_size = 0
        
    # 3. Prepare request with Range header if resuming
    req = urllib.request.Request(url)
    if local_size > 0:
        print(f"→ Resuming download of {dest_path} from byte {local_size} (already have {local_size / (1024*1024):.2f} MB of {remote_size / (1024*1024):.2f} MB)...")
        req.add_header('Range', f'bytes={local_size}-')
        mode = 'ab'
    else:
        print(f"→ Downloading {dest_path} ({remote_size / (1024*1024):.2f} MB)...")
        mode = 'wb'
        
    # 4. Perform download
    try:
        with urllib.request.urlopen(req) as response, open(dest_path, mode) as out_file:
            # Check if server responded with 206 Partial Content when we requested a Range
            if local_size > 0 and response.status != 206:
                print("  ⚠ Server did not support Range requests. Downloading from scratch...")
                out_file.close()
                with open(dest_path, 'wb') as new_out:
                    block_size = 1024 * 1024
                    downloaded = 0
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        new_out.write(buffer)
                        downloaded += len(buffer)
                        percent = (downloaded / remote_size) * 100 if remote_size else 0
                        print(f"\r  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{remote_size / (1024*1024):.1f} MB)", end="", flush=True)
                    print()
            else:
                block_size = 1024 * 1024  # 1MB blocks
                downloaded = local_size
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    percent = (downloaded / remote_size) * 100 if remote_size else 0
                    print(f"\r  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{remote_size / (1024*1024):.1f} MB)", end="", flush=True)
                print()
        
        # Verify size matches remote size
        final_size = dest_path.stat().st_size
        if remote_size > 0 and final_size != remote_size:
            print(f"  ✗ Incomplete: local size {final_size} != remote size {remote_size}")
            return False
        print(f"  ✓ Complete: {dest_path} ({final_size / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False

def main():
    print("="*60)
    print("M+ Paper - Resumable Dataset Downloader")
    print("="*60)
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    success = True
    for relative_path, url in FILES.items():
        dest = data_dir / relative_path
        if not download_file_with_resume(url, dest):
            success = False
                
    if success:
        print("\nAll datasets downloaded and ready to use in ./data/!")
    else:
        print("\nSome datasets failed or are incomplete. Please check your internet connection and rerun the script to resume.")

if __name__ == "__main__":
    main()
