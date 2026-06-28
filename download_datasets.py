#!/usr/bin/env python3
"""
Download and prepare datasets for M+ paper tests.
Saves to ./data/ directory for use with test_qa_memory.py
Intelligently skips downloads if files already exist.
"""

import os
import json
import sys
from pathlib import Path

# Configuration
DOWNLOAD_TRAIN_SETS = False  # Set to True to also download training sets for --nuc > 0

def check_file_exists(filepath, description):
   """Check if a file exists and report its size"""
   path = Path(filepath)
   if path.exists():
       size_mb = path.stat().st_size / (1024 * 1024)
       print(f"  ✓ Found: {filepath} ({size_mb:.1f} MB)")
       return True
   else:
       print(f"  ✗ Missing: {filepath}")
       return False


def verify_file_integrity(filepath):
   """
   Verify file integrity and completeness.
   Returns: 'valid', 'empty', 'corrupted', or 'missing'
   """
   path = Path(filepath)

   # Check if file exists
   if not path.exists():
       return 'missing'

   # Check if file is empty
   file_size = path.stat().st_size
   if file_size == 0:
       return 'empty'

   # Try to verify based on file type
   try:
       if filepath.endswith('.jsonl'):
           # For JSONL, try to read first and last line
           with open(filepath, 'r') as f:
               first_line = f.readline()
               if not first_line:
                   return 'empty'
               json.loads(first_line)  # Try to parse
               # Try to read last line
               f.seek(0, 2)  # Go to end
               pos = f.tell()
               if pos > 1:
                   f.seek(pos - 1)
                   while f.read(1) != b'\n' and f.tell() > 1:
                       f.seek(f.tell() - 2)
                   last_line = f.readline().decode('utf-8', errors='ignore')
                   if last_line.strip():
                       json.loads(last_line)  # Try to parse
           return 'valid'

       elif filepath.endswith('.json'):
           # For JSON, try to parse entire file
           with open(filepath, 'r') as f:
               json.load(f)
           return 'valid'

       elif filepath.endswith('.npy'):
           # For numpy files, try to load
           try:
               import numpy as np
               np.load(filepath, allow_pickle=False)
               return 'valid'
           except Exception:
               return 'corrupted'

       else:
           # For unknown types, just check size
           if file_size > 0:
               return 'valid'
           return 'empty'

   except Exception as e:
       print(f"    [Integrity check failed: {str(e)[:50]}]")
       return 'corrupted'


def delete_file_if_exists(filepath):
   """Delete a file if it exists, return True if deleted"""
   path = Path(filepath)
   if path.exists():
       try:
           path.unlink()  # Delete the file
           return True
       except Exception as e:
           print(f"    ⚠ Failed to delete {filepath}: {e}")
           return False
   return False

def download_naturalqa():
   """Download Natural Questions validation dataset"""
   print("\n" + "="*60)
   print("Natural Questions Dataset")
   print("="*60)

   try:
       from datasets import load_dataset
   except ImportError:
       print("ERROR: 'datasets' library not found")
       print("Install with: pip install datasets")
       return False

   # Create directory
   nq_dir = Path("./data/nq")
   nq_dir.mkdir(parents=True, exist_ok=True)

   # Check what exists
   val_file = nq_dir / "v1.0-simplified_nq-dev-all.jsonl"
   train_file = nq_dir / "v1.0-simplified_simplified-nq-train.jsonl"
   indices_file = nq_dir / "indices_nq_4.npy"

   print("\nChecking existing files:")

   # Check validation file integrity
   val_status = verify_file_integrity(val_file)
   if val_status == 'valid':
       size_mb = val_file.stat().st_size / (1024 * 1024)
       print(f"  ✓ Valid: {val_file} ({size_mb:.1f} MB)")
       val_needs_download = False
   elif val_status == 'missing':
       print(f"  ✗ Missing: {val_file}")
       val_needs_download = True
   else:  # corrupted or empty
       print(f"  ⚠ {val_status.capitalize()}: {val_file} - will re-download")
       val_needs_download = True

   # Check training file integrity (if enabled)
   if DOWNLOAD_TRAIN_SETS:
       train_status = verify_file_integrity(train_file)
       if train_status == 'valid':
           size_mb = train_file.stat().st_size / (1024 * 1024)
           print(f"  ✓ Valid: {train_file} ({size_mb:.1f} MB)")
           train_needs_download = False
       elif train_status == 'missing':
           print(f"  ✗ Missing: {train_file}")
           train_needs_download = True
       else:
           print(f"  ⚠ {train_status.capitalize()}: {train_file} - will re-download")
           train_needs_download = True
   else:
       print(f"  ⊘ Skipped: {train_file} (DOWNLOAD_TRAIN_SETS = False)")
       train_needs_download = False

   # Check indices file
   indices_status = verify_file_integrity(indices_file)
   if indices_status == 'valid':
       size_mb = indices_file.stat().st_size / (1024 * 1024)
       print(f"  ✓ Valid: {indices_file} ({size_mb:.1f} MB)")
   elif indices_status == 'missing':
       print(f"  ✗ Missing: {indices_file}")
   else:
       print(f"  ⚠ {indices_status.capitalize()}: {indices_file}")

   # Download validation if needed
   if val_needs_download:
       if val_status in ['empty', 'corrupted']:
           print(f"\n→ Deleting corrupted validation file...")
           delete_file_if_exists(val_file)
       print(f"→ Downloading Natural Questions validation split...")
       ds = load_dataset('natural_questions', split='validation')
       print(f"  Writing {len(ds)} samples to {val_file}...")
       with open(val_file, 'w') as f:
           for example in ds:
               f.write(json.dumps(example) + '\n')
       print(f"  ✓ Saved: {val_file}")
   else:
       print(f"\n→ Validation set valid, skipping download")

   # Download training if needed and enabled
   if DOWNLOAD_TRAIN_SETS and train_needs_download:
       if train_status in ['empty', 'corrupted']:
           print(f"\n→ Deleting corrupted training file...")
           delete_file_if_exists(train_file)
       print(f"→ Downloading Natural Questions training split...")
       print("  (This is used for --nuc > 0 unrelated contexts)")
       ds = load_dataset('natural_questions', split='train')
       print(f"  Writing {len(ds)} samples to {train_file}...")
       with open(train_file, 'w') as f:
           for example in ds:
               f.write(json.dumps(example) + '\n')
       print(f"  ✓ Saved: {train_file}")
   elif DOWNLOAD_TRAIN_SETS and not train_needs_download:
       print(f"\n→ Training set valid, skipping download")

   # Note about indices
   if indices_status != 'valid':
       print(f"\nℹ️  WARNING: indices_nq_4.npy not found or corrupted!")
       print(f"   This file should be downloaded from:")
       print(f"   https://huggingface.co/datasets/YuWangX/KnowledgeRetention")
       print(f"   Place it as: data/nq/indices_nq_4.npy")
   else:
       print(f"\n✓ Indices file valid")

   return val_status == 'valid' or val_needs_download == False


def download_squad():
   """Download SQuAD validation dataset"""
   print("\n" + "="*60)
   print("SQuAD Dataset")
   print("="*60)

   try:
       from datasets import load_dataset
   except ImportError:
       print("ERROR: 'datasets' library not found")
       print("Install with: pip install datasets")
       return False

   # Create directory
   squad_dir = Path("./data/squad")
   squad_dir.mkdir(parents=True, exist_ok=True)

   # Check what exists
   val_file = squad_dir / "dev-v2.0.json"
   train_file = squad_dir / "train-v2.0.json"
   indices_file = squad_dir / "indices_squad_3.npy"

   print("\nChecking existing files:")

   # Check validation file integrity
   val_status = verify_file_integrity(val_file)
   if val_status == 'valid':
       size_mb = val_file.stat().st_size / (1024 * 1024)
       print(f"  ✓ Valid: {val_file} ({size_mb:.1f} MB)")
       val_needs_download = False
   elif val_status == 'missing':
       print(f"  ✗ Missing: {val_file}")
       val_needs_download = True
   else:  # corrupted or empty
       print(f"  ⚠ {val_status.capitalize()}: {val_file} - will re-download")
       val_needs_download = True

   # Check training file integrity (if enabled)
   if DOWNLOAD_TRAIN_SETS:
       train_status = verify_file_integrity(train_file)
       if train_status == 'valid':
           size_mb = train_file.stat().st_size / (1024 * 1024)
           print(f"  ✓ Valid: {train_file} ({size_mb:.1f} MB)")
           train_needs_download = False
       elif train_status == 'missing':
           print(f"  ✗ Missing: {train_file}")
           train_needs_download = True
       else:
           print(f"  ⚠ {train_status.capitalize()}: {train_file} - will re-download")
           train_needs_download = True
   else:
       print(f"  ⊘ Skipped: {train_file} (DOWNLOAD_TRAIN_SETS = False)")
       train_needs_download = False

   # Check indices file
   indices_status = verify_file_integrity(indices_file)
   if indices_status == 'valid':
       size_mb = indices_file.stat().st_size / (1024 * 1024)
       print(f"  ✓ Valid: {indices_file} ({size_mb:.1f} MB)")
   elif indices_status == 'missing':
       print(f"  ✗ Missing: {indices_file}")
   else:
       print(f"  ⚠ {indices_status.capitalize()}: {indices_file}")

   # Download validation if needed
   if val_needs_download:
       if val_status in ['empty', 'corrupted']:
           print(f"\n→ Deleting corrupted validation file...")
           delete_file_if_exists(val_file)
       print(f"→ Downloading SQuAD validation split...")
       ds = load_dataset('squad', split='validation')
       print(f"  Writing {len(ds)} samples to {val_file}...")

       # Convert dataset to SQuAD format
       squad_format = {
           "version": "2.0",
           "data": []
       }

       for example in ds:
           article = {
               "title": example.get('title', 'unknown'),
               "paragraphs": [{
                   "qas": [{
                       "question": example['question'],
                       "id": example['id'],
                       "answers": [{"text": ans, "answer_start": example['answers']['answer_start'][i]}
                                  for i, ans in enumerate(example['answers']['text'])],
                       "is_impossible": False
                   }],
                   "context": example['context']
               }]
           }
           squad_format["data"].append(article)

       with open(val_file, 'w') as f:
           json.dump(squad_format, f, indent=2)
       print(f"  ✓ Saved: {val_file}")
   else:
       print(f"\n→ Validation set valid, skipping download")

   # Download training if needed and enabled
   if DOWNLOAD_TRAIN_SETS and train_needs_download:
       if train_status in ['empty', 'corrupted']:
           print(f"\n→ Deleting corrupted training file...")
           delete_file_if_exists(train_file)
       print(f"→ Downloading SQuAD training split...")
       print("  (This is used for --nuc > 0 unrelated contexts)")
       ds = load_dataset('squad', split='train')
       print(f"  Writing {len(ds)} samples to {train_file}...")

       # Convert dataset to SQuAD format
       squad_format = {
           "version": "2.0",
           "data": []
       }

       for example in ds:
           article = {
               "title": example.get('title', 'unknown'),
               "paragraphs": [{
                   "qas": [{
                       "question": example['question'],
                       "id": example['id'],
                       "answers": [{"text": ans, "answer_start": example['answers']['answer_start'][i]}
                                  for i, ans in enumerate(example['answers']['text'])],
                       "is_impossible": False
                   }],
                   "context": example['context']
               }]
           }
           squad_format["data"].append(article)

       with open(train_file, 'w') as f:
           json.dump(squad_format, f, indent=2)
       print(f"  ✓ Saved: {train_file}")
   elif DOWNLOAD_TRAIN_SETS and not train_needs_download:
       print(f"\n→ Training set valid, skipping download")

   # Note about indices
   if indices_status != 'valid':
       print(f"\nℹ️  WARNING: indices_squad_3.npy not found or corrupted!")
       print(f"   This file should be downloaded from:")
       print(f"   https://huggingface.co/datasets/YuWangX/KnowledgeRetention")
       print(f"   Place it as: data/squad/indices_squad_3.npy")
   else:
       print(f"\n✓ Indices file valid")

   return val_status == 'valid' or val_needs_download == False


if __name__ == "__main__":
   print("="*60)
   print("M+ Paper - Dataset Downloader (Smart Mode)")
   print("="*60)
   print("Configuration:")
   print(f"  DOWNLOAD_TRAIN_SETS = {DOWNLOAD_TRAIN_SETS}")
   print("\nThis script checks what's already downloaded and only")
   print("downloads what's missing or corrupted.\n")

   # Check if datasets library is installed
   try:
       import datasets
       print(f"✓ 'datasets' library found: {datasets.__version__}\n")
   except ImportError:
       print("⚠ 'datasets' library not installed")
       print("Installing: pip install datasets")
       os.system("pip install datasets")

   # Download datasets
   nq_ok = download_naturalqa()
   squad_ok = download_squad()

   print("\n" + "="*60)
   print("Summary")
   print("="*60)

   if nq_ok and squad_ok:
       print("✓ Dataset check/download complete!")

       # Check what we have
       nq_dir = Path("./data/nq")
       squad_dir = Path("./data/squad")

       nq_val_status = verify_file_integrity(nq_dir / "v1.0-simplified_nq-dev-all.jsonl")
       nq_train_status = verify_file_integrity(nq_dir / "v1.0-simplified_simplified-nq-train.jsonl")
       nq_indices_status = verify_file_integrity(nq_dir / "indices_nq_4.npy")

       squad_val_status = verify_file_integrity(squad_dir / "dev-v2.0.json")
       squad_train_status = verify_file_integrity(squad_dir / "train-v2.0.json")
       squad_indices_status = verify_file_integrity(squad_dir / "indices_squad_3.npy")

       print("\n✓ Ready to use:")
       if nq_val_status == 'valid':
           print("  - Natural Questions validation (--nuc 0)")
       if nq_train_status == 'valid':
           print("  - Natural Questions training (--nuc > 0)")
       if squad_val_status == 'valid':
           print("  - SQuAD validation (--nuc 0)")
       if squad_train_status == 'valid':
           print("  - SQuAD training (--nuc > 0)")

       print("\n⚠ Missing or incomplete:")
       if nq_val_status != 'valid':
           print(f"  - Natural Questions validation ({nq_val_status})")
       if nq_train_status != 'valid' and DOWNLOAD_TRAIN_SETS:
           print(f"  - Natural Questions training ({nq_train_status})")
       if nq_indices_status != 'valid':
           print(f"  - Natural Questions indices ({nq_indices_status})")
       if squad_val_status != 'valid':
           print(f"  - SQuAD validation ({squad_val_status})")
       if squad_train_status != 'valid' and DOWNLOAD_TRAIN_SETS:
           print(f"  - SQuAD training ({squad_train_status})")
       if squad_indices_status != 'valid':
           print(f"  - SQuAD indices ({squad_indices_status})")

       print("\nNext steps:")
       print("1. For validation only (--nuc 0):")
       print("   python test_qa_memory.py --model YuWangX/mplus-8b --datasets naturalqa squad --num_samples 10 --nuc 0")
       print("\n2. For full test with distractors (--nuc 10):")
       if not DOWNLOAD_TRAIN_SETS:
           print("   - Set DOWNLOAD_TRAIN_SETS = True and re-run this script")
           print("   - Download indices files from: https://huggingface.co/datasets/YuWangX/KnowledgeRetention")
       else:
           if nq_train_status == 'valid' and squad_train_status == 'valid' and nq_indices_status == 'valid' and squad_indices_status == 'valid':
               print("   python test_qa_memory.py --model YuWangX/mplus-8b --datasets naturalqa squad --num_samples 100 --nuc 10")
           else:
               print("   - Download missing indices files from: https://huggingface.co/datasets/YuWangX/KnowledgeRetention")
               print("   - Then run: python test_qa_memory.py --model YuWangX/mplus-8b --datasets naturalqa squad --num_samples 100 --nuc 10")
       print("\n3. For LongBench (no download needed):")
       print("   python longbench_pred.py --model mplus-8b --path YuWangX/mplus-8b --dataset hotpotqa --max_length 16384")
   else:
       print("✗ Some datasets failed to download")
       sys.exit(1)


