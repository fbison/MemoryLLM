#!/usr/bin/env python3
"""
Download and prepare datasets for M+ paper tests.
Saves to ./data/ directory for use with test_qa_memory.py
"""

import os
import json
import sys
from pathlib import Path

def download_naturalqa():
   """Download Natural Questions validation dataset"""
   print("\n" + "="*60)
   print("Downloading Natural Questions validation dataset...")
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
   print(f"✓ Created directory: {nq_dir}")

   # Download NQ validation split
   print("Loading Natural Questions validation split from HuggingFace...")
   ds = load_dataset('natural_questions', split='validation')

   # Save validation split as JSONL (following the repo's expected format)
   output_file = nq_dir / "v1.0-simplified_nq-dev-all.jsonl"
   print(f"Writing {len(ds)} samples to {output_file}...")

   with open(output_file, 'w') as f:
       for example in ds:
           f.write(json.dumps(example) + '\n')

   print(f"✓ Saved: {output_file}")

   print("\nℹ️  NOTE: Training set not downloaded. If you need to use --nuc > 0 (unrelated contexts),")
   print("   you'll need to manually download the training split from:")
   print("   https://huggingface.co/datasets/YuWangX/KnowledgeRetention")
   print("   And place it as: data/nq/v1.0-simplified_simplified-nq-train.jsonl")

   return True


def download_squad():
   """Download SQuAD validation dataset"""
   print("\n" + "="*60)
   print("Downloading SQuAD validation dataset...")
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
   print(f"✓ Created directory: {squad_dir}")

   # Download SQuAD validation split
   print("Loading SQuAD validation split from HuggingFace...")
   ds = load_dataset('squad', split='validation')

   # Convert to SQuAD format (JSON)
   output_file = squad_dir / "dev-v2.0.json"
   print(f"Writing {len(ds)} samples to {output_file}...")

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

   with open(output_file, 'w') as f:
       json.dump(squad_format, f, indent=2)

   print(f"✓ Saved: {output_file}")

   print("\nℹ️  NOTE: Training set not downloaded. If you need to use --nuc > 0 (unrelated contexts),")
   print("   you'll need to manually download the training split from:")
   print("   https://huggingface.co/datasets/YuWangX/KnowledgeRetention")
   print("   And place it as: data/squad/train-v2.0.json")

   return True


if __name__ == "__main__":
   print("M+ Paper - Dataset Downloader")
   print("This script downloads datasets needed for test_qa_memory.py\n")

   # Check if datasets library is installed
   try:
       import datasets
       print(f"✓ 'datasets' library found: {datasets.__version__}")
   except ImportError:
       print("⚠ 'datasets' library not installed")
       print("Installing: pip install datasets")
       os.system("pip install datasets")

   # Download datasets
   nq_ok = download_naturalqa()
   squad_ok = download_squad()

   print("\n" + "="*60)
   if nq_ok and squad_ok:
       print("✓ All datasets downloaded successfully!")
       print("\nNext steps:")
       print("1. Run: python test_qa_memory.py --model YuWangX/mplus-8b --datasets naturalqa squad --num_samples 100")
       print("2. Results will be saved to results/ directory")
   else:
       print("✗ Some datasets failed to download")
       sys.exit(1)


