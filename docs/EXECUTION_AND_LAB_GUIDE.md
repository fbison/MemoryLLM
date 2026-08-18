# Execution & Lab Guide

This document contains step-by-step instructions for running the evaluation scripts on remote lab infrastructure, managing long-running jobs over SSH with `tmux`, and a reference registry of all solved bugs and environment fixes.

---

## 1. Quickstart Execution Guide

To reproduce the Knowledge Retention QA benchmark for M+ (`YuWangX/mplus-8b`) on **NaturalQA** and **SQuAD**:

```bash
# 1. Activate environment
conda activate memoryllm

### Quickstart Commands

#### A. Standard Short-Horizon Evaluation (`nuc=10` $\approx$ 5k tokens)
```bash
python test_qa_memory.py \
  --model YuWangX/mplus-8b \
  --datasets naturalqa squad \
  --num_samples 100 \
  --nuc 10 \
  2>&1 | tee logs/qa_test_$(date +%Y%m%d_%H%M%S).log
```

#### B. Full 160k-Token Retention Curve (Matching Paper Figures 3 & 4: 0k to 160k at 10k intervals)

**Option 1: Using the helper script (Recommended to avoid copy-paste line break issues):**
```bash
bash run_eval_160k.sh
```

**Option 2: As a single unbroken command:**
```bash
python test_qa_memory.py --model YuWangX/mplus-8b --datasets squad naturalqa --num_samples 100 --nuc 320 --eval_interval 20 2>&1 | tee logs/eval_160k_$(date +%Y%m%d_%H%M%S).log
```

### Command Parameters
* `--model`: Model path or Hugging Face hub ID (`YuWangX/mplus-8b`).
* `--datasets`: Target datasets (`naturalqa`, `squad`).
* `--num_samples`: Number of test evaluation samples (default: `100` to match paper benchmark).
* `--nuc`: Number of Unrelated Contexts / distractor context chunks inserted into long-term memory (`nuc=320` $\approx$ 160,000 tokens).
* `--eval_interval`: Evaluation interval for text generation (e.g. `20` = evaluate every 20 chunks $\approx$ 10k tokens).
* `--related_position`: Position of the target context in the memory sequence (`begin`, `end`, or `random`; default: `begin`).

---

## 2. SSH & `tmux` Workflow for Remote Lab Server

Evaluating 100 samples with 10 distractor steps per sample can take considerable time. To ensure that an SSH connection drop, laptop sleep, or Wi-Fi disconnection does **not** abort your evaluation, use `tmux`.

### Running in a Persistent `tmux` Session
1. **Connect via SSH and start a named session:**
   ```bash
   tmux new -s qa_eval
   ```
2. **Activate environment and start evaluation:**
   ```bash
   conda activate memoryllm
   python test_qa_memory.py --model YuWangX/mplus-8b --datasets naturalqa squad --num_samples 100 --nuc 10 2>&1 | tee logs/qa_test_$(date +%Y%m%d_%H%M%S).log
   ```
3. **Detach from the session (leave running in background):**
   * Press **`Ctrl + B`**, release both keys, then press **`D`**.
   * You can now safely close your SSH connection or laptop.

### Managing Active Sessions
* **List running sessions:**
  ```bash
  tmux ls
  ```
* **Reattach to an ongoing evaluation:**
  ```bash
  tmux attach -t qa_eval
  ```
* **Kill a finished session:**
  ```bash
  tmux kill-session -t qa_eval
  ```

---

## 3. Environment & Dependency Setup

* **Python Version:** `3.10`
* **Conda Environment:** `memoryllm`
* **Key Packages:**
  * PyTorch `2.5.1` (`torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1`)
  * Hugging Face Transformers `4.48.2`
  * Accelerate `1.2.0`
  * Datasets `2.18.0`

> **Note on `flash-attn`:** If `flash-attn` is not installed on the GPU host, the model automatically falls back to PyTorch native Scaled Dot-Product Attention (SDPA) or Eager attention.

---

## 4. Solved Issues & Bugfix Registry

### Issue 1: PyTorch / Transformers Version Mismatch
* **Symptom:** `[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2` followed by `NameError: name 'nn' is not defined`.
* **Root Cause:** Older `transformers` version disabled PyTorch backend when PyTorch 2.2 was detected.
* **Fix:** Upgraded environment to PyTorch `2.5.1` and Hugging Face `transformers==4.48.2`.

### Issue 2: SentencePiece / Tokenizer `TypeError`
* **Symptom:** `TypeError: LlamaTokenizer.__init__() missing 1 required positional argument: 'vocab_file'`.
* **Root Cause:** Llama-3-based `mplus-8b` uses a Byte-Pair Encoding (BPE) tokenizer (`tokenizer.json`), but the evaluation script was hardcoded to instantiate `LlamaTokenizer` (which expects SentencePiece `tokenizer.model`).
* **Fix:** Replaced `LlamaTokenizer` with `AutoTokenizer` in `test_qa_memory.py` and passed the pre-instantiated tokenizer directly to dataset constructors.

### Issue 3: Missing Datasets & Download Failures
* **Symptom:** Dataset file not found or corrupted during downloading.
* **Root Cause:** Original script attempted to build dataset from scratch or failed on interrupted connections.
* **Fix:** Rewrote `download_datasets.py` to stream pre-filtered dataset files directly from `YuWangX/KnowledgeRetention` on Hugging Face with HTTP `Range` resumption support.

### Issue 4: Parent Directory Creation Failure
* **Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: 'results/naturalqa'`.
* **Root Cause:** `os.mkdir` failed when parent directories did not exist.
* **Fix:** Replaced `os.mkdir` with `os.makedirs(..., exist_ok=True)` in `test_qa_memory.py`.

### Issue 5: Attention Backend Unpacking Error (CRITICAL)
* **Symptom:** `ValueError: not enough values to unpack (expected 5, got 3)` in `modeling_mplus.py` line 931.
* **Root Cause:** `LlamaDecoderLayer.forward()` expects 5 output values from self-attention (`hidden_states, self_attn_weights, present_key_value, retriever_weights, encoder_retriever_weights`). However, under SDPA fallback (`LlamaSdpaAttention`), only 3 values were returned, and under Eager fallback (`LlamaAttention`), only 4 values were returned.
* **Fix:** Modified `modeling_mplus.py` so that both `LlamaSdpaAttention` and `LlamaAttention` return 5 elements (appending `None` placeholders for missing retriever/encoder weights).

### Issue 6: Unformatted Single-Line JSON & Risk of Lost Progress
* **Symptom:** Output JSON results were saved as a single 260KB string without indentation, and only written at the end of the entire run.
* **Fix:** Updated `test_qa_memory.py` to use `json.dump(..., indent=4)` for formatted multi-line JSONs, and added incremental saving after every evaluated sample.

### Issue 7: Multi-Process DataLoader File Descriptor Exhaustion at High `nuc`
* **Symptom:** `OSError: [Errno 24] Too many open files` in `torch/multiprocessing/reductions.py` when evaluating large distractor counts (`nuc=320`).
* **Root Cause:** When `num_workers > 0`, PyTorch shares all 646 batch tensors via IPC file descriptors across worker queues, exceeding the OS default limit (`1024`).
* **Fix:** Added `--num_workers` parameter defaulting to `0` in `test_qa_memory.py`, executing data loading in the main process and eliminating IPC file descriptor overhead.

