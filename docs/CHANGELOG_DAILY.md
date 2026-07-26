# Daily Work Log

A high-level, goal-oriented summary of work completed, key discoveries, and progress made in this project. Entries are ordered in **reverse-chronological order** (latest session at the top).

---

## 📅 July 25, 2026 — Benchmark Execution, Metrics Storage & Lab Safety

### 🎯 Objective
Run the official M+ Knowledge Retention benchmarks on remote lab servers, preserve accuracy metrics directly in output files, and ensure evaluation reliability.

### 💡 Key Discoveries
* **Answer Length Design:** Verified that benchmark ground-truth answers are intentionally short ($\le 3–4$ tokens). The 10-token generation limit (`max_new_tokens=10`) is designed to capture these short factual answers efficiently without wasting compute time.
* **Benchmark Scale:** Confirmed that reproducing the paper's exact accuracy curves requires evaluating **100 samples** (`--num_samples 100`) to eliminate variance.
* **Metrics Persistence:** Identified that standard result JSONs only stored raw prediction text, forcing accuracy to be read from console logs.

### 🚀 Deliverables & Actions
* **Embedded Accuracy Metrics:** Updated `test_qa_memory.py` so that step-by-step Exact Hit Accuracies are automatically calculated and stored inside the `"metrics"` header of each JSON result file.
* **Incremental Saving & Formatting:** Updated `test_qa_memory.py` to format output JSONs with `indent=4` and **save progress after every single question**. Interruptions will no longer lose finished work.
* **SSH & `tmux` Workflow:** Configured a persistent `tmux` session protocol so long evaluation runs continue executing on the lab server even if SSH connections drop or laptops sleep.

---

## 📅 July 9, 2026 — Attention Fallback Compatibility

### 🎯 Objective
Enable the M+ model (`YuWangX/mplus-8b`) to run evaluation tests without requiring custom `flash-attn` CUDA packages.

### 💡 Key Discoveries
* Discovered a output variable mismatch in `modeling_mplus.py`: the model's decoder layers expected 5 output variables, but standard PyTorch attention fallbacks (SDPA/Eager mode) only returned 3 or 4 variables, causing crashes on systems without FlashAttention-2.

### 🚀 Deliverables & Actions
* Modified `modeling_mplus.py` so both SDPA and Eager attention modes return the expected 5-variable structure, enabling M+ to run smoothly on standard PyTorch environments.

---

## 📅 July 7, 2026 — Dataset Procurement & Resumable Downloads

### 🎯 Objective
Download and prepare the pre-filtered NaturalQA and SQuAD benchmark datasets required for evaluation.

### 💡 Key Discoveries
* Identified that large dataset files (over 16 GB) were failing due to network interruptions during bulk downloads.

### 🚀 Deliverables & Actions
* Rewrote `download_datasets.py` to stream pre-formatted benchmark datasets directly from Hugging Face with HTTP Range resumption support, ensuring interrupted downloads resume automatically without losing progress.

---

## 📅 June 28, 2026 — Initial Environment Setup & Model Compatibility

### 🎯 Objective
Set up the lab Python environment and resolve initial startup errors when running `test_qa_memory.py`.

### 💡 Key Discoveries
* Found that Llama-3-based models like `mplus-8b` use Byte-Pair Encoding (BPE) tokenizers (`AutoTokenizer`), which crashed when loaded through legacy `LlamaTokenizer` classes.

### 🚀 Deliverables & Actions
* Standardized environment dependencies (PyTorch 2.5.1 and Transformers 4.48.2).
* Updated `test_qa_memory.py` to use `AutoTokenizer` and safely handle directory creation for test outputs.
