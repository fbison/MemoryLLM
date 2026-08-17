# Daily Work Log

A high-level, goal-oriented summary of work completed, key discoveries, and progress made in this project. Entries are ordered in **reverse-chronological order** (latest session at the top).

---

## 📅 August 17, 2026 — Master Test Reproduction Matrix & Paper Benchmark Taxonomy

### 🎯 Objective
Catalog every experiment, ablation study, and efficiency benchmark in the M+ paper (*arXiv:2502.00592v2*), verify sample filtering alignment (including GPT-4o-mini answerability subsets) for executed runs, and map paper assets to repo execution scripts.

### 💡 Key Discoveries
* **Filter Alignment Verification:** Confirmed that the executed SQuAD and NaturalQA runs strictly used the author-provided pre-filtered indices (`indices_squad_3.npy` and `indices_nq_4.npy` from `YuWangX/KnowledgeRetention`), which inherently enforce both the short-answer length cutoff ($\le 3-4$ tokens) and the GPT-4o-mini answerability filter described in Section 5.3.
* **Paper Benchmark Coverage:** Mapped all 13 paper experiments across 4 readiness tiers (Executed, Actionable, Ablations, External Datasets).

### 🚀 Deliverables & Actions
* **Created Master Matrix:** Added [docs/TEST_REPRODUCTION_MATRIX.md](TEST_REPRODUCTION_MATRIX.md) containing the full cross-referenced comparison table linking paper figures, tables, code scripts, and execution statuses.
* **Updated Documentation Sitemap:** Indexed the new matrix in [docs/README.md](README.md).

---

## 📅 August 16, 2026 — Full 100-Sample SQuAD & NaturalQA Evaluation & Delta Analysis

### 🎯 Objective
Complete the full 100-sample benchmark evaluation for SQuAD, synthesize complete 100-sample results for both SQuAD and NaturalQA, document the numerical performance delta relative to paper claims (*arXiv:2502.00592v2*), and analyze potential root causes for the gap.

### 📊 Benchmark Results Summary

| Benchmark Metric | Local Execution (SDPA Fallback) | Paper Baseline (M+ 8B) | Measured Delta (Local vs. Paper) |
| :--- | :---: | :---: | :---: |
| **SQuAD Step 0** (No Distractors) | **37.00%** | **~65.00%** | **-28.00%** |
| **SQuAD Step 10** (10 Distractors) | **28.00%** | **~62.00%** | **-34.00%** |
| **NaturalQA Step 0** (No Distractors) | **47.00%** | **~75.00%** | **-28.00%** |
| **NaturalQA Step 10** (10 Distractors) | **33.00%** | **~70.00%** | **-37.00%** |

### 💡 Key Discoveries
1. **Memory Retention Behavior Confirmed:** On both datasets, accuracy remains relatively stable across 10 inserted distractor contexts (~4,000 tokens of distraction). NaturalQA stays between 31%–47%, and SQuAD stays between 28%–40%. This confirms M+'s design goal of retaining long-term memory across distractor noise without catastrophic loss.
2. **Execution Timing:** 
   * SQuAD 100 samples completed in **42 minutes 54 seconds** (~25.7s/sample).
   * NaturalQA 100 samples completed in **43 minutes 05 seconds** (~25.8s/sample).
3. **Performance Delta Analysis (Open Research Hypotheses):**
   * **Empirical Observation:** The local environment achieves ~37% (SQuAD) and ~47% (NaturalQA) initial recall, representing a ~28%–37% gap compared to paper plots.
   * **Open Hypotheses for the Gap:**
     1. *Attention Backend & Precision Differences:* Local runs use PyTorch's native SDPA fallback (`LlamaSdpaAttention`), whereas the paper used `FlashAttention-2` CUDA C++ kernels.
     2. *Prompt Formatting & Normalization:* Differences in exact match string normalization (e.g. whitespace handling, BOS/EOS token decoding).
     3. *Sample Selection:* Discrepancy between the paper's GPT-4o-mini filtered 100-sample subset and `indices_squad_3.npy` / `indices_nq_4.npy`.

### 🚀 Deliverables & Actions
* **Complete Output Data:** Preserved complete 100-sample result files for [SQuAD](file:///C:/usp/MemoryLLM/results/squad/mplus-8b/results_samples_100_nuc_10_begin.json) and [NaturalQA](file:///C:/usp/MemoryLLM/results/naturalqa/mplus-8b/results_samples_100_nuc_10_begin.json).
* **Documented Delta:** Added numerical delta tracking table to daily logs for ongoing benchmark comparison.
* **Upstream Branch Integrity Verification:** Conducted a comprehensive line-by-line git diff between `upstream/main` and `upstream/mplus`. Confirmed that the upstream `mplus` branch is solely dedicated to pre-training code (`train/` directory). Root-level model definitions (`modeling_mplus.py`), benchmark evaluation scripts (`test_qa_memory.py`), and tokenizer logic are 100% identical between branches. Running evaluation benchmarks on our current branch (`main`) is fully valid and evaluates the genuine `YuWangX/mplus-8b` model.

## Possible reasons to the delta

  There are several plausible hypotheses for this ~28%–37% performance delta that would require isolated experiments to verify:
  1. Attention Kernel & Scaling Differences: FlashAttention-2 kernels compute attention differently than PyTorch's native scaled_dot_product_attention (SDPA). In modeling_mplus.py, SDPA returns None for selector retriever weights, whereas FlashAttention-2 computes custom kernel paths. 
  2. String Normalization & Decoding: Differences in text post-processing, whitespace stripping, lowercasing, or EOS token handling between calculate_exact_hit_accuracy and the authors' internal evaluation scripts.
  3. Sample Subset Differences: The paper describes filtering out ambiguous examples that gpt-4o-mini failed to answer before taking the first 100 samples. The pre-filtered indices (indices_squad_3.npy / indices_nq_4.npy) might represent a slightly different subset than the paper's final 100 samples.   <--- PROBABLY THIS ONE
---

## 📅 August 9, 2026 — 100-Sample Benchmark Evaluation & Interruption Analysis

### 🎯 Objective
Evaluate the M+ model (`YuWangX/mplus-8b`) on the full 100-sample benchmark for NaturalQA and SQuAD, measure execution timing, and analyze performance retention curves.

'''
python test_qa_memory.py --model YuWangX/mplus-8b --datasets squad --num_samples 100 --nuc 10 2>&1 | tee logs/squad_test_$(date +%Y%m%d_%H%M%S).log
'''
### 💡 Key Discoveries
* **NaturalQA Benchmark Completion:** NaturalQA successfully evaluated all 100 samples across 10 distractor steps in **43 minutes and 05 seconds** (~25.8 seconds per question).
  * **Initial Recall (Step 0):** **47.00%**
  * **Retention under Distractors (Step 10):** **33.00%** (stabilized around ~33%–36% from Step 5 to Step 10).
* **SQuAD Interruption Cause:** SQuAD evaluation stopped at sample 3 (`test_samples: 3`) because the process was prematurely interrupted/terminated 1 minute 15 seconds into the SQuAD loop.
* **Incremental Saving Verification:** Verified that incremental progress saving worked as intended: samples 0 through 2 of SQuAD were safely written to `results/squad/mplus-8b/results_samples_100_nuc_10_begin.json` despite the sudden interruption.

### 🚀 Deliverables & Actions
* **Execution Time Profile:** Calculated full evaluation duration (~25.8s per sample, total estimated runtime for both 100-sample datasets: **~1 hour 26 minutes**).
* **Variance Reduction:** Confirmed that evaluating 100 samples smoothed out the noisy step-by-step variance observed in 10-sample runs.

---

## 📅 July 25, 2026 — Benchmark Execution, Metrics Storage & Caching Fix

### 🎯 Objective
Run the official M+ Knowledge Retention benchmarks on remote lab servers, preserve accuracy metrics directly in output files, fix cached file evaluation bypasses, and ensure evaluation reliability.

### 💡 Key Discoveries
* **Evaluation Caching Bypass:** Identified why running `--num_samples 100` finished instantly and only output 10 samples: `test_qa_memory.py` previously checked `if os.path.exists(filename)` without checking sample counts or including `num_samples` in the filename. Because a 10-sample JSON file already existed from an earlier test run, the script skipped evaluation completely and re-read the 10-sample file.
* **Answer Length Design:** Verified that benchmark ground-truth answers are intentionally short ($\le 3–4$ tokens). The 10-token generation limit (`max_new_tokens=10`) is designed to capture these short factual answers efficiently without wasting compute time.
* **Benchmark Scale:** Confirmed that reproducing the paper's exact accuracy curves requires evaluating **100 samples** (`--num_samples 100`) to eliminate variance.

### 🚀 Deliverables & Actions
* **Sample Count in Filename & Smart Caching:** Updated `test_qa_memory.py` to save files with sample counts (e.g., `results_samples_100_nuc_10_begin.json`) and verify that cached files contain at least `opt.num_samples` before skipping evaluation.
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
