# M+ Paper Benchmark & Test Reproduction Matrix

This document provides a comprehensive inventory and execution status of all experiments, benchmarks, ablation studies, and efficiency profiling tests described in the **M+ paper (*arXiv:2502.00592v2*)**.

---

## 📊 Master Reproduction Matrix

| # | Experiment / Test Name | Paper Section & Target | Status | Samples & Config (Paper vs. Local) | Paper Asset / Figure | Repo Script & Readiness |
| :---: | :--- | :--- | :---: | :--- | :--- | :--- |
| **1** | **SQuAD Knowledge Retention** | Sec 5.3 / Fig 3 | ✅ **Partially Executed** (Short-Range) | **Paper Full Curve:** 100 samples, ans $\le 3$ tokens, GPT-4o-mini filtered, distractors scaled up to **160k tokens** (~320 chunks).<br>**Local Executed:** 100 samples, ans $\le 3$ tokens, **10 distractors** (`nuc=10` $\approx$ **5k tokens**). | [figures/squad.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/squad.pdf) | [test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py)<br>*(Ready as-is for `nuc=10`; needs higher step loop for 160k curve)* |
| **2** | **NaturalQA Knowledge Retention** | Sec 5.3 / App B.1 / Fig 4 | ✅ **Partially Executed** (Short-Range) | **Paper Full Curve:** 100 samples, ans $\le 4$ tokens, GPT-4o-mini filtered, distractors scaled up to **160k tokens** (~320 chunks).<br>**Local Executed:** 100 samples, ans $\le 4$ tokens, **10 distractors** (`nuc=10` $\approx$ **5k tokens**). | [figures/nqa.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/nqa.pdf) | [test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py)<br>*(Ready as-is for `nuc=10`; needs higher step loop for 160k curve)* |
| **3** | **LongBench Benchmark (8k & 16k)** | Sec 5.4 / Table 2 | 🛠️ **Pending (Actionable)** | 6 datasets (`hotpotqa`, `2wikimqa`, `musique`, `qasper`, `multifieldqa_en`, `narrativeqa`) at 8k and 16k tokens; QA-F1 metric. | Table 2 (in text) | [longbench_pred.py](file:///C:/usp/MemoryLLM/longbench_pred.py) & [metrics.py](file:///C:/usp/MemoryLLM/metrics.py)<br>*(Needs adaptation for `MPlus` class)* |
| **4** | **LongBench Stage-by-Stage Ablation** | Sec 5.5.1 / Table 3 | 🛠️ **Pending (Actionable)** | 6 datasets at 8k context comparing Stage 1 (`MemoryLLM-8B`), Stage 2 (`MemoryLLM-8B-Long`), and Stage 3 (`M+ 8B`). | Table 3 (in text) | [longbench_pred.py](file:///C:/usp/MemoryLLM/longbench_pred.py)<br>*(Requires evaluating all 3 checkpoints)* |
| **5** | **LongBook-QA ($\infty$-Bench)** | Sec 5.1 / Fig 1 | 📦 **Pending (External Data)** | 351 `(book, question, answer)` tuples (~192k avg tokens); QA-F1 metric. | [figures/longbook_qa_results.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/longbook_qa_results.pdf) | Needs $\infty$-Bench LongBook loader & script |
| **6** | **LongBook Event QA** | Sec 5.1 / Fig 1 | 📦 **Pending (External Data)** | 5 books, 4k-token chunks, 6-option multiple choice generated via SpaCy NER & GPT-4o; Accuracy metric. | [figures/longbook_qa_results.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/longbook_qa_results.pdf) | Needs author event QA dataset or pipeline |
| **7** | **GPU Memory Allocation Cost** | Sec 5.2 / Table 1 | ⏳ **Pending (Ready)** | Peak GPU memory allocation (MB) during inference across LongBook: M+ (standard: 21.18 GB) vs. M+ (offload: 17.97 GB). | Table 1 (in text) | [inference.py](file:///C:/usp/MemoryLLM/inference.py) / [test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py) (`--put_memory_on_cpu`) |
| **8** | **Slim-Pajama Long-Context Validation Loss** | Sec 5.5.1 / Fig 4 | ⏳ **Pending** | 1,000 held-out examples from Slim-Pajama (32k–64k tokens); validation cross-entropy loss across Stages 1, 2, and 3. | [figures/loss.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/loss.pdf) | Evaluated via standard causal LM loss on long sequences |
| **9** | **Knowledge Retention Stage & Retriever Ablations** | Sec 5.5.1, 5.5.2 / Fig 5, Fig 6 | ⏳ **Pending** | SQuAD & NaturalQA retention comparison across stages and vs. `M+-Attn` (SnapKV-style attention retriever). | [figures/squad_ablation.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/squad_ablation.pdf)<br>[figures/nqa_ablation.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/nqa_ablation.pdf) | [test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py) |
| **10** | **Base Context Window Perplexity** | Sec 5.6.1 | ⏳ **Pending (Ready)** | Perplexity on 1,000 non-overlapping examples from `fineweb-edu` within 2,048 tokens ($\text{PPL} \approx 1.9828$). | Text Section 5.6.1 | Standalone perplexity evaluation script |
| **11** | **Ground-Truth Retrieval Quality (Recall Curve)** | Sec 5.6.2 / Fig 6 | ⏳ **Pending** | Tracking retrieval of the original 256 ground-truth memory vectors as LTM expands to 80k tokens. | [figures/recall_curve_squad.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/recall_curve_squad.pdf) | [test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py) (retriever weight inspection) |
| **12** | **Inference Latency Scaling Analysis** | Sec 5.6.3 / Fig 7 | ⏳ **Pending (Ready)** | End-to-end forward/prediction latency across 16k, 32k, 64k, 128k input sequences on single GPU. | [figures/latency_comparison.pdf](file:///C:/usp/MemoryLLM/arXiv-2502.00592v2/figures/latency_comparison.pdf) | [inference.py](file:///C:/usp/MemoryLLM/inference.py) / latency benchmark script |
| **13** | **FLOPs Profiling** | App E.4 / Table | ⏳ **Pending** | FLOPs for 1-token generation after processing sequences from 2k to 128k using `torch.profiler`. | Table in Appendix E.4 | Standalone `torch.profiler` script |

---

## 🔍 Detailed Analysis of Executed Tests (SQuAD & NaturalQA)

### 1. Verification of the GPT-4o-mini Answerability Filtering

> [!IMPORTANT]
> **Was the GPT-4o-mini answerability filtering used in your local runs?**
> **YES.** Both local runs strictly utilized the paper's pre-filtered subset.

#### How It Works:
* **Paper Specification (*5_experiments.tex*, Section 5.3):**
  > *"Consistent with MemoryLLM, we extract samples with answer lengths of three tokens or fewer for SQuAD and four tokens or fewer for NaturalQA. After filtering out ambiguous examples that gpt-4o-mini fails to answer, we select the first 100 examples from the remaining answerable set to conduct our evaluation."*
* **Codebase Verification:**
  * In [dataset/squad.py](file:///C:/usp/MemoryLLM/dataset/squad.py#L39-L92), the dataset loads `indices_squad_3.npy`:
    ```python
    indices = np.load(os.path.join(os.path.dirname(filename), 'indices_squad_3.npy'))
    ...
    if num is not None:
        indices = indices[:num]
    self.data = [self.data[i] for i in indices]
    ```
  * In [dataset/nq.py](file:///C:/usp/MemoryLLM/dataset/nq.py#L45-L85), the dataset loads `indices_nq_4.npy`:
    ```python
    indices = np.load(os.path.join(os.path.dirname(filename), 'indices_nq_4.npy'))
    ...
    if num is not None:
        indices = indices[:num]
    self.data = [self.data[i] for i in indices]
    ```
  * These index arrays (`indices_squad_3.npy` and `indices_nq_4.npy`) were published directly by the paper authors on Hugging Face (`YuWangX/KnowledgeRetention`) and downloaded via [download_datasets.py](file:///C:/usp/MemoryLLM/download_datasets.py). They contain the exact row indices that passed both the length filter and the GPT-4o-mini answerability filter.

---

### 2. Execution Comparison: NaturalQA vs. SQuAD

| Dimension | SQuAD Execution | NaturalQA Execution |
| :--- | :--- | :--- |
| **Evaluated Samples** | 100 samples (`--num_samples 100`) | 100 samples (`--num_samples 100`) |
| **Distractor Steps** | Step 0 to Step 10 (11 total evaluation points) | Step 0 to Step 10 (11 total evaluation points) |
| **Ground-Truth Filter** | Length $\le 3$ tokens (`indices_squad_3.npy`) | Length $\le 4$ tokens (`indices_nq_4.npy`) |
| **Distractor Contexts** | SQuAD Training set paragraphs (300–512 tokens) | SQuAD Training set paragraphs (300–512 tokens) |
| **Generation Budget** | `max_new_tokens = 10` | `max_new_tokens = 10` |
| **Local Accuracy Range** | **37.00%** (Step 0) $\rightarrow$ **28.00%** (Step 10) | **47.00%** (Step 0) $\rightarrow$ **33.00%** (Step 10) |
| **Paper Accuracy Range** | **~65.00%** (Step 0) $\rightarrow$ **~62.00%** (Step 10) | **~75.00%** (Step 0) $\rightarrow$ **~70.00%** (Step 10) |
| **Measured Delta** | -28.00% (Step 0) to -34.00% (Step 10) | -28.00% (Step 0) to -37.00% (Step 10) |
| **Output Results Path** | [results/squad/mplus-8b/results_samples_100_nuc_10_begin.json](file:///C:/usp/MemoryLLM/results/squad/mplus-8b/results_samples_100_nuc_10_begin.json) | [results/naturalqa/mplus-8b/results_samples_100_nuc_10_begin.json](file:///C:/usp/MemoryLLM/results/naturalqa/mplus-8b/results_samples_100_nuc_10_begin.json) |

#### Nuances and Differences in Local Execution:
1. **Distractor Horizon (Short-Range vs. Full 160k Curve):**
   * **In the Paper (Figures 3 & 4):** Distractors are scaled up to **160,000 tokens** (160k). With each SQuAD distractor paragraph chunk averaging ~500 tokens, 160k tokens corresponds to injecting **~320 distractor chunks** (evaluated at 10k token intervals from 0k to 160k).
   * **In the Local Run (`--nuc 10`):** We injected **10 distractor chunks** ($\sim 5,000$ tokens of distraction total). This corresponds strictly to the first segment of the paper's curve (**0k to ~5k tokens**).
2. **Attention Backend:** The local execution runs under PyTorch native SDPA fallback (`LlamaSdpaAttention`), whereas the paper was run using custom `FlashAttention-2` CUDA kernels.
3. **Exact Hit Matcher:** Accuracy is computed using case-insensitive substring matching in [test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py#L335) (`calculate_exact_hit_accuracy`).

---

## 🚀 Execution Roadmap & Script Availability

### Phase 1: High Priority (Directly Available in Codebase)

#### 1. LongBench Multi-Task Evaluation (Table 2 & Table 3)
* **What it tests:** 6 datasets (`2wikimqa`, `hotpotqa`, `qasper`, `musique`, `multifieldqa_en`, `narrativeqa`) at 8k and 16k input lengths.
* **Code in Repo:** [longbench_pred.py](file:///C:/usp/MemoryLLM/longbench_pred.py) and [metrics.py](file:///C:/usp/MemoryLLM/metrics.py).
* **Adaptation Needed:** Update `longbench_pred.py` argument parsing to accept `YuWangX/mplus-8b` and load `MPlus` in `torch.bfloat16`.

#### 2. Base Model Perplexity within Context Window (Section 5.6.1)
* **What it tests:** Causal language modeling perplexity on 1,000 examples from `fineweb-edu` (snapshot `CC-MAIN-2024-10`) capped at 2,048 tokens.
* **Code in Repo:** Standalone evaluation script utilizing `AutoTokenizer` and `MPlus.from_pretrained()`.

#### 3. GPU Memory Cost & CPU Offloading Verification (Table 1)
* **What it tests:** Measuring peak GPU memory in MB using `torch.cuda.max_memory_allocated()` with standard M+ vs. M+ with CPU memory offloading (`--put_memory_on_cpu`).
* **Code in Repo:** [test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py) (supports `--put_memory_on_cpu`) and [inference.py](file:///C:/usp/MemoryLLM/inference.py).

---

### Phase 2: Medium Priority (Extended Knowledge Retention & Ablations)

#### 4. Extended Distractor Depth & Position Ablations (Figures 3, 5, 6)
* **What it tests:** Retention decay over 20–50 distractor steps and testing context position (`--related_position random` and `--related_position end`).
* **Code in Repo:** [test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py) *(Ready as-is)*.

---

### Phase 3: External Dataset Dependencies

#### 5. LongBook-QA ($\infty$-Bench) and LongBook Event QA (Figure 1)
* **What it tests:** 192k-token document understanding and chronological event reasoning across 5 books.
* **Requirements:** Download the $\infty$-Bench LongBook dataset and Event QA questions generated by GPT-4o.
