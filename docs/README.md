# MemoryLLM & M+ Project Documentation

Welcome to the documentation repository for reproducing and evaluating **MemoryLLM** and **M+ (`YuWangX/mplus-8b`)** knowledge retention benchmarks.

This directory maintains comprehensive, persistent documentation to prevent context degradation and ensure smooth execution, debugging, and experimentation on lab computing environments.

---

## 📚 Documentation Index

1. **[Execution and Lab Guide](EXECUTION_AND_LAB_GUIDE.md)**
   * How to run evaluation tests (`test_qa_memory.py`).
   * Remote SSH & `tmux` workflow instructions to prevent test interruptions.
   * Environment requirements & dependency setup.
   * Registry of all solved bugs, errors, and fixes applied to the codebase.

2. **[Memory & Architecture Notes](MEMORY_AND_ARCHITECTURE.md)**
   * Architectural comparison between MemoryLLM and M+.
   * Long-Term Memory (LTM) pooling, injection, and retriever mechanisms.
   * Key file map and module responsibilities.

3. **[Daily Work Log & Changelog](CHANGELOG_DAILY.md)**
   * Reverse-chronological record of all work, bugfixes, and code changes (newest first).
   * Detailed context on what changes were made and why.

4. **[Paper & Benchmark Specifications](PAPER_AND_BENCHMARK_SPECS.md)**
   * Summary of the M+ paper (*arXiv:2502.00592v2*).
   * Experimental protocol for Knowledge Retention (SQuAD & NaturalQA).
   * Answer token length filtering ($\le 3-4$ tokens) vs. generation budget (`max_new_tokens=10`).
   * Evaluation metrics (Exact Hit Accuracy) and benchmark sample sizes ($N=100$).

5. **[Paper Benchmark & Test Reproduction Matrix](TEST_REPRODUCTION_MATRIX.md)**
   * Inventory of all 13 paper experiments, figures, tables, and execution statuses.
   * Verification of GPT-4o-mini answerability filter and subset indices.
   * Direct script mapping and readiness for remaining tests (LongBench, VRAM, Perplexity, LongBook).

