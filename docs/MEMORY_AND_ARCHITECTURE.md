# Memory & Architecture Notes

This document details the architectural concepts of **MemoryLLM** and **M+**, how long-term memory (LTM) is pooled and injected, and the key source code files that govern execution.

---

## 1. Architectural Overview: MemoryLLM vs. M+

Both models enable Large Language Models to store, update, and retrieve long-term context beyond traditional attention context windows by embedding a persistent **Long-Term Memory Pool** directly into the Transformer architecture.

| Feature | MemoryLLM | M+ (`YuWangX/mplus-8b`) |
| :--- | :--- | :--- |
| **Base Model** | Llama-2-7B | Llama-3-8B |
| **Memory Capacity** | Single memory pool block (~256 tokens per layer) | Multi-block / Extended Long-Term Memory (LTM) |
| **Retriever Mechanism** | Basic key-query matching | Memory Selector & Encoder-Decoder Retriever Weights |
| **Memory Weights Location** | GPU / CPU | Movable to CPU/Numpy (`.put_ltm_to_numpy()`) |

---

## 2. Long-Term Memory (LTM) Mechanisms

### Memory Pool Parameters & Injection
* **Memory Pool Size:** `1.3422 B` parameters dedicated to layer-wise long-term memory blocks.
* **Token Budget per Memory Block (`num_tokens`):** `256` tokens.
* **Memory Injection (`model.inject_memory`):**
  1. Input context text is passed through the model.
  2. The hidden states of the context are extracted and compressed into fixed-size memory representations.
  3. The model updates its layer-wise memory pool (`delta_memory`).
  4. During subsequent generation, self-attention layers query both the immediate context and the long-term memory pool.

### Retriever & Selector Weights
In `M+`, a memory selector calculates relevance weights (`retriever_weights`) between incoming query tokens and stored long-term memory keys using sigmoid dot-product attention:
$$\text{retriever\_weights} = \sigma(Q_{\text{retriever}} \cdot K_{\text{memory}}^T)$$

These weights dynamically scale or bias the attention logits over stored memory tokens, allowing the model to selectively retrieve relevant factual contexts while ignoring distractor contexts.

---

## 3. Codebase File Map & Module Responsibilities

```
MemoryLLM/
├── modeling_mplus.py          # Primary model architecture for M+ (subclasses LlamaForCausalLM)
├── modeling_memoryllm.py      # Base architecture for standard MemoryLLM models
├── test_qa_memory.py          # Main evaluation script for Knowledge Retention QA benchmark
├── download_datasets.py       # Resumable downloader for SQuAD and NaturalQA test files
├── dataset/
│   ├── nq.py                  # PyTorch Dataset class for NaturalQA (NQ)
│   └── squad.py               # PyTorch Dataset class for SQuAD 2.0
├── data/                      # Local data directory for downloaded jsonl/json benchmark files
└── results/                   # Evaluation results JSON outputs (organized by dataset/model)
```

### Critical Implementation Details in `modeling_mplus.py`
* **Model Class (`MPlus`):** Main Causal LM wrapper class.
* **Decoder Layer (`LlamaDecoderLayer`):** Coordinates self-attention, feed-forward layers, and memory retrieval. Line 931 performs the self-attention output unpacking:
  ```python
  hidden_states, self_attn_weights, present_key_value, retriever_weights, encoder_retriever_weights = self.self_attn(...)
  ```
* **Attention Classes (`LLAMA_ATTENTION_CLASSES`):**
  * `LlamaFlashAttention2`: High-speed CUDA kernel attention. Returns 5 values.
  * `LlamaSdpaAttention`: PyTorch native SDPA attention fallback. Modified to return 5 values (`attn_output, None, past_key_value, None, None`).
  * `LlamaAttention`: Eager PyTorch attention fallback. Modified to return 5 values (`attn_output, attn_weights, past_key_value, retriever_weights, None`).
