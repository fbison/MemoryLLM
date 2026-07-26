# Paper & Benchmark Specifications

This document summarizes the experimental protocol and benchmark specifications for evaluating **M+ (`YuWangX/mplus-8b`)** based on the official paper (*arXiv:2502.00592v2*).

---

## 1. Paper Reference

* **Title:** *M+: Extending Memory Capacity of Language Models via Modular Long-Term Memory Pooling*
* **Authors:** Yu Wang et al.
* **Repository / Paper Reference:** `arXiv-2502.00592v2`
* **Hugging Face Model:** `YuWangX/mplus-8b`
* **Hugging Face Dataset:** `YuWangX/KnowledgeRetention`

---

## 2. Knowledge Retention QA Benchmark Protocol

The Knowledge Retention evaluation tests a model's ability to retain and retrieve factual information stored in its long-term memory while being subjected to distractors (unrelated context blocks).

### Dataset Format
Each dataset sample is formatted as a triplet:
$$\text{Sample} = (\text{context}, \text{question}, \text{answer})$$

1. **Context Injection:** The target `context` is injected into the model's memory (`model.inject_memory(..., update_memory=True)`).
2. **Distractor Insertion (`nuc`):** Unrelated distractor contexts (`num_unrelated_contexts`, default `nuc=10`) are sequentially injected into memory to test retrieval retention under increasing memory load.
3. **Question Answering:** At each step $i \in [0, \text{nuc}]$, the model is queried with `question` to evaluate whether it can still recall `answer`.

---

## 3. Answer Token Length Filtering vs. Generation Budget

### Ground-Truth Answer Filtering ($\le 3 - 4$ Tokens)
In Section 5.2.1 of the paper (*5_experiments.tex*, line 98):
> *"To evaluate the ability of M+ to recall long-term knowledge, we follow the experimental setup in MemoryLLM on datasets SQuAD and NaturalQA... Consistent with MemoryLLM, we extract samples with **answer lengths of three tokens or fewer for SQuAD** and **four tokens or fewer for NaturalQA**."*

* **SQuAD Filter Index (`indices_squad_3.npy`):** Retains questions where ground-truth answer length $\le 3$ tokens.
* **NaturalQA Filter Index (`indices_nq_4.npy`):** Retains questions where ground-truth answer length $\le 4$ tokens.

### Generation Token Budget (`max_new_tokens=10`)
In `test_qa_memory.py` (lines 293 and 309), text generation is configured with:
```python
max_new_tokens=10
```

#### Why are the two values different?
* **Filtering ($\le 3-4$ tokens):** Defines the length of the *ground-truth answer string* (e.g. `"France"` = 1 token).
* **Generation Budget (`10` tokens):** Defines the maximum number of tokens the model is allowed to output during `generate()`. Because LLMs often generate short leading preambles (e.g., `"Normandy is in France."`), a 10-token budget ensures the model is not cut off prematurely before outputting the answer word.

---

## 4. Evaluation Metric: Exact Hit Accuracy

The accuracy is calculated using **Exact Hit / Exact Match Accuracy** ([test_qa_memory.py](file:///C:/usp/MemoryLLM/test_qa_memory.py#L335)):

```python
def calculate_exact_hit_accuracy(predictions, targets):
    count = 0
    hit = 0
    for i in range(len(predictions)):
        if targets[i].replace("</s>", "").strip() in predictions[i]:
            hit += 1
        count += 1
    return hit / count
```

A sample is counted as a **Hit (1.0)** if the cleaned ground-truth target string (e.g., `"France"`) appears as a substring inside the generated prediction string.

---

## 5. Benchmark Sample Size & Expected Paper Results

* **Benchmark Sample Size:** **100 samples** (`--num_samples 100`).
* **Expected Results (M+ 8B at `nuc=10`):**
  * **SQuAD:** Accuracy remains stable around **60% – 65%** across 0 to 10 distractor steps.
  * **NaturalQA:** Accuracy remains stable around **68% – 75%** across 0 to 10 distractor steps.
