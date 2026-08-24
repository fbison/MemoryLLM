#!/bin/bash
# 160k Knowledge Retention Evaluation Script for M+ (YuWangX/mplus-8b)
set -e

mkdir -p logs results

LOG_FILE="logs/eval_160k_$(date +%Y%m%d_%H%M%S).log"
echo "========================================================================"
echo "Starting 160k Knowledge Retention Benchmark (SQuAD & NaturalQA)"
echo "Logging output to: $LOG_FILE"
echo "========================================================================"

python test_qa_memory.py \
  --model YuWangX/mplus-8b \
  --datasets naturalqa \
  --num_samples 100 \
  --nuc 320 \
  --eval_interval 20 \
  2>&1 | tee "$LOG_FILE"
