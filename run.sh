#!/bin/bash
set -e

echo "=============================================="
echo "🚀 Priori Judgment Evaluation (CARE Data)"
echo "=============================================="

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate priori_care

# 完整评估
python run_eval.py \
    --data_root data_care/eval \
    --model_name NousResearch/Meta-Llama-3-8B-Instruct \
    --output_dir results

echo ""
echo "✅ Evaluation complete!"
echo "Results: results/results.json"