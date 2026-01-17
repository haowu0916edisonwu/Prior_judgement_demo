#!/usr/bin/env python3
"""
极速算分脚本 (Fast Re-Evaluation)
不需要加载模型，直接读取现有的 .jsonl 文件，按照论文标准 (分母=6) 重新计算平均分。
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
from src.metrics import Metrics  # 复用你现有的指标计算逻辑

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", help="Directory containing .jsonl predictions")
    parser.add_argument("--output_file", default="final_paper_results.json", help="Output JSON file")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    
    # 论文对齐的评估配置
    # (数据集名称, JSONL文件名, 指标函数, 指标名称, 目标分数)
    tasks = [
        ('nq', 'nq_predictions.jsonl', Metrics.compute_span_em, 'span_em', 0.458),
        ('trivia', 'trivia_predictions.jsonl', Metrics.compute_span_em, 'span_em', 0.704),
        ('webqa', 'webqa_predictions.jsonl', Metrics.compute_span_em, 'span_em', 0.406),
        ('truthfulqa', 'truthfulqa_predictions.jsonl', Metrics.compute_f1, 'f1', 0.254),
        ('truthfulqa', 'truthfulqa_predictions.jsonl', Metrics.compute_rouge_l, 'rouge_l', 0.231), # 同一份文件算两次
        ('factkg', 'factkg_predictions.jsonl', Metrics.compute_accuracy, 'accuracy', 0.666)
    ]

    final_report = {}
    collected_scores = []

    print(f"{'='*70}")
    print(f"🚀 FAST RE-EVALUATION (Paper Aligned: Average / 6)")
    print(f"📂 Reading from: {results_dir}")
    print(f"{'='*70}")

    # 缓存已读取的数据，避免 TruthfulQA 读两次
    data_cache = {}

    for dataset, filename, metric_fn, metric_name, target in tasks:
        file_path = results_dir / filename
        
        # 1. 加载数据 (如果没加载过)
        if filename not in data_cache:
            data_cache[filename] = load_jsonl(file_path)
        
        samples = data_cache[filename]
        if not samples:
            print(f"❌ Skipping {dataset} ({metric_name}): No data found.")
            continue

        # 2. 重新计算分数 (Re-compute)
        # 我们重新跑一遍 metric_fn，确保逻辑与现在的一致
        scores = []
        for item in samples:
            pred = item['prediction']
            golds = item['gold_answers']
            scores.append(metric_fn(pred, golds))
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 3. 记录
        collected_scores.append(avg_score)
        
        if dataset not in final_report:
            final_report[dataset] = {}
        final_report[dataset][metric_name] = avg_score

        # 4. 打印状态
        status = "✅" if avg_score >= target - 0.005 else "❌"
        print(f"{dataset:<15} {metric_name.upper():<10} {avg_score:.4f} (Target: {target:.4f}) {status}")

    # 5. 计算最终平均分
    print(f"{'-'*70}")
    if len(collected_scores) == 6:
        final_avg = sum(collected_scores) / 6
        target_avg = 0.453
        status = "✅" if final_avg >= target_avg - 0.005 else "❌"
        print(f"{'AVERAGE (Div/6)':<26} {final_avg:.4f} (Target: {target_avg:.4f}) {status}")
        
        final_report['AVERAGE'] = final_avg
    else:
        print(f"⚠️ Warning: Only found {len(collected_scores)}/6 components. Cannot compute valid average.")

    print(f"{'='*70}")

    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)
    print(f"💾 Report saved to {args.output_file}")

if __name__ == "__main__":
    main()