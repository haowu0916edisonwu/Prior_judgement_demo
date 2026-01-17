#!/usr/bin/env python3
"""
Priori Judgment 专项评估 - TruthfulQA 独享版 (修复版)
只跑 TruthfulQA，并生成标准的 results.json 汇总文件
"""

import argparse
import json
import os
from pathlib import Path

# 复用现有的模块
from src.data_loader import CAREDataLoader
from src.evaluator import PrioriJudgmentEvaluator
from src.metrics import Metrics

def save_predictions(results, output_dir, dataset):
    """保存详细的预测结果"""
    filename = f"{dataset}_predictions.jsonl"
    if hasattr(output_dir, 'joinpath'):
        output_file = output_dir / filename
    else:
        output_file = os.path.join(output_dir, filename)

    print(f"正在保存详细预测到: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            pred_data = {
                'id': getattr(r, 'id', 'unknown_id'),
                'question': getattr(r, 'question', ''),
                'prediction': getattr(r, 'prediction', ''),
                'gold_answers': getattr(r, 'gold_answers', []),
                'mode': getattr(r, 'mode', 'unknown'),
                'priori_output': getattr(r, 'priori_output', None),
            }
            pred_data['correct'] = False 
            f.write(json.dumps(pred_data, ensure_ascii=False) + '\n')
    
    print("保存完成！")

def main():
    parser = argparse.ArgumentParser(description="Priori Judgment - TruthfulQA Only")
    parser.add_argument("--data_root", default="data_care/eval", help="Data directory")
    parser.add_argument("--model_name", default="NousResearch/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--output_dir", default="results_truthfulqa_v24", help="Output directory") 
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples (debug)")
    parser.add_argument("--save_predictions", action="store_true", default=True, help="Save detailed predictions") 
    args = parser.parse_args()
    
    dataset = 'truthfulqa'
    targets = {'f1': 0.2540, 'rouge_l': 0.2310}
    
    print("=" * 70)
    print("🚀 Priori Judgment Evaluation - TRUTHFULQA ONLY (v24.0 Check)")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # 初始化
    data_loader = CAREDataLoader(args.data_root, verbose=False)
    evaluator = PrioriJudgmentEvaluator(args.model_name)
    
    print(f"\n{'='*70}")
    print(f"📊 Evaluating {dataset.upper()}")
    print(f"{'='*70}")
    
    # 加载数据
    samples = data_loader.load_dataset(dataset)
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"🔧 Debug mode: using {len(samples)} samples")
    
    # 评估
    result = evaluator.evaluate_dataset(samples)
    metrics = result['metrics']
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存详细预测 (.jsonl)
    if args.save_predictions:
        save_predictions(result['results'], output_dir, dataset)
    
    # 2. [关键修复] 保存标准汇总结果 (results.json)
    # 这就是您要的那个格式
    final_results = {
        dataset: {
            "metrics": metrics,
            "mode_distribution": result['mode_distribution']
        }
    }
    
    json_path = output_dir / "results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print(f"\n📈 Results (TruthfulQA):")
    f1 = metrics.get('f1', 0)
    rouge = metrics.get('rouge_l', 0)
    f1_status = "✅" if f1 >= targets['f1'] - 0.005 else "❌"
    rouge_status = "✅" if rouge >= targets['rouge_l'] - 0.005 else "❌"
    
    print(f"  F1:      {f1:.4f} (Target: {targets['f1']:.4f}) {f1_status}")
    print(f"  ROUGE-L: {rouge:.4f} (Target: {targets['rouge_l']:.4f}) {rouge_status}")
    
    print(f"\n🔀 Mode Distribution:")
    for mode, count in result['mode_distribution'].items():
        pct = count / len(samples) * 100
        print(f"  {mode}: {count}/{len(samples)} ({pct:.1f}%)")

    print(f"\n💾 Files saved:")
    print(f"  - {json_path} (标准 JSON 汇总)")
    print(f"  - {output_dir}/{dataset}_predictions.jsonl (详细预测)")

if __name__ == "__main__":
    main()