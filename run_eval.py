#!/usr/bin/env python3
"""
Priori Judgment 评估 - 修正版 (v25.0 Final)
适配实际 CARE 数据格式
修正算分逻辑：TruthfulQA 计入 F1 和 ROUGE 两项，总分母为 6。
"""

import argparse
import json
import os
from pathlib import Path

# 需要将修正后的 data_loader 复制到 src/ 目录
# 这里假设已经完成
from src.data_loader import CAREDataLoader
from src.evaluator import PrioriJudgmentEvaluator
from src.metrics import Metrics


def debug_single_sample(data_loader, evaluator, dataset):
    """调试单个样本"""
    print("\n" + "=" * 70)
    print(f"🐛 DEBUG MODE: {dataset.upper()}")
    print("=" * 70)
    
    samples = data_loader.load_dataset(dataset)
    sample = samples[0]
    
    print(f"\n📝 Sample Info:")
    print(f"  ID: {sample.id}")
    print(f"  Question: {sample.question[:200]}...")
    print(f"  Answers: {sample.answers}")
    
    # [修改点 1] 将 sample.top1_context 改为 sample.context
    print(f"  Context length: {len(sample.context)} chars")
    # [修改点 2] 将 sample.top1_context 改为 sample.context
    print(f"  Context preview: {sample.context[:300]}...")
    
    print(f"\n🔄 Running Two-Stage Inference...")
    result = evaluator.evaluate_sample(sample)
    
    print(f"\n📊 Results:")
    print(f"  Stage 1 (Priori) Output: {result.priori_output}")
    # print(f"  Unknown detected: {evaluator.is_unknown(result.priori_output)}") # evaluator 可能没有公开这个方法，注释掉以防万一
    print(f"  Final Answer: {result.prediction}")
    print(f"  Mode: {result.mode}")
    print(f"  Gold Answers: {result.gold_answers}")
    
    # 计算指标
    task_type = evaluator.TASK_TYPES.get(dataset, 'open_qa')
    
    if task_type == "fact_checking":
        score = Metrics.compute_accuracy(result.prediction, result.gold_answers)
        metric_name = "Accuracy"
    elif task_type == "long_form":
        score = Metrics.compute_f1(result.prediction, result.gold_answers)
        metric_name = "F1"
    else:
        score = Metrics.compute_span_em(result.prediction, result.gold_answers)
        metric_name = "Span EM"
    
    print(f"\n✅ {metric_name}: {score:.4f}")
    print("=" * 70)


def save_predictions(results, output_dir, dataset):
    """保存详细的预测结果 (修复版)"""
    # 兼容处理：无论 output_dir 是字符串还是 Path 对象都能跑
    filename = f"{dataset}_predictions.jsonl"
    if hasattr(output_dir, 'joinpath'): # 如果是 Path 对象
        output_file = output_dir / filename
    else: # 如果是字符串
        output_file = os.path.join(output_dir, filename)

    print(f"正在保存详细预测到: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            # 1. 基础字段
            pred_data = {
                'id': getattr(r, 'id', 'unknown_id'),
                'question': getattr(r, 'question', ''),
                'prediction': getattr(r, 'prediction', ''),
                'gold_answers': getattr(r, 'gold_answers', []),
                'mode': getattr(r, 'mode', 'unknown'),
                'priori_output': getattr(r, 'priori_output', None),
            }

            # 2. 智能推断 Task Type
            task_type = getattr(r, 'task_type', None)
            
            # 如果没有 task_type，尝试通过答案推断
            if not task_type:
                gold = getattr(r, 'gold_answers', [])
                if len(gold) == 1 and str(gold[0]).lower() in ['true', 'false']:
                    task_type = 'fact_checking'
                else:
                    task_type = 'qa'

            # 3. 根据类型安全地读取分数
            if task_type == "fact_checking":
                pred_data['correct'] = getattr(r, 'accuracy', 0) == 1.0
            else:
                pred_data['correct'] = getattr(r, 'span_em', 0) == 1.0

            f.write(json.dumps(pred_data, ensure_ascii=False) + '\n')
    
    print("保存完成！")


def main():
    parser = argparse.ArgumentParser(
        description="Priori Judgment Baseline Evaluation (CARE Data - Fixed)"
    )
    parser.add_argument("--data_root", default="data_care/eval", help="Data directory")
    parser.add_argument("--model_name", default="NousResearch/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--datasets", nargs='+', default=None, help="Datasets to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples (debug)")
    parser.add_argument("--debug_sample", action="store_true", help="Debug single sample")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--save_predictions", action="store_true", help="Save detailed predictions")
    args = parser.parse_args()
    
    # 数据集配置
    all_datasets = ['nq', 'trivia', 'webqa', 'truthfulqa', 'factkg']
    if args.datasets:
        all_datasets = [d for d in all_datasets if d in args.datasets]
    
    # [关键修改] 论文标准指标列表 (6项)
    # 格式: (Dataset, Metric Key, Target Value)
    paper_metrics_def = [
        ('nq', 'span_em', 0.458),
        ('trivia', 'span_em', 0.704),
        ('webqa', 'span_em', 0.406),
        ('truthfulqa', 'f1', 0.254),       # TQA Item 1
        ('truthfulqa', 'rouge_l', 0.231),  # TQA Item 2
        ('factkg', 'accuracy', 0.666)
    ]
    
    # 用于单项展示的目标字典
    targets_map = {
        'nq': {'span_em': 0.458},
        'trivia': {'span_em': 0.704},
        'webqa': {'span_em': 0.406},
        'truthfulqa': {'f1': 0.254, 'rouge_l': 0.231},
        'factkg': {'accuracy': 0.666}
    }
    
    print("=" * 70)
    print("🚀 Priori Judgment Evaluation (CARE Data - Fixed)")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_root}")
    print(f"Using: test_question_aware.jsonl (Top-10 Context)") 
    print(f"Format: Answer + Topk Fields")
    print("=" * 70)
    
    # 初始化
    data_loader = CAREDataLoader(args.data_root, verbose=args.verbose)
    evaluator = PrioriJudgmentEvaluator(args.model_name)
    
    # Debug mode
    if args.debug_sample:
        for dataset in all_datasets:
            debug_single_sample(data_loader, evaluator, dataset)
        return
    
    # 评估
    all_results = {}
    
    for dataset in all_datasets:
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
        all_results[dataset] = result
        
        # 保存详细预测
        if args.save_predictions:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_predictions(result['results'], output_dir, dataset)
        
        # 打印结果
        metrics = result['metrics']
        print(f"\n📈 Results:")
        for k, v in metrics.items():
            target_v = targets_map[dataset].get(k, 0)
            status = "✅" if v >= target_v - 0.005 else "❌"
            print(f"  {k.upper()}: {v:.4f} (target: {target_v:.4f}) {status}")
        
        # 打印模式分布
        print(f"\n🔀 Mode Distribution:")
        for mode, count in result['mode_distribution'].items():
            pct = count / len(samples) * 100
            print(f"  {mode}: {count}/{len(samples)} ({pct:.1f}%)")
    
    # ======================================================================
    # 📊 FINAL RESULTS (Strict Format & Logic)
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"📊 FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Metric':<12} {'Result':<10} {'Target':<10} {'Status'}")
    print(f"{'-'*70}")
    
    collected_scores = []
    
    # 按照定义的顺序遍历 6 项指标
    for ds_name, metric_key, target_val in paper_metrics_def:
        # 检查是否跑了这个数据集
        if ds_name in all_results and metric_key in all_results[ds_name]['metrics']:
            score = all_results[ds_name]['metrics'][metric_key]
            collected_scores.append(score)
            
            status = "✅" if score >= target_val - 0.005 else "❌"
            
            print(
                f"{ds_name:<15} "
                f"{metric_key.upper():<12} "
                f"{score:<10.4f} "
                f"{target_val:<10.4f} "
                f"{status}"
            )
        else:
            # 如果没跑，显示 N/A
            print(
                f"{ds_name:<15} "
                f"{metric_key.upper():<12} "
                f"{'N/A':<10} "
                f"{target_val:<10.4f} "
                f"⚪"
            )

    print(f"{'-'*70}")
    
    # 计算平均分 (Standard: Div by 6)
    final_avg = 0.0
    avg_status = ""
    
    if len(collected_scores) == 6:
        final_avg = sum(collected_scores) / 6
        target_avg = 0.453
        status_icon = "✅" if final_avg >= target_avg - 0.005 else "❌"
        
        print(
            f"{'AVERAGE':<15} "
            f"{'':<12} "
            f"{final_avg:<10.4f} "
            f"{target_avg:<10.4f} "
            f"{status_icon}"
        )
    elif collected_scores:
        # 部分运行的情况
        final_avg = sum(collected_scores) / len(collected_scores)
        print(
            f"{'PARTIAL AVG':<15} "
            f"{'(Div/'+str(len(collected_scores))+')':<12} "
            f"{final_avg:<10.4f} "
            f"{'0.4530':<10} "
            f"⚠️"
        )
    else:
        print("No scores collected.")

    print(f"{'='*70}")
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存汇总结果 (JSON)
    with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump({
            dataset: {
                'metrics': result['metrics'],
                'mode_distribution': result.get('mode_distribution', {})
            }
            for dataset, result in all_results.items()
        }, f, indent=2, ensure_ascii=False)
    
    # 保存汇总统计 (TXT)
    with open(output_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write("PRIORI JUDGMENT EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # 写入每个数据集的详细分
        for dataset in all_datasets:
            if dataset not in all_results:
                continue
            f.write(f"{dataset.upper()}\n")
            for k, v in all_results[dataset]['metrics'].items():
                f.write(f"  {k.upper()}: {v:.4f}\n")
            f.write("\n")
        
        # 写入最终平均分
        if collected_scores:
            label = "AVERAGE" if len(collected_scores) == 6 else "PARTIAL AVG"
            f.write(f"{label}: {final_avg:.4f}\n")
    
    print(f"\n💾 Results saved:")
    print(f"  - {output_dir}/results.json (主结果)")
    print(f"  - {output_dir}/summary.txt (文本汇总)")
    if args.save_predictions:
        print(f"  - {output_dir}/{{dataset}}_predictions.jsonl (详细预测)")


if __name__ == "__main__":
    main()