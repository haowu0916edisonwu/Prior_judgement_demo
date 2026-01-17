#!/usr/bin/env python3
"""
Demo Check - 快速检查模型输出和清洗效果
(Strictly aligned with your Git Repo logic)
"""

import argparse
import random
import sys
import os
import torch

# 确保能导入 src 目录下的模块 (和 run_eval.py 一致)
sys.path.append(os.getcwd())

from src.data_loader import CAREDataLoader
from src.evaluator import PrioriJudgmentEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Demo Check for Priori Judgment System")
    
    # === 核心参数 (完全对齐 run_eval.py 的参数命名) ===
    parser.add_argument("--model_name", type=str, 
                        default="NousResearch/Meta-Llama-3-8B-Instruct", 
                        help="Model name or path")
    
    parser.add_argument("--data_root", type=str, 
                        default="data_care/eval", 
                        help="Data directory")
    
    # === Check 专用参数 ===
    # 默认检查这三个关键数据集
    parser.add_argument("--datasets", nargs='+', 
                        default=["nq", "trivia", "webqa"], 
                        help="Datasets to check")
    
    parser.add_argument("--num_samples", type=int, default=2, 
                        help="Number of random samples to check per dataset")
    
    # 增加 device 参数方便调试
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    
    return parser.parse_args()

def run_demo(args):
    print(f"\n🔧 [Init] Loading Model from: {args.model_name}")
    print(f"   (Device: {args.device})")
    
    try:
        # 复用你 evaluator.py 的初始化逻辑
        evaluator = PrioriJudgmentEvaluator(args.model_name, device=args.device)
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print(f"💡 请检查路径 '{args.model_name}' 是否正确？")
        return

    print(f"\n📂 [Init] Loading Data Loader from: {args.data_root}")
    try:
        # 复用你 data_loader.py 的初始化逻辑
        loader = CAREDataLoader(args.data_root)
    except Exception as e:
        print(f"\n❌ 数据加载器初始化失败: {e}")
        print(f"💡 请检查 data_root '{args.data_root}' 是否存在？")
        return

    print("\n" + "="*70)
    print(f"🔍 SOTA Logic & Prompt Check (Running {args.num_samples} samples per dataset)")
    print("="*70)

    for ds_name in args.datasets:
        print(f"\n📂 Dataset: {ds_name.upper()}")
        try:
            # load_dataset 会自动处理 trivia -> triviaqa 的映射
            samples = loader.load_dataset(ds_name)
        except Exception as e:
            print(f"   ⚠️ Skipping {ds_name}: {e}")
            continue
        
        if not samples:
            print(f"   ⚠️ No samples found for {ds_name}")
            continue

        # 随机采样
        k = min(len(samples), args.num_samples)
        test_samples = random.sample(samples, k)
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n🔹 [Sample #{i} | ID: {sample.id}]")
            print(f"   Q: {sample.question}")
            
            # === 运行评估 (调用你修改后的 evaluator.py) ===
            result = evaluator.evaluate_sample(sample)
            
            # === 结果分析 ===
            raw_out = result.priori_output
            clean_pred = result.prediction
            mode = result.mode
            
            # 1. 检查 Prompt 锚点是否生效
            # 如果 prompts.py 改好了，raw_out 应该直接是答案，不包含 '?\nAnswer:'
            prompt_status = "✅ Clean"
            if "?\nAnswer:" in raw_out:
                prompt_status = "❌ LEAKED (Old Prompt detected)"
            elif raw_out.strip().startswith("Note:") or raw_out.strip().startswith("Question:"):
                 prompt_status = "⚠️ Messy Start"

            print(f"   --------------------------------------------------")
            print(f"   🤖 Raw Output   : {repr(raw_out)}")
            print(f"   🔎 Prompt Status: {prompt_status}")
            print(f"   ✨ Final Answer : {repr(clean_pred)}")
            print(f"   🏷️  Mode        : {mode.upper()}")
            
            # 2. 策略逻辑验证 (Strategy Check)
            
            # Check TriviaQA: 应该允许 Unknown -> Closedbook
            if ds_name in ['trivia', 'triviaqa']:
                if "unknown" in clean_pred.lower():
                    if mode == 'closedbook':
                        print(f"   ✅ Strategy: Correctly fell back to Closed-book")
                    else:
                        print(f"   ❌ Strategy: Predicted Unknown but stuck in RAG (Strategy B1 failed?)")
            
            # Check NQ/WebQA: 应该捕捉 'does not mention'
            if ds_name in ['nq', 'webqa']:
                 refusal_keywords = ["does not mention", "not provided", "no information", "cannot answer"]
                 is_refusal = any(k in raw_out.lower() for k in refusal_keywords)
                 
                 if is_refusal:
                     if mode == 'closedbook':
                         print(f"   ✅ Strategy: Caught indirect refusal ('{clean_pred[:20]}...') -> Closedbook")
                     else:
                         # 注意：如果你保留了“强行召回”策略，这里可能是 RAG，视你的修改而定
                         print(f"   ℹ️  Strategy: Refusal detected but kept in RAG (Check if this is intended)")
            
            print(f"   --------------------------------------------------")

    print("\n✅ Check finished!")

if __name__ == "__main__":
    args = parse_args()
    run_demo(args)