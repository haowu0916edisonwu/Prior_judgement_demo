"""
Experimental Patch: Soft-Refusal Rescue (Ablation Study)
------------------------------------------------------------------
实验目的 (Objective):
探究 NQ 和 TriviaQA 数据集中是否存在“伪拒答（Pseudo-Refusals）”导致的性能损失。
即：模型输出了 "Context does not mention..." 但未触发闭卷模式的情况。

方法 (Methodology):
1. 扫描 RAG 输出，提取包含 "not mention", "no information" 等拒答关键词的样本。
2. 强制将这些样本的模式从 'rag' 切换为 'closedbook'。
3. 使用 Llama-3 模型重新生成答案。

实验结论 (Conclusion):
经过 v26/v27 两轮测试，发现分数并未显著提升（NQ 保持在 0.4457，TriviaQA 保持在 0.7034）。
原因分析：
1. 模型本身的 Priori Judgment (v25) 已经达到了帕累托最优，大多数“拒答”是由于模型确实不知道答案（Knowledge Gap），切换闭卷后依然无法回答。
2. 强制切换闭卷引入了 False Positive 风险（误伤了包含 "unknown" 单词的正确答案）。

最终决策 (Verdict):
保留 v25.0 版本结果作为最终 Baseline。本脚本留作实验记录，证明我们
"""
import json
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 配置 ---
MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
DATA_DIR = "results"         # 注意：这里我们要读取 v25 (原始 SOTA) 的数据，不要读 v26 (已损坏) 的！
OUTPUT_DIR = "results_v27"   # 存到新目录

# 1. 绝对安全的 Trigger (去掉 unknown, unclear)
SAFE_TRIGGERS = [
    "does not mention", "doesn't mention", 
    "not provide", "no information",
    "context does not", "passage does not",
    "cannot answer", "text does not",
    "provided text"
]

TARGETS = [
    {"name": "nq", "filename": "nq_predictions.jsonl"},
    {"name": "trivia", "filename": "trivia_predictions.jsonl"}
]

def load_jsonl(path):
    data = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    print(f"🚑 启动 v27.0 最终微创手术 (NQ + Trivia)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tasks = []
    
    # 1. 扫描与筛选
    for t in TARGETS:
        path = os.path.join(DATA_DIR, t["filename"])
        print(f"📖 读取原始 v25 数据: {t['name']}...")
        
        if not os.path.exists(path):
            print(f"❌ 找不到文件: {path}，请确保 results 文件夹里有原始数据！")
            continue
            
        data = load_jsonl(path)
        indices_to_rescue = []
        skipped_long = 0
        skipped_ambiguous = 0
        
        for i, item in enumerate(data):
            if item.get('mode') == 'rag':
                pred = item.get('prediction', '').strip()
                pred_lower = pred.lower()
                
                # A. 命中严格拒答词
                hit_trigger = any(trig in pred_lower for trig in SAFE_TRIGGERS)
                
                # B. 长度检查 (关键！)
                # 如果回答很长(>20词)，大概率是正确的解释，或者是 "Although unknown, it is X"
                is_short = len(pred.split()) < 20
                
                if hit_trigger:
                    if is_short:
                        indices_to_rescue.append(i)
                    else:
                        skipped_long += 1
                elif "unknown" in pred_lower:
                    # 记录一下我们故意放过的 unknown
                    skipped_ambiguous += 1

        print(f"   [{t['name'].upper()}] 扫描结果:")
        print(f"     - 🚑 待手术 (确信拒答): {len(indices_to_rescue)}")
        print(f"     - 🛡️ 已保护 (长句误伤): {skipped_long} (v26就是死在这里)")
        print(f"     - 🛡️ 已忽略 (Unknown): {skipped_ambiguous}")
        
        if indices_to_rescue:
            tasks.append({'name': t['name'], 'data': data, 'indices': indices_to_rescue, 'filename': t['filename']})
        else:
            # 如果没得救，直接保存原版
            save_jsonl(data, os.path.join(OUTPUT_DIR, t["filename"]))
            print(f"     ✅ 无需修复，已原样保存。")

    if not tasks:
        print("✅ 所有文件处理完毕。")
        return

    # 2. 加载模型
    print(f"\n🔧 加载模型 {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    # 3. 执行手术
    print(f"\n⚡ 开始微创手术...")
    
    def make_prompt(q):
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question concisely.\nQuestion: {q}\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    for task in tasks:
        name = task['name']
        data = task['data']
        indices = task['indices']
        
        print(f"👉 修复 {name} ({len(indices)} 题)...")
        
        for idx in tqdm(indices):
            item = data[idx]
            q = item['question']
            
            # 生成
            inputs = tokenizer(make_prompt(q), return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=20, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
                )
            new_pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # 截断
            if "." in new_pred: new_pred = new_pred.split(".", 1)[0].strip()
            
            # 更新
            item['prediction'] = new_pred
            item['mode'] = 'closedbook_rescue'
            data[idx] = item
            
        # 保存
        out_path = os.path.join(OUTPUT_DIR, task['filename'])
        save_jsonl(data, out_path)
        print(f"✅ {name} 修复版已保存")

    # 4. 复制其他 SOTA 文件 (WebQA, FactKG, TruthfulQA)
    import shutil
    other_files = ["webqa_predictions.jsonl", "truthfulqa_predictions.jsonl", "factkg_predictions.jsonl"]
    for f_name in other_files:
        src = os.path.join(DATA_DIR, f_name)
        dst = os.path.join(OUTPUT_DIR, f_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"📦 已复制 {f_name}")

    print(f"\n🎉 v27.0 结束！请运行: python fast_re_eval.py --results_dir results_v27")

if __name__ == "__main__":
    main()