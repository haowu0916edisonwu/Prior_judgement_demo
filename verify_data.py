#!/usr/bin/env python3
"""
验证数据格式和 Top-1 提取
"""

import json
from pathlib import Path


def check_data_format():
    """检查数据格式"""
    data_root = Path("data_care/eval")
    
    # 检查 NQ 数据
    print("=" * 70)
    print("🔍 Checking NQ Data Format")
    print("=" * 70)
    
    # 1. 问题文件
    question_file = data_root / "nq" / "test.jsonl"
    with open(question_file) as f:
        q_sample = json.loads(f.readline())
    
    print("\n📝 Question Format:")
    print(f"  Keys: {list(q_sample.keys())}")
    print(f"  ID: {q_sample.get('id')}")
    print(f"  Question: {q_sample.get('question', '')[:100]}...")
    print(f"  Answers: {q_sample.get('answers', [])}")
    
    # 2. 检索文件（question_aware）
    retrieval_file = data_root / "nq" / "retrieval" / "colbertv2" / "test_question_aware.jsonl"
    with open(retrieval_file) as f:
        r_sample = json.loads(f.readline())
    
    print("\n🔎 Retrieval Format (question_aware):")
    print(f"  Keys: {list(r_sample.keys())}")
    print(f"  ID: {r_sample.get('id')}")
    print(f"  Number of contexts: {len(r_sample.get('ctxs', []))}")
    
    if r_sample.get('ctxs'):
        top1 = r_sample['ctxs'][0]
        print(f"\n✅ Top-1 Context:")
        print(f"  Keys: {list(top1.keys())}")
        print(f"  Text (first 200 chars): {top1.get('text', '')[:200]}...")
        print(f"  Title: {top1.get('title', 'N/A')}")
        print(f"  Score: {top1.get('score', 'N/A')}")
    
    # 3. ID 匹配验证
    print("\n🔗 ID Matching:")
    q_id = str(q_sample.get('id', ''))
    r_id = str(r_sample.get('id', ''))
    match = "✅" if q_id == r_id else "❌"
    print(f"  Question ID: {q_id}")
    print(f"  Retrieval ID: {r_id}")
    print(f"  Match: {match}")
    
    # 4. 统计样本数
    print("\n📊 Dataset Statistics:")
    for dataset in ['nq', 'triviaqa', 'webqa', 'truthfulqa', 'factkg']:
        q_file = data_root / dataset / "test.jsonl"
        r_file = data_root / dataset / "retrieval" / "colbertv2" / "test_question_aware.jsonl"
        
        with open(q_file) as f:
            n_questions = sum(1 for _ in f)
        with open(r_file) as f:
            n_retrievals = sum(1 for _ in f)
        
        match = "✅" if n_questions == n_retrievals else "❌"
        print(f"  {dataset:<12} Q: {n_questions:<5} R: {n_retrievals:<5} {match}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_data_format()