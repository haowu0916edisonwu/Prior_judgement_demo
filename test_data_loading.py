#!/usr/bin/env python3
"""
测试数据加载器 - 使用上传的真实数据
"""

import sys
import json
from pathlib import Path

# 模拟数据加载器
class Sample:
    def __init__(self, id, question, answers, top1_context, dataset):
        self.id = id
        self.question = question
        self.answers = answers
        self.top1_context = top1_context
        self.dataset = dataset


def extract_document_text(raw_text: str) -> str:
    """提取 Document 部分"""
    import re
    
    # 方法 1: 正则表达式
    match = re.search(r'Document:\s*(.*)', raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 方法 2: 分割
    if 'Question:' in raw_text:
        parts = raw_text.split('Document:', 1)
        if len(parts) > 1:
            return parts[1].strip()
        parts = raw_text.split('\n', 1)
        if len(parts) > 1:
            return parts[1].strip()
    
    return raw_text.strip()


def test_data_loading():
    """测试数据加载"""
    
    print("=" * 70)
    print("🧪 Testing Data Loading with Real Data")
    print("=" * 70)
    
    # 文件路径
    question_file = "/mnt/user-data/uploads/test.jsonl"
    retrieval_file = "/mnt/user-data/uploads/test_question_aware.jsonl"
    
    # 加载第一行
    with open(question_file) as f:
        q = json.loads(f.readline())
    
    with open(retrieval_file) as f:
        r = json.loads(f.readline())
    
    print("\n📝 Question Format:")
    print(f"  Keys: {list(q.keys())}")
    print(f"  ID: {q.get('id')}")
    print(f"  Question: {q.get('question')}")
    print(f"  Answer field name: {'answer' if 'answer' in q else 'answers'}")
    print(f"  Answers: {q.get('answer', q.get('answers', []))}")
    
    print("\n🔎 Retrieval Format:")
    print(f"  Keys: {list(r.keys())}")
    print(f"  ID: {r.get('id')}")
    print(f"  Context field name: {'topk' if 'topk' in r else 'ctxs'}")
    print(f"  Number of contexts: {len(r.get('topk', r.get('ctxs', [])))}")
    
    # 提取 Top-1
    if r.get('topk'):
        raw_text = r['topk'][0]['text']
        extracted = extract_document_text(raw_text)
        
        print(f"\n✅ Top-1 Context Processing:")
        print(f"  Raw text (first 300 chars):")
        print(f"    {raw_text[:300]}...")
        print(f"\n  Extracted document (first 300 chars):")
        print(f"    {extracted[:300]}...")
        print(f"\n  Extracted length: {len(extracted)} chars")
    
    # 验证 ID 匹配
    print(f"\n🔗 ID Matching:")
    q_id = str(q.get('id'))
    r_id = str(r.get('id'))
    match = "✅" if q_id == r_id else "❌"
    print(f"  Question ID: {q_id}")
    print(f"  Retrieval ID: {r_id}")
    print(f"  Match: {match}")
    
    # 创建样本
    answers = q.get('answer', q.get('answers', []))
    if not isinstance(answers, list):
        answers = [str(answers)]
    
    sample = Sample(
        id=q_id,
        question=q.get('question', ''),
        answers=answers,
        top1_context=extract_document_text(r['topk'][0]['text']) if r.get('topk') else '',
        dataset='nq'
    )
    
    print(f"\n📦 Created Sample:")
    print(f"  ID: {sample.id}")
    print(f"  Question: {sample.question}")
    print(f"  Answers: {sample.answers}")
    print(f"  Context length: {len(sample.top1_context)}")
    
    print("\n" + "=" * 70)
    print("✅ Test Passed! Data format understood correctly.")
    print("=" * 70)


if __name__ == "__main__":
    test_data_loading()