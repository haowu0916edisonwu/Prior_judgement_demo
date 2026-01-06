"""
数据加载器 - CARE 完整数据格式适配（终极版）

基于完整 eval.zip 的深度分析，支持所有5个数据集的3种匹配模式：
1. ID 匹配 (NQ)
2. Query 匹配 (TriviaQA)
3. Question 匹配 (WebQA, TruthfulQA, FactKG)

关键修正：
- answer 字段（不是 answers）
- topk 字段（不是 ctxs）
- question_aware 的特殊文本格式
- 自动检测并适配3种匹配模式
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Sample:
    """统一样本格式"""
    id: str
    question: str
    answers: List[str]
    top1_context: str
    dataset: str


class CAREDataLoader:
    """CARE 数据格式加载器（终极版 - 支持所有3种匹配模式）"""
    
    def __init__(self, data_root: str = "data_care/eval", verbose: bool = False):
        self.data_root = Path(data_root)
        self.verbose = verbose
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")
    
    def load_dataset(self, dataset_name: str) -> List[Sample]:
        """
        加载数据集
        
        自动检测匹配模式：
        - 如果检索文件有 id → 使用 ID 匹配（NQ）
        - 如果检索文件有 query → 使用 Query 匹配（TriviaQA）
        - 如果检索文件有 question → 使用 Question 匹配（WebQA, TruthfulQA, FactKG）
        
        Args:
            dataset_name: nq, trivia, webqa, truthfulqa, factkg
        
        Returns:
            Sample 对象列表
        """
        # 映射数据集名称
        dataset_map = {
            'nq': 'nq',
            'trivia': 'triviaqa',
            'triviaqa': 'triviaqa',
            'webqa': 'webqa',
            'truthfulqa': 'truthfulqa',
            'factkg': 'factkg'
        }
        
        folder = dataset_map.get(dataset_name, dataset_name)
        dataset_dir = self.data_root / folder
        
        # 文件路径
        question_file = dataset_dir / "test.jsonl"
        retrieval_file = dataset_dir / "retrieval" / "colbertv2" / "test_question_aware.jsonl"
        
        if self.verbose:
            print(f"\n📂 Loading {dataset_name}:")
            print(f"  Question: {question_file}")
            print(f"  Retrieval: {retrieval_file}")
        
        # 检查文件
        if not question_file.exists():
            raise FileNotFoundError(f"Question file not found: {question_file}")
        if not retrieval_file.exists():
            raise FileNotFoundError(f"Retrieval file not found: {retrieval_file}")
        
        # 加载数据
        questions = self._load_jsonl(question_file)
        retrievals = self._load_jsonl(retrieval_file)
        
        if self.verbose:
            print(f"  Loaded: {len(questions)} questions, {len(retrievals)} retrievals")
        
        # 检测匹配模式
        match_mode = self._detect_match_mode(retrievals)
        
        if self.verbose:
            print(f"  Match mode: {match_mode}")
        
        # 根据匹配模式合并数据
        if match_mode == "ID":
            samples = self._merge_by_id(questions, retrievals, dataset_name)
        elif match_mode == "QUERY":
            samples = self._merge_by_query(questions, retrievals, dataset_name)
        elif match_mode == "QUESTION":
            samples = self._merge_by_question(questions, retrievals, dataset_name)
        else:
            raise ValueError(f"Unknown match mode: {match_mode}")
        
        print(f"✅ Loaded {len(samples)} valid samples from {dataset_name}")
        return samples
    
    def _detect_match_mode(self, retrievals: List[Dict]) -> str:
        """
        自动检测匹配模式
        
        检测逻辑：
        1. 检查第一条记录是否有 'id' → ID 匹配（NQ）
        2. 检查第一条记录是否有 'query' → Query 匹配（TriviaQA）
        3. 检查第一条记录是否有 'question' → Question 匹配（WebQA等）
        
        Args:
            retrievals: 检索结果列表
        
        Returns:
            "ID", "QUERY", or "QUESTION"
        """
        if not retrievals:
            raise ValueError("Empty retrievals list")
        
        first = retrievals[0]
        
        if 'id' in first:
            return "ID"
        elif 'query' in first:
            return "QUERY"
        elif 'question' in first:
            return "QUESTION"
        else:
            raise ValueError(
                f"Cannot detect match mode. Available keys: {list(first.keys())}"
            )
    
    def _merge_by_id(
        self,
        questions: List[Dict],
        retrievals: List[Dict],
        dataset_name: str
    ) -> List[Sample]:
        """
        通过 ID 匹配合并数据（NQ 模式）
        
        特点：
        - 检索文件有 id 字段
        - 使用 zip 遍历，O(n) 时间复杂度
        - 要求文件顺序一致
        
        Args:
            questions: 问题列表
            retrievals: 检索列表
            dataset_name: 数据集名称
        
        Returns:
            Sample 列表
        """
        # 验证数量
        if len(questions) != len(retrievals):
            print(f"  ⚠️  Warning: question count ({len(questions)}) != retrieval count ({len(retrievals)})")
        
        samples = []
        skipped = 0
        
        for idx, (q, r) in enumerate(zip(questions, retrievals)):
            # 验证 ID 匹配（转为字符串比较，因为 NQ 是 int，可能有类型差异）
            q_id = str(q.get('id', ''))
            r_id = str(r.get('id', ''))
            
            if q_id != r_id:
                if self.verbose and skipped < 3:
                    print(f"  ⚠️  Sample {idx}: ID mismatch (Q={q_id}, R={r_id})")
                skipped += 1
                continue
            
            # 提取并创建样本
            sample = self._create_sample(q, r, dataset_name, idx)
            if sample:
                samples.append(sample)
        
        if skipped > 0:
            print(f"  ⚠️  Skipped {skipped} samples due to ID mismatch")
        
        return samples
    
    def _merge_by_query(
        self,
        questions: List[Dict],
        retrievals: List[Dict],
        dataset_name: str
    ) -> List[Sample]:
        """
        通过 Query 匹配合并数据（TriviaQA 模式）
        
        特点：
        - 检索文件有 query 字段（无 id）
        - 使用字典匹配，O(n) 空间和时间
        - 不依赖文件顺序
        
        匹配关系: q['question'] == r['query']
        
        Args:
            questions: 问题列表
            retrievals: 检索列表
            dataset_name: 数据集名称
        
        Returns:
            Sample 列表
        """
        if self.verbose:
            print(f"  Using query-based matching (r['query'] == q['question'])")
        
        # 创建检索字典：query -> retrieval
        retrieval_dict = {}
        for r in retrievals:
            query_text = r.get('query', '')
            if query_text:
                retrieval_dict[query_text] = r
        
        if self.verbose:
            print(f"  Built retrieval dict: {len(retrieval_dict)} entries")
        
        samples = []
        skipped = 0
        
        for idx, q in enumerate(questions):
            question_text = q.get('question', '')
            
            # 通过 question 查找对应的 query
            r = retrieval_dict.get(question_text)
            
            if r is None:
                if self.verbose and skipped < 3:
                    print(f"  ⚠️  Sample {idx}: No retrieval found for: {question_text[:50]}...")
                skipped += 1
                continue
            
            # 提取并创建样本
            sample = self._create_sample(q, r, dataset_name, idx)
            if sample:
                samples.append(sample)
        
        if skipped > 0:
            print(f"  ⚠️  Skipped {skipped} samples (no matching retrieval)")
        
        return samples
    
    def _merge_by_question(
        self,
        questions: List[Dict],
        retrievals: List[Dict],
        dataset_name: str
    ) -> List[Sample]:
        """
        通过 Question 匹配合并数据（WebQA, TruthfulQA, FactKG 模式）
        
        特点：
        - 检索文件有 question 字段（无 id）
        - 使用字典匹配，O(n) 空间和时间
        - 不依赖文件顺序
        
        匹配关系: q['question'] == r['question']
        
        Args:
            questions: 问题列表
            retrievals: 检索列表
            dataset_name: 数据集名称
        
        Returns:
            Sample 列表
        """
        if self.verbose:
            print(f"  Using question-based matching (r['question'] == q['question'])")
        
        # 创建检索字典：question -> retrieval
        retrieval_dict = {}
        for r in retrievals:
            q_text = r.get('question', '')
            if q_text:
                retrieval_dict[q_text] = r
        
        if self.verbose:
            print(f"  Built retrieval dict: {len(retrieval_dict)} entries")
        
        samples = []
        skipped = 0
        
        for idx, q in enumerate(questions):
            question_text = q.get('question', '')
            
            # 查找对应的检索结果
            r = retrieval_dict.get(question_text)
            
            if r is None:
                if self.verbose and skipped < 3:
                    print(f"  ⚠️  Sample {idx}: No retrieval found for: {question_text[:50]}...")
                skipped += 1
                continue
            
            # 提取并创建样本
            sample = self._create_sample(q, r, dataset_name, idx)
            if sample:
                samples.append(sample)
        
        if skipped > 0:
            print(f"  ⚠️  Skipped {skipped} samples (no matching retrieval)")
        
        return samples
    
    def _create_sample(
        self,
        question: Dict,
        retrieval: Dict,
        dataset_name: str,
        idx: int
    ) -> Sample:
        """
        创建统一的 Sample 对象
        
        Args:
            question: 问题字典
            retrieval: 检索字典
            dataset_name: 数据集名称
            idx: 样本索引
        
        Returns:
            Sample 对象
        """
        # 提取 ID（优先使用问题文件的 id，如果没有则使用索引）
        sample_id = str(question.get('id', str(idx)))
        
        # 提取问题文本
        question_text = question.get('question', question.get('claim', ''))
        
        # 提取答案（关键：字段名是 answer 不是 answers）
        answers = question.get('answer', question.get('answers', []))
        if not isinstance(answers, list):
            answers = [str(answers)]
        
        # 提取 Top-1 context
        top1_ctx = self._extract_top1_context(retrieval, idx)
        
        return Sample(
            id=sample_id,
            question=question_text,
            answers=answers,
            top1_context=top1_ctx,
            dataset=dataset_name
        )
    
    def _extract_top1_context(self, retrieval: Dict, idx: int) -> str:
        """
        提取 Top-1 context
        
        Args:
            retrieval: 检索结果字典
            idx: 样本索引（用于调试）
        
        Returns:
            提取的文档内容
        """
        top1_ctx = ""
        
        if retrieval.get('topk') and len(retrieval['topk']) > 0:
            raw_text = retrieval['topk'][0].get('text', '')
            
            # 解析 question_aware 的特殊格式
            top1_ctx = self._extract_document_text(raw_text)
            
            if self.verbose and idx == 0:
                print(f"\n  🔍 Sample 0 - Top-1 Context:")
                print(f"     Total contexts available: {len(retrieval['topk'])}")
                print(f"     Using: topk[0] (Top-1 only)")
                print(f"     Raw text preview: {raw_text[:200]}...")
                print(f"     Extracted context length: {len(top1_ctx)} chars")
                print(f"     Extracted context preview: {top1_ctx[:150]}...")
        
        return top1_ctx
    
    @staticmethod
    def _extract_document_text(raw_text: str) -> str:
        """
        从 question_aware 格式中提取 Document 部分
        
        格式示例：
        "Question: when was the last time anyone was on the moon
         Document: Space technology | ... December 1972 ..."
        
        需要提取 "Document:" 之后的部分
        
        Args:
            raw_text: 原始文本
        
        Returns:
            提取的文档内容
        """
        # 方法 1: 使用正则表达式提取 Document: 之后的内容
        match = re.search(r'Document:\s*(.*)', raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 方法 2: 如果没有 Document: 标记，检查是否有 Question: 标记
        if 'Question:' in raw_text:
            # 分割并取第二部分
            parts = raw_text.split('Document:', 1)
            if len(parts) > 1:
                return parts[1].strip()
            # 如果只有 Question，去掉它
            parts = raw_text.split('\n', 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        # 方法 3: 如果都没有，直接返回原文
        return raw_text.strip()
    
    @staticmethod
    def _load_jsonl(file_path: Path) -> List[Dict]:
        """读取 JSONL 文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Line {line_num}: {e}")
        return data