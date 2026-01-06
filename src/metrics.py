"""
评估指标 - 完整实现
"""

import re
import string
from typing import List
from collections import Counter


class Metrics:
    """评估指标类"""
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """
        标准化答案（DPR/NQ 风格）
        
        步骤：
        1. 转小写
        2. 移除冠词 (a, an, the)
        3. 移除标点
        4. 规范化空格
        """
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @staticmethod
    def compute_span_em(prediction: str, ground_truths: List[str]) -> float:
        """
        计算 Span EM（子串匹配）
        
        关键：这不是严格的 token-level EM！
        而是检查标准化后的 ground truth 是否是 prediction 的子串
        
        Args:
            prediction: 模型预测文本
            ground_truths: 标准答案列表
        
        Returns:
            1.0 如果任一答案是预测的子串，否则 0.0
        """
        pred_normalized = Metrics.normalize_answer(prediction)
        
        for gt in ground_truths:
            gt_normalized = Metrics.normalize_answer(gt)
            
            # 关键：使用子串匹配而非完全匹配
            if gt_normalized and gt_normalized in pred_normalized:
                return 1.0
        
        return 0.0
    
    @staticmethod
    def compute_f1(prediction: str, ground_truths: List[str]) -> float:
        """
        计算 Token-level F1 score
        
        采用 SQuAD 风格的 F1 计算：
        1. 将文本分词
        2. 计算 token 集合的交集
        3. 计算 precision 和 recall
        4. 返回 F1 = 2PR/(P+R)
        
        对多个 ground truth 取最大值
        
        Args:
            prediction: 模型预测文本
            ground_truths: 标准答案列表
        
        Returns:
            最大的 F1 分数
        """
        def get_tokens(s):
            """分词并标准化"""
            return Metrics.normalize_answer(s).split()
        
        pred_tokens = get_tokens(prediction)
        if not pred_tokens:
            return 0.0
        
        max_f1 = 0.0
        
        for gt in ground_truths:
            gt_tokens = get_tokens(gt)
            if not gt_tokens:
                continue
            
            # 使用 Counter 计算交集（处理重复 token）
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                continue
            
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            
            max_f1 = max(max_f1, f1)
        
        return max_f1
    
    @staticmethod
    def compute_rouge_l(prediction: str, ground_truths: List[str]) -> float:
        """
        计算 ROUGE-L score
        
        ROUGE-L 基于 Longest Common Subsequence (LCS)
        用于 TruthfulQA 的评估
        
        Args:
            prediction: 模型预测文本
            ground_truths: 标准答案列表
        
        Returns:
            最大的 ROUGE-L F1 分数
        """
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            max_score = 0.0
            for gt in ground_truths:
                # rouge_scorer.score(reference, hypothesis)
                score = scorer.score(gt, prediction)['rougeL'].fmeasure
                max_score = max(max_score, score)
            
            return max_score
        
        except ImportError:
            # Fallback：如果 rouge-score 未安装，使用 F1
            print("⚠️  rouge-score not installed, using F1 as approximation")
            return Metrics.compute_f1(prediction, ground_truths)
    
    @staticmethod
    def compute_accuracy(prediction: str, ground_truths: List[str]) -> float:
        """
        计算二分类准确率（用于 FactKG）
        
        从预测文本中鲁棒地提取 "true" 或 "false" 标签
        
        处理的情况：
        1. 直接输出 "True" 或 "False"
        2. 输出 "The answer is True"
        3. 输出 "True. The claim is correct."
        4. 其他变体
        
        Args:
            prediction: 模型预测文本
            ground_truths: 标准答案（应该是 ["True"] 或 ["False"]）
        
        Returns:
            1.0 如果预测正确，否则 0.0
        """
        pred_lower = prediction.lower().strip()
        
        # 提取预测标签（增强鲁棒性）
        # 优先检查 "false"（因为可能出现 "not true" 等情况）
        if 'false' in pred_lower:
            pred_label = 'false'
        elif 'true' in pred_lower:
            pred_label = 'true'
        else:
            # 无法识别，从第一个词判断
            words = pred_lower.split()
            if words:
                first_word = words[0]
                pred_label = 'true' if 'true' in first_word else 'false'
            else:
                pred_label = 'false'  # 默认
        
        # 标准化 ground truth
        gt_normalized = Metrics.normalize_answer(ground_truths[0])
        
        return 1.0 if pred_label == gt_normalized else 0.0