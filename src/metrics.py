"""
评估指标 - SOTA 对齐版
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
        这是学术界标准的清洗方案，不要动它。
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
        [增强版] 增加数字/文本互转的 Span EM
        """
        # 简单的数字映射表
        num_map = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
            "10": "ten"
        }
        
        # 归一化预测
        pred_norm = Metrics.normalize_answer(prediction)
        
        # 尝试生成预测的“数字变文字”版本 (例如 "1" -> "one")
        pred_text_version = pred_norm
        for k, v in num_map.items():
            # 这里做个简单的全词替换
            pred_text_version = re.sub(r'\b' + k + r'\b', v, pred_text_version)

        for gt in ground_truths:
            gt_norm = Metrics.normalize_answer(gt)
            
            # 1. 直接匹配
            if gt_norm in pred_norm:
                return 1.0
            
            # 2. 数字转换匹配 (Pred转文字 vs GT)
            if gt_norm in pred_text_version:
                return 1.0
                
            # 3. 反向匹配 (GT转数字 vs Pred)
            # (例如 GT="one", Pred="1")
            gt_digit_version = gt_norm
            for k, v in num_map.items():
                gt_digit_version = re.sub(r'\b' + v + r'\b', k, gt_digit_version)
            
            if gt_digit_version in pred_norm:
                return 1.0
                
        return 0.0
    
    @staticmethod
    def compute_f1(prediction: str, ground_truths: List[str]) -> float:
        """SQuAD 风格 F1 (保持不变)"""
        def get_tokens(s):
            return Metrics.normalize_answer(s).split()
        
        pred_tokens = get_tokens(prediction)
        if not pred_tokens:
            return 0.0
        
        max_f1 = 0.0
        for gt in ground_truths:
            gt_tokens = get_tokens(gt)
            if not gt_tokens: continue
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            if num_same == 0: continue
            
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)
        
        return max_f1
    
    @staticmethod
    def compute_rouge_l(prediction: str, ground_truths: List[str]) -> float:
        """ROUGE-L (保持不变)"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            max_score = 0.0
            for gt in ground_truths:
                score = scorer.score(gt, prediction)['rougeL'].fmeasure
                max_score = max(max_score, score)
            return max_score
        except ImportError:
            print("⚠️ rouge-score not installed, using F1 as approximation")
            return Metrics.compute_f1(prediction, ground_truths)
    
    @staticmethod
    def compute_accuracy(prediction: str, ground_truths: List[str]) -> float:
        """[防弹版] FactKG 准确率计算"""
        # 1. 预处理
        gt_label = ground_truths[0].lower().strip()
        pred_clean = prediction.lower()
        # 去标点 (保留字母和空格)
        pred_clean = ''.join(c for c in pred_clean if c.isalnum() or c.isspace())
        pred_tokens = pred_clean.split()

        # 2. 优先匹配句首 (最强信号)
        if pred_clean.startswith(gt_label):
            return 1.0
            
        # 3. 互斥检测 (防止 "True, not false")
        has_true = 'true' in pred_tokens
        has_false = 'false' in pred_tokens
        
        if gt_label == 'true':
            # 只有在 (有true 且 没有false) 或者 (true 出现在 false 之前) 时才算对
            if has_true and (not has_false or pred_clean.find('true') < pred_clean.find('false')):
                return 1.0
        elif gt_label == 'false':
            if has_false and (not has_true or pred_clean.find('false') < pred_clean.find('true')):
                return 1.0
                
        return 0.0