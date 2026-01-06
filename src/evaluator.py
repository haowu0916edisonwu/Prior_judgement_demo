"""
Priori Judgment 评估器 - 两阶段推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from tqdm import tqdm
from dataclasses import dataclass

from .prompts import PromptTemplates
from .metrics import Metrics
from .data_loader import Sample


@dataclass
class EvalResult:
    """评估结果"""
    id: str
    question: str
    prediction: str
    gold_answers: List[str]
    mode: str  # "rag" or "closedbook"
    priori_output: str


class PrioriJudgmentEvaluator:
    """
    两阶段推理评估器
    
    实现论文的 Priori Judgment baseline：
    Stage 1: Priori Judgment with Top-1 retrieval
    Stage 2: Fallback to Closed-book if "Unknown"
    """
    
    # 任务类型映射
    TASK_TYPES = {
        'nq': 'open_qa',
        'trivia': 'open_qa',
        'triviaqa': 'open_qa',
        'webqa': 'open_qa',
        'truthfulqa': 'long_form',
        'factkg': 'fact_checking'
    }
    
    def __init__(
        self,
        model_name: str = "NousResearch/Meta-Llama-3-8B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 30
    ):
        """
        初始化评估器
        
        Args:
            model_name: HuggingFace 模型名称或路径
            device: 设备 (cuda/cpu)
            max_new_tokens: 最大生成 token 数（论文设置为 30）
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        
        print(f"🔧 Loading model: {model_name}")
        print(f"   Device: {self.device}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"  # 重要：左侧填充用于生成
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 使用 FP16 节省显存
            device_map="auto",          # 自动设备映射
            trust_remote_code=True,
            low_cpu_mem_usage=True      # 优化 CPU 内存使用
        )
        self.model.eval()
        
        print(f"✅ Model loaded")
    
    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        生成文本
        
        使用 Greedy Decoding（temperature=0，无采样）
        确保结果可复现
        
        Args:
            prompt: 输入 prompt
        
        Returns:
            生成的文本（不包含 prompt）
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Greedy decoding（关键）
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode（只返回新生成的部分）
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def is_unknown(self, text: str) -> bool:
        """
        检测是否包含 "Unknown" 拒绝信号
        
        鲁棒处理多种拒绝回答的模式：
        - "Unknown"
        - "I don't know"
        - "I'm not sure"
        - "Cannot answer"
        - 等等
        
        Args:
            text: 模型输出文本
        
        Returns:
            True 如果检测到拒绝信号
        """
        text_lower = text.lower().strip()
        
        # 1. 精确匹配
        if text_lower == "unknown":
            return True
        
        # 2. 开头匹配
        if text_lower.startswith("unknown"):
            return True
        
        # 3. 常见拒绝模式
        unknown_patterns = [
            "unknown",
            "i don't know",
            "i do not know",
            "i'm not sure",
            "i am not sure",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "no information",
            "not enough information"
        ]
        
        for pattern in unknown_patterns:
            if pattern in text_lower:
                return True
        
        return False
    
    def evaluate_sample(self, sample: Sample) -> EvalResult:
        """
        评估单个样本（两阶段推理）
        
        Stage 1: Priori Judgment
        - 使用 Top-1 检索上下文
        - 判断是否能回答
        
        Stage 2: Fallback (if needed)
        - 如果输出 "Unknown"，回退到 Closed-book
        
        Args:
            sample: 评估样本
        
        Returns:
            评估结果
        """
        task_type = self.TASK_TYPES[sample.dataset]
        
        # === Stage 1: Priori Judgment ===
        if task_type == "fact_checking":
            priori_prompt = PromptTemplates.priori_judgment_fact(
                sample.question, sample.top1_context
            )
        else:
            priori_prompt = PromptTemplates.priori_judgment_qa(
                sample.question, sample.top1_context
            )
        
        priori_output = self.generate(priori_prompt)
        
        # === Stage 2: Check Unknown & Fallback ===
        if self.is_unknown(priori_output):
            # Fallback to closed-book
            if task_type == "fact_checking":
                cb_prompt = PromptTemplates.closedbook_fact(sample.question)
            elif task_type == "long_form":
                cb_prompt = PromptTemplates.closedbook_qa_long(sample.question)
            else:
                cb_prompt = PromptTemplates.closedbook_qa_short(sample.question)
            
            final_answer = self.generate(cb_prompt)
            mode = "closedbook"
        else:
            # 使用 priori output 直接作为答案
            final_answer = priori_output
            mode = "rag"
        
        return EvalResult(
            id=sample.id,
            question=sample.question,
            prediction=final_answer,
            gold_answers=sample.answers,
            mode=mode,
            priori_output=priori_output
        )
    
    def evaluate_dataset(self, samples: List[Sample]) -> Dict:
        """
        评估整个数据集
        
        Args:
            samples: 样本列表
        
        Returns:
            包含 metrics, results, mode_distribution 的字典
        """
        results = []
        
        print(f"🔄 Evaluating {len(samples)} samples...")
        for sample in tqdm(samples, desc="Processing"):
            result = self.evaluate_sample(sample)
            results.append(result)
        
        # 计算指标
        task_type = self.TASK_TYPES[samples[0].dataset]
        metrics = self._compute_metrics(results, task_type)
        
        # 统计模式分布
        mode_counts = {'rag': 0, 'closedbook': 0}
        for r in results:
            mode_counts[r.mode] += 1
        
        return {
            'metrics': metrics,
            'results': results,
            'mode_distribution': mode_counts
        }
    
    def _compute_metrics(
        self,
        results: List[EvalResult],
        task_type: str
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        根据任务类型选择合适的指标：
        - fact_checking: Accuracy
        - long_form: F1 + ROUGE-L
        - open_qa: Span EM
        
        Args:
            results: 评估结果列表
            task_type: 任务类型
        
        Returns:
            指标字典
        """
        if task_type == "fact_checking":
            # FactKG: Accuracy
            scores = [
                Metrics.compute_accuracy(r.prediction, r.gold_answers)
                for r in results
            ]
            return {'accuracy': sum(scores) / len(scores)}
        
        elif task_type == "long_form":
            # TruthfulQA: F1 + ROUGE-L（重要：论文报告了两个指标）
            f1_scores = [
                Metrics.compute_f1(r.prediction, r.gold_answers)
                for r in results
            ]
            rouge_scores = [
                Metrics.compute_rouge_l(r.prediction, r.gold_answers)
                for r in results
            ]
            return {
                'f1': sum(f1_scores) / len(f1_scores),
                'rouge_l': sum(rouge_scores) / len(rouge_scores)
            }
        
        else:
            # Open-domain QA: Span EM
            scores = [
                Metrics.compute_span_em(r.prediction, r.gold_answers)
                for r in results
            ]
            return {'span_em': sum(scores) / len(scores)}