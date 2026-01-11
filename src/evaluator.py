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
    id: str
    question: str
    prediction: str
    gold_answers: List[str]
    mode: str
    priori_output: str

class PrioriJudgmentEvaluator:
    
    TASK_TYPES = {
        'nq': 'open_qa',
        'trivia': 'open_qa',
        'triviaqa': 'open_qa',
        'webqa': 'open_qa',
        'truthfulqa': 'long_form',
        'factkg': 'fact_checking'
    }

    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 30):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        
        print(f"🔧 Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
        )
        self.model.eval()
        print(f"✅ Model loaded")

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, 
            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def is_unknown(self, text: str) -> bool:
        """严格的 Unknown 检测，防止误判"""
        text_lower = text.lower().strip()
        if text_lower in ["unknown", "unknown."]: return True
        if text_lower.startswith("unknown"): return True
        refusal_starts = ["i don't know", "i do not know", "i'm not sure", "cannot answer", "unable to answer", "no information"]
        for pattern in refusal_starts:
            if text_lower.startswith(pattern): return True
        return False

    def clean_output(self, text: str) -> str:
        """清洗 Llama-3 的废话前缀"""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        if text_lower.startswith("answer:"):
            text_clean = text_clean[7:].strip()
        elif text_lower.startswith("prediction:"):
            text_clean = text_clean[11:].strip()
        # 处理 FactKG 可能出现的 "The claim is True"
        if text_clean.lower().startswith("the claim is"):
            text_clean = text_clean[12:].strip()
        return text_clean

    def evaluate_sample(self, sample: Sample) -> EvalResult:
        task_type = self.TASK_TYPES.get(sample.dataset, 'open_qa')
        
        # === Stage 1: Priori Judgment (RAG Setting) ===
        if task_type == "fact_checking":
            # [修改 1] 将 sample.top1_context 改为 sample.context
            priori_prompt = PromptTemplates.priori_judgment_fact(sample.question, sample.context)
            
        elif task_type == "long_form":
            # [修改 2] 修复 TruthfulQA 逻辑：
            # (1) 传入 sample.context (之前漏传了)
            # (2) 变量名更新为 sample.context
            priori_prompt = PromptTemplates.priori_judgment_truthful(sample.question, sample.context)
            
        else:
            # [修改 3] 将 sample.top1_context 改为 sample.context
            priori_prompt = PromptTemplates.priori_judgment_qa(sample.question, sample.context)
        
        raw_priori_output = self.generate(priori_prompt)
        cleaned_priori_output = self.clean_output(raw_priori_output)
        
        # === Stage 2: Check Unknown & Fallback ===
        if self.is_unknown(cleaned_priori_output):
            # Fallback to Closed-book
            if task_type == "fact_checking":
                cb_prompt = PromptTemplates.closedbook_fact(sample.question)
            elif task_type == "long_form":
                cb_prompt = PromptTemplates.closedbook_qa_long(sample.question)
            else:
                cb_prompt = PromptTemplates.closedbook_qa_short(sample.question)
            
            final_answer = self.clean_output(self.generate(cb_prompt))
            mode = "closedbook"
        else:
            final_answer = cleaned_priori_output
            mode = "rag"
            
        return EvalResult(
            id=sample.id, question=sample.question, prediction=final_answer,
            gold_answers=sample.answers, mode=mode, priori_output=raw_priori_output
        )

    def evaluate_dataset(self, samples: List[Sample]) -> Dict:
        results = []
        print(f"🔄 Evaluating {len(samples)} samples...")
        for sample in tqdm(samples, desc="Processing"):
            results.append(self.evaluate_sample(sample))
            
        task_type = self.TASK_TYPES.get(samples[0].dataset, 'open_qa')
        metrics = self._compute_metrics(results, task_type)
        
        mode_counts = {'rag': 0, 'closedbook': 0}
        for r in results:
            mode_counts[r.mode] += 1
            
        return {'metrics': metrics, 'results': results, 'mode_distribution': mode_counts}

    def _compute_metrics(self, results: List[EvalResult], task_type: str) -> Dict[str, float]:
        if task_type == "fact_checking":
            scores = [Metrics.compute_accuracy(r.prediction, r.gold_answers) for r in results]
            return {'accuracy': sum(scores) / len(scores) if scores else 0}
        elif task_type == "long_form":
            f1_scores = [Metrics.compute_f1(r.prediction, r.gold_answers) for r in results]
            rouge_scores = [Metrics.compute_rouge_l(r.prediction, r.gold_answers) for r in results]
            return {
                'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
                'rouge_l': sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
            }
        else:
            scores = [Metrics.compute_span_em(r.prediction, r.gold_answers) for r in results]
            return {'span_em': sum(scores) / len(scores) if scores else 0}
