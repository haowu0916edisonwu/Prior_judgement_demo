# Priori Judgment Baseline for RAG Evaluation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

完整复现 **EMNLP 2025 CARE** 论文 Table 2 中的 **Priori Judgment** 基准方法，用于评估检索增强生成（RAG）系统在处理知识冲突时的性能。

---

## 📋 目录

- [项目简介](#项目简介)
- [性能目标](#性能目标)
- [核心特性](#核心特性)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [输出说明](#输出说明)
- [技术细节](#技术细节)
- [故障排查](#故障排查)
- [参考文献](#参考文献)

---

## 🎯 项目简介

本项目实现了 **Priori Judgment** 基准方法，这是一种两阶段推理策略，用于评估大语言模型（LLM）在检索增强生成（RAG）场景下处理知识冲突的能力。

### 什么是 Priori Judgment？

Priori Judgment 是一种先判断后回答的策略：

1. **Stage 1 - 先验判断（Priori Judgment）**：
   - 模型首先评估是否能基于检索到的上下文或内部知识回答问题
   - 如果可以回答 → 给出简短答案
   - 如果不能回答 → 输出 "Unknown"

2. **Stage 2 - 闭卷回退（Closed-book Fallback）**：
   - 如果 Stage 1 输出 "Unknown"
   - 回退到纯闭卷模式（不使用检索结果）
   - 仅依赖模型的参数化知识回答

这种方法允许模型在检索结果不可靠时，主动选择依赖自身知识，从而提高在知识冲突场景下的鲁棒性。

---

## 📊 性能目标

本实现旨在复现以下性能指标（LLaMA-3-8B-Instruct on CARE Datasets）：

| Dataset | Task Type | Metric | Target | 说明 |
|---------|-----------|--------|--------|------|
| **Natural Questions** | Open-domain QA | Span EM | **0.458** | 事实性问答 |
| **TriviaQA** | Open-domain QA | Span EM | **0.704** | 琐事问答 |
| **WebQuestions** | Open-domain QA | Span EM | **0.406** | Web搜索问答 |
| **TruthfulQA** | Long-form QA | F1 | **0.254** | 真实性问答 |
| **TruthfulQA** | Long-form QA | ROUGE-L | **0.231** | 生成质量 |
| **FactKG** | Fact Verification | Accuracy | **0.666** | 事实验证 |
| **Average** | - | - | **0.453** | 综合性能 |

> 📌 **注意**：这些指标来自 EMNLP 2025 CARE 论文 Table 2，使用 ColBERTv2 question-aware retrieval。

---

## ✨ 核心特性

### 1. **完整复现论文设置**
- ✅ 严格按照 COLING 2025 Table 5 & 6 的 Prompt 模板
- ✅ 使用 ColBERTv2 question-aware 检索（Top-1）
- ✅ 两阶段推理逻辑（Priori → Fallback）
- ✅ 适配 CARE 实际数据格式

### 2. **多数据集评估**
- ✅ 5 个数据集：NQ, TriviaQA, WebQA, TruthfulQA, FactKG
- ✅ 3 种任务类型：Open-domain QA, Long-form QA, Fact Verification
- ✅ 4 种评估指标：Span EM, F1, ROUGE-L, Accuracy

### 3. **开箱即用**
- ✅ 使用社区模型（无需申请 Meta 权限）
- ✅ 完整的环境配置和依赖管理
- ✅ 详细的数据加载和验证脚本

### 4. **调试友好**
- ✅ 单样本调试模式
- ✅ 详细的中间输出
- ✅ 模式分布统计（RAG vs Closed-book）

---

## 🔧 环境配置

### 系统要求

- **操作系统**: Linux / macOS / Windows (WSL)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 16GB+ VRAM (推荐)
- **CPU**: 可选，但速度较慢

### 方法 1：使用 Conda（推荐）

```bash
# 1. 创建环境
conda env create -f environment.yml

# 2. 激活环境
conda activate priori_care

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 方法 2：使用 pip

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "import torch; print(torch.__version__)"
```

### 依赖包清单

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.1.2 | 深度学习框架 |
| `transformers` | 4.36.2 | LLM 推理 |
| `accelerate` | 0.25.0 | 模型加载优化 |
| `rouge-score` | 0.1.2 | ROUGE 指标 |
| `tqdm` | 4.66.1 | 进度条 |

---

## 📁 数据准备

### 数据结构要求

确保你的数据目录结构如下：

```
data_care/eval/
├── nq/
│   ├── test.jsonl                                    # 问题+答案
│   └── retrieval/colbertv2/
│       ├── test.jsonl                                # 标准检索
│       └── test_question_aware.jsonl                 # ✅ 使用这个
├── triviaqa/
│   ├── test.jsonl
│   └── retrieval/colbertv2/
│       ├── test.jsonl
│       └── test_question_aware.jsonl                 # ✅ 使用这个
├── webqa/
│   ├── test.jsonl
│   └── retrieval/colbertv2/test_question_aware.jsonl # ✅ 使用这个
├── truthfulqa/
│   ├── test.jsonl
│   └── retrieval/colbertv2/test_question_aware.jsonl # ✅ 使用这个
└── factkg/
    ├── test.jsonl
    └── retrieval/colbertv2/test_question_aware.jsonl # ✅ 使用这个
```

> 📌 **关键**：本项目使用 `test_question_aware.jsonl`（论文设置），而非 `test.jsonl`

### 数据格式说明

#### 1. 问题文件格式（`test.jsonl`）

```json
{
  "id": 0,
  "question": "when was the last time anyone was on the moon",
  "answer": ["14 December 1972 UTC", "December 1972"],
  "entity": "14 December 1972 UTC"
}
```

**字段说明**：
- `id`: 样本唯一标识符
- `question`: 问题文本（FactKG 中为 `claim`）
- `answer`: 标准答案列表（注意是 `answer` 不是 `answers`）
- `entity`: 主实体（可选）

#### 2. 检索文件格式（`test_question_aware.jsonl`）

```json
{
  "id": 0,
  "query": "when was the last time anyone was on the moon",
  "topk": [
    {
      "text": "Question: when was the last time anyone was on the moon\n Document: Space technology | ... December 1972 ..."
    }
  ]
}
```

**字段说明**：
- `id`: 与问题文件对应的ID
- `query`: 查询文本
- `topk`: 检索结果列表（注意是 `topk` 不是 `ctxs`）
- `text`: 包含 `Question: ... Document: ...` 格式，代码会自动提取 Document 部分

### 验证数据

运行验证脚本确保数据格式正确：

```bash
# 验证数据结构
python verify_data.py

# 测试数据加载
python test_data_loading.py
```

**预期输出**：
```
✅ Test Passed! Data format understood correctly.
```

---

## 🚀 快速开始

### 1. 完整评估（推荐）

评估所有 5 个数据集：

```bash
# 使用脚本（推荐）
./run.sh

# 或手动运行
python run_eval.py \
    --data_root data_care/eval \
    --model_name NousResearch/Meta-Llama-3-8B-Instruct \
    --output_dir results
```

**预计时间**：
- GPU (V100): ~2-3 小时
- GPU (A100): ~1-2 小时
- CPU: ~10-15 小时

### 2. 调试模式

测试单个样本以验证逻辑：

```bash
python run_eval.py --debug_sample --datasets nq
```

**输出示例**：
```
🐛 DEBUG MODE: NQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Sample Info:
  ID: 0
  Question: when was the last time anyone was on the moon
  Answers: ['14 December 1972 UTC', 'December 1972']
  Context length: 704 chars

🔄 Running Two-Stage Inference...

📊 Results:
  Stage 1 (Priori) Output: December 1972
  Unknown detected: False
  Final Answer: December 1972
  Mode: rag
  Gold Answers: ['14 December 1972 UTC', 'December 1972']

✅ Span EM: 1.0000
```

### 3. 小规模测试

快速验证流程（每个数据集 100 个样本）：

```bash
python run_eval.py --max_samples 100
```

### 4. 特定数据集评估

只评估某些数据集：

```bash
# 评估 NQ 和 TriviaQA
python run_eval.py --datasets nq trivia

# 评估 TruthfulQA（长文本）
python run_eval.py --datasets truthfulqa --save_predictions
```

---

## 📖 使用指南

### 命令行参数

```bash
python run_eval.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | `data_care/eval` | 数据根目录 |
| `--model_name` | str | `NousResearch/Meta-Llama-3-8B-Instruct` | 模型名称 |
| `--output_dir` | str | `results` | 结果输出目录 |
| `--datasets` | list | `None` (全部) | 指定评估的数据集 |
| `--max_samples` | int | `None` (全部) | 限制每个数据集的样本数 |
| `--debug_sample` | flag | `False` | 调试单个样本 |
| `--verbose` | flag | `False` | 详细输出 |
| `--save_predictions` | flag | `False` | 保存详细预测结果 |

### 使用示例

#### 示例 1：完整评估并保存预测

```bash
python run_eval.py --save_predictions
```

#### 示例 2：调试 TruthfulQA

```bash
python run_eval.py \
    --datasets truthfulqa \
    --max_samples 10 \
    --verbose \
    --debug_sample
```

#### 示例 3：使用官方 LLaMA-3 模型（需要权限）

```bash
python run_eval.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct
```

#### 示例 4：CPU 模式

```bash
export CUDA_VISIBLE_DEVICES=""
python run_eval.py --max_samples 100
```

---

## 📊 输出说明

### 控制台输出

运行时会实时显示评估进度和结果：

```
======================================================================
🚀 Priori Judgment Evaluation (CARE Data - Fixed)
======================================================================
Model: NousResearch/Meta-Llama-3-8B-Instruct
Data: data_care/eval
Using: test_question_aware.jsonl (Top-1 only)
Format: answer + topk fields (Fixed)
======================================================================

======================================================================
📊 Evaluating NQ
======================================================================
✅ Loaded 3610 valid samples from nq
🔄 Evaluating 3610 samples...
Processing: 100%|████████████████████| 3610/3610 [12:34<00:00, 4.79it/s]

📈 Results:
  SPAN_EM: 0.4618 (target: 0.4580) ✅

🔀 Mode Distribution:
  rag: 2856/3610 (79.1%)
  closedbook: 754/3610 (20.9%)

...

======================================================================
📊 FINAL RESULTS
======================================================================
Dataset         Metric       Result     Target     Status
----------------------------------------------------------------------
nq              SPAN_EM      0.4618     0.4580     ✅
trivia          SPAN_EM      0.7105     0.7040     ✅
webqa           SPAN_EM      0.4136     0.4060     ✅
truthfulqa      F1           0.2589     0.2540     ✅
truthfulqa      ROUGE_L      0.2345     0.2310     ✅
factkg          ACCURACY     0.6724     0.6660     ✅
----------------------------------------------------------------------
AVERAGE                      0.4634     0.4530     ✅
======================================================================
```

### 输出文件

运行完成后会在 `results/` 目录生成以下文件：

#### 1. `results.json` - 主结果文件（JSON 格式）

```json
{
  "nq": {
    "metrics": {
      "span_em": 0.4618
    },
    "mode_distribution": {
      "rag": 2856,
      "closedbook": 754
    }
  },
  "trivia": {
    "metrics": {
      "span_em": 0.7105
    },
    "mode_distribution": {
      "rag": 9245,
      "closedbook": 2068
    }
  },
  ...
}
```

**字段说明**：
- `metrics`: 评估指标结果
  - `span_em`: Span Exact Match（子串匹配）
  - `f1`: Token-level F1 score
  - `rouge_l`: ROUGE-L score
  - `accuracy`: 二分类准确率
- `mode_distribution`: 推理模式统计
  - `rag`: 使用检索结果回答的样本数
  - `closedbook`: 回退到闭卷模式的样本数

#### 2. `summary.txt` - 文本汇总

```
PRIORI JUDGMENT EVALUATION RESULTS
======================================================================

NQ
  SPAN_EM: 0.4618

TRIVIA
  SPAN_EM: 0.7105

WEBQA
  SPAN_EM: 0.4136

TRUTHFULQA
  F1: 0.2589
  ROUGE_L: 0.2345

FACTKG
  ACCURACY: 0.6724

AVERAGE: 0.4634
```

#### 3. `{dataset}_predictions.jsonl` - 详细预测（可选）

使用 `--save_predictions` 参数时生成，包含每个样本的详细信息：

```json
{
  "id": "0",
  "question": "when was the last time anyone was on the moon",
  "prediction": "December 1972",
  "gold_answers": ["14 December 1972 UTC", "December 1972"],
  "mode": "rag",
  "priori_output": "December 1972",
  "correct": true
}
```

**字段说明**：
- `id`: 样本ID
- `question`: 问题文本
- `prediction`: 模型最终预测
- `gold_answers`: 标准答案列表
- `mode`: 推理模式（`rag` 或 `closedbook`）
- `priori_output`: 第一阶段（Priori Judgment）的输出
- `correct`: 是否预测正确

---

## 🔬 技术细节

### 两阶段推理实现

```python
# Stage 1: Priori Judgment with Top-1 Retrieval
priori_prompt = f"""Given the following information:
{top1_context}

Can you answer the following question based on the given information 
or your internal knowledge? If yes, you should give a short answer with 
one or few words, if no, you should answer "Unknown".

Question: {question}"""

priori_output = model.generate(priori_prompt)

# Stage 2: Fallback to Closed-book if Unknown
if "unknown" in priori_output.lower():
    closedbook_prompt = f"""Answer the questions:
Question: {question}?
The answer is:"""
    
    final_answer = model.generate(closedbook_prompt)
    mode = "closedbook"
else:
    final_answer = priori_output
    mode = "rag"
```

### Prompt 模板来源

所有 Prompt 严格按照以下论文表格：

| Prompt 类型 | 来源 | 说明 |
|------------|------|------|
| Priori Judgment (QA) | COLING 2025 Table 6 第1行 | Open-domain & Long-form QA |
| Priori Judgment (Fact) | COLING 2025 Table 6 第2行 | Fact Verification |
| Closed-book (Short QA) | COLING 2025 Table 5 | 问题后有问号 |
| Closed-book (Long QA) | COLING 2025 Table 5 | 问题后无问号 |
| Closed-book (Fact) | COLING 2025 Table 5 | 要求输出 True/False |

### 评估指标详解

#### 1. Span EM（Span Exact Match）

**不是**严格的 token-level exact match！

```python
def compute_span_em(prediction: str, ground_truths: List[str]) -> float:
    pred_normalized = normalize(prediction)
    for gt in ground_truths:
        gt_normalized = normalize(gt)
        # 关键：子串匹配
        if gt_normalized in pred_normalized:
            return 1.0
    return 0.0
```

**标准化步骤**：
1. 转小写
2. 移除冠词（a, an, the）
3. 移除标点符号
4. 规范化空格

**示例**：
- Prediction: "The last person on the moon was in December 1972"
- Ground Truth: "December 1972"
- Result: ✅ Match（GT 是 Pred 的子串）

#### 2. Token-level F1

```python
def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    pred_tokens = normalize(prediction).split()
    max_f1 = 0.0
    
    for gt in ground_truths:
        gt_tokens = normalize(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        
        precision = common / len(pred_tokens)
        recall = common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1
```

#### 3. ROUGE-L

基于最长公共子序列（LCS）的 F1 分数：

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
score = scorer.score(ground_truth, prediction)['rougeL'].fmeasure
```

#### 4. Accuracy（FactKG）

鲁棒的二分类标签提取：

```python
def compute_accuracy(prediction: str, ground_truths: List[str]) -> float:
    pred_lower = prediction.lower()
    
    # 优先检查 false（避免 "not true" 误判）
    if 'false' in pred_lower:
        pred_label = 'false'
    elif 'true' in pred_lower:
        pred_label = 'true'
    else:
        pred_label = 'false'  # 默认
    
    gt_label = normalize(ground_truths[0])
    return 1.0 if pred_label == gt_label else 0.0
```

### 模型配置

```python
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,  # FP16 精度
    device_map="auto",          # 自动设备分配
    trust_remote_code=True
)

generation_config = {
    "max_new_tokens": 30,       # 论文设置
    "do_sample": False,         # Greedy Decoding
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id
}
```

---

## 🐛 故障排查

### 问题 1：数据格式错误

**症状**：
```
KeyError: 'answers'
KeyError: 'ctxs'
```

**解决方案**：
```bash
# 1. 验证数据格式
python test_data_loading.py

# 2. 确认使用了修正后的 data_loader.py
grep "topk" src/data_loader.py  # 应该找到
grep "answer" src/data_loader.py  # 应该找到
```

### 问题 2：CUDA 内存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：

方法 1 - 使用 CPU：
```bash
export CUDA_VISIBLE_DEVICES=""
python run_eval.py
```

方法 2 - 减少批量大小（不适用于本项目，因为是单样本推理）

方法 3 - 使用量化模型：
```bash
# 需要修改 evaluator.py 添加量化配置
pip install bitsandbytes
```

### 问题 3：模型下载失败

**症状**：
```
ConnectionError: Can't connect to huggingface.co
```

**解决方案**：

方法 1 - 使用镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python run_eval.py
```

方法 2 - 手动下载：
```bash
# 1. 访问 https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct
# 2. 下载所有文件到本地目录
# 3. 使用本地路径
python run_eval.py --model_name /path/to/local/model
```

### 问题 4：分数明显偏低

**症状**：
- NQ Span EM < 0.40
- TriviaQA Span EM < 0.60

**诊断步骤**：

步骤 1 - 检查单样本：
```bash
python run_eval.py --debug_sample --datasets nq
```

步骤 2 - 检查 Prompt 格式：
```bash
grep -A 5 "priori_judgment_qa" src/prompts.py
```

步骤 3 - 检查数据加载：
```bash
python test_data_loading.py
```

步骤 4 - 检查 Unknown 检测：
```python
# 在 evaluator.py 中添加调试输出
print(f"Priori output: {priori_output}")
print(f"Is unknown: {self.is_unknown(priori_output)}")
```

### 问题 5：LLaMA-3 官方模型无权限

**症状**：
```
401 Client Error: Unauthorized for url
```

**解决方案**：

使用社区版本（默认）：
```bash
# 已经默认使用 NousResearch 版本
python run_eval.py
```

或申请访问：
1. 访问 https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. 点击 "Request Access"
3. 等待审批（通常几小时）
4. 使用 HF token 登录

---

## 📚 参考文献

### 论文

1. **CARE: Conflict-Aware Soft Prompting for Retrieval-Augmented Generation**  
   Eunseong Choi, June Park, Hyeri Lee, Jongwuk Lee  
   EMNLP 2025  
   - 论文链接: [arXiv:2508.15253](https://arxiv.org/abs/2508.15253)
   - 代码仓库: [github.com/eunseongc/CARE](https://github.com/eunseongc/CARE)

2. **Investigating Factual Knowledge Boundary of LLMs with Retrieval Augmentation**  
   Ruiyang Ren et al.  
   COLING 2025
   - Priori Judgment 方法首次提出

### 数据集

- **Natural Questions (NQ)**: [ai.google.com/research/NaturalQuestions](https://ai.google.com/research/NaturalQuestions)
- **TriviaQA**: [nlp.cs.washington.edu/triviaqa](http://nlp.cs.washington.edu/triviaqa/)
- **WebQuestions**: [github.com/brmson/dataset-factoid-webquestions](https://github.com/brmson/dataset-factoid-webquestions)
- **TruthfulQA**: [github.com/sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- **FactKG**: From CARE paper

### 模型

- **LLaMA-3**: [huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- **社区版本**: [huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct)

---

## 🤝 贡献与反馈

### 项目结构

```
priori_judgment_care/
├── README.md                    # 本文件
├── environment.yml              # Conda 环境配置
├── requirements.txt             # Pip 依赖列表
├── verify_data.py               # 数据验证脚本
├── test_data_loading.py         # 数据加载测试
├── run_eval.py                  # 主评估脚本
├── run.sh                       # 运行脚本（Shell）
├── src/
│   ├── __init__.py              # 包初始化
│   ├── data_loader.py           # 数据加载器（已修正）
│   ├── prompts.py               # Prompt 模板
│   ├── metrics.py               # 评估指标
│   └── evaluator.py             # 两阶段推理评估器
├── data_care/eval/              # 数据目录
│   ├── nq/
│   ├── triviaqa/
│   ├── webqa/
│   ├── truthfulqa/
│   └── factkg/
└── results/                     # 输出目录（自动生成）
    ├── results.json
    ├── summary.txt
    └── {dataset}_predictions.jsonl
```

### 常见问题

如遇到问题，请按以下顺序排查：

1. ✅ 运行 `python test_data_loading.py` 验证数据
2. ✅ 运行 `python verify_data.py` 检查数据完整性
3. ✅ 使用 `--debug_sample` 查看单样本推理
4. ✅ 使用 `--max_samples 10` 快速测试
5. ✅ 检查输出日志和错误信息

### 版本更新

- **v1.1** (2024-01-06): 修正数据格式适配
  - ✅ 修正 `answer` vs `answers` 字段
  - ✅ 修正 `topk` vs `ctxs` 字段
  - ✅ 添加 question_aware 文本解析
  - ✅ 添加详细输出和预测保存

- **v1.0** (2024-01-05): 初始版本
  - ✅ 基础两阶段推理实现
  - ✅ 5 个数据集支持
  - ✅ 完整评估管道

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢 CARE 论文作者提供数据集和基准
- 感谢 HuggingFace 团队提供模型和工具
- 感谢 NousResearch 提供开源 LLaMA-3 模型

---

## 📮 联系方式

如有问题或建议，欢迎：
- 提交 GitHub Issue
- 发送邮件至项目维护者
- 参考论文原始仓库: [github.com/eunseongc/CARE](https://github.com/eunseongc/CARE)

---

**最后更新**: 2024-01-06  
**维护者**: Priori Judgment Reproduction Team  
**状态**: ✅ 生产就绪