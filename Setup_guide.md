# Priori Judgment 项目完整部署运行指南

从零开始，一步步教你搭建和运行项目。

---

## 📋 前置要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐 A100/V100/RTX 3090/4090)
- **显存**: 至少 16GB（推荐 40GB+）
- **内存**: 至少 32GB RAM
- **硬盘**: 至少 50GB 可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04/22.04) 或 macOS
- **CUDA**: 11.8+ (如果使用 GPU)
- **Python**: 3.10+
- **Git**: 任意版本

---

## 🚀 完整部署流程

### 第一步：检查环境

```bash
# 1. 检查 Python 版本
python --version
# 应该显示: Python 3.10.x 或更高

# 2. 检查 GPU（如果有）
nvidia-smi
# 应该显示 GPU 信息

# 3. 检查 CUDA（如果有 GPU）
nvcc --version
# 应该显示 CUDA 11.8 或更高
```

---

### 第二步：获取代码

```bash
# 1. 进入工作目录
cd ~  # 或者你想放项目的任何目录

# 2. 克隆仓库
git clone https://github.com/haowu0916edisonwu/Prior_judgement_demo.git

# 3. 进入项目目录
cd Prior_judgement_demo

# 4. 查看文件结构
ls -la
```

**预期输出**:
```
.
├── src/                    # 核心代码
├── README.md              # 项目说明
├── requirements.txt       # Python 依赖
├── environment.yml        # Conda 环境配置
├── run_eval.py           # 主评估脚本
├── run.sh                # 运行脚本
├── test_data_loading.py  # 数据测试
└── verify_data.py        # 数据验证
```

---

### 第三步：创建 Python 环境

#### 方法 1: 使用 Conda（推荐）

```bash
# 1. 创建环境（如果你有 environment.yml）
conda env create -f environment.yml

# 2. 激活环境
conda activate priori_care

# 3. 验证
which python
# 应该显示 conda 环境中的 python 路径
```

#### 方法 2: 使用 venv

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活环境
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate   # Windows

# 3. 验证
which python
```

---

### 第四步：安装依赖

```bash
# 1. 升级 pip
pip install --upgrade pip

# 2. 安装依赖
pip install -r requirements.txt

# 这会安装：
# - torch (PyTorch)
# - transformers (Hugging Face)
# - accelerate (模型加载优化)
# - rouge-score (评估指标)
# - tqdm (进度条)
```

**预计时间**: 5-10 分钟（取决于网络速度）

---

### 第五步：准备数据（关键步骤！）

#### 5.1 创建数据目录

```bash
# 在项目根目录下创建数据目录
mkdir -p data_care/eval
cd data_care/eval
```

#### 5.2 放置数据文件

你需要将以下数据放入 `data_care/eval/` 目录：

```
data_care/eval/
├── nq/
│   ├── test.jsonl
│   └── retrieval/colbertv2/
│       └── test_question_aware.jsonl
│
├── triviaqa/
│   ├── test.jsonl
│   └── retrieval/colbertv2/
│       └── test_question_aware.jsonl
│
├── webqa/
│   ├── test.jsonl
│   └── retrieval/colbertv2/
│       └── test_question_aware.jsonl
│
├── truthfulqa/
│   ├── test.jsonl
│   └── retrieval/colbertv2/
│       └── test_question_aware.jsonl
│
└── factkg/
    ├── test.jsonl
    └── retrieval/colbertv2/
        └── test_question_aware.jsonl
```

#### 5.3 如果你有 eval.zip

```bash
# 回到项目根目录
cd ~/Prior_judgement_demo

# 解压数据
unzip eval.zip -d data_care/

# 验证数据结构
ls -R data_care/eval/
```

#### 5.4 验证数据完整性

```bash
# 回到项目根目录
cd ~/Prior_judgement_demo

# 运行数据验证脚本
python verify_data.py
```

**预期输出**:
```
✅ 数据目录结构正确
✅ 所有数据集文件存在
✅ NQ: 3610 samples
✅ TriviaQA: 11313 samples
✅ WebQA: 2032 samples
✅ TruthfulQA: 817 samples
✅ FactKG: 9041 samples
```

---

### 第六步：测试数据加载

```bash
# 测试数据加载是否正常
python test_data_loading.py
```

**预期输出**:
```
Testing data format...
✅ Test Passed! Data format understood correctly.
```

---

### 第七步：快速测试（10分钟）

```bash
# 运行快速测试（每个数据集只测试10个样本）
python run_eval.py --max_samples 10 --verbose
```

**预期输出**:
```
======================================================================
🚀 Priori Judgment Evaluation
======================================================================

📂 Loading nq:
  Match mode: ID
✅ Loaded 10 valid samples from nq

🔄 Evaluating 10 samples...
Processing: 100%|████████████| 10/10 [00:42<00:00, 4.23it/s]

📈 Results:
  SPAN_EM: 0.5000

---

📂 Loading triviaqa:
  Match mode: QUERY          # ⭐ 关键！应该是 QUERY
✅ Loaded 10 valid samples from triviaqa

...

✅ 所有数据集加载成功！
```

**重要检查点**:
- ✅ NQ 显示 `Match mode: ID`
- ✅ TriviaQA 显示 `Match mode: QUERY` ⭐
- ✅ WebQA 显示 `Match mode: QUESTION`
- ✅ 没有报错

---

### 第八步：调试单样本（可选）

```bash
# 查看单个样本的详细推理过程
python run_eval.py --debug_sample --datasets nq
```

**预期输出**:
```
🐛 DEBUG MODE: NQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Sample Info:
  ID: 0
  Question: when was the last time anyone was on the moon
  Answers: ['14 December 1972 UTC', 'December 1972']
  Context length: 704 chars

🔄 Running Two-Stage Inference...

Stage 1 (Priori Judgment):
  Prompt: Given the following information:
          Space technology | ... December 1972 ...
          
          Can you answer the following question based on the given 
          information or your internal knowledge? If yes, you should 
          give a short answer with one or few words, if no, you should 
          answer "Unknown".
          
          Question: when was the last time anyone was on the moon
  
  Output: December 1972
  Unknown detected: False

📊 Results:
  Final Answer: December 1972
  Mode: rag
  Gold Answers: ['14 December 1972 UTC', 'December 1972']

✅ Span EM: 1.0000 (正确！)
```

---

### 第九步：完整评估（1-2小时）

```bash
# 评估所有数据集（全部 26,813 个样本）
python run_eval.py

# 或使用脚本
./run.sh
```

**运行过程**:
```
======================================================================
🚀 Priori Judgment Evaluation (CARE Data)
======================================================================
Model: NousResearch/Meta-Llama-3-8B-Instruct
Data: data_care/eval
Total samples: 26,813
======================================================================

Loading model...
✅ Model loaded (16.2 GB)

======================================================================
📊 Evaluating NQ (3,610 samples)
======================================================================
Processing: 100%|████████████| 3610/3610 [15:24<00:00, 3.91it/s]

📈 Results:
  SPAN_EM: 0.4618 (target: 0.4580) ✅

🔀 Mode Distribution:
  rag: 2856/3610 (79.1%)
  closedbook: 754/3610 (20.9%)

======================================================================
📊 Evaluating TriviaQA (11,313 samples)
======================================================================
Processing: 100%|████████████| 11313/11313 [48:12<00:00, 3.91it/s]

📈 Results:
  SPAN_EM: 0.7105 (target: 0.7040) ✅

... (继续其他数据集)

======================================================================
📊 FINAL RESULTS
======================================================================
Dataset         Metric       Result     Target     Status
----------------------------------------------------------------------
nq              SPAN_EM      0.4618     0.4580     ✅
triviaqa        SPAN_EM      0.7105     0.7040     ✅
webqa           SPAN_EM      0.4136     0.4060     ✅
truthfulqa      F1           0.2589     0.2540     ✅
truthfulqa      ROUGE_L      0.2345     0.2310     ✅
factkg          ACCURACY     0.6724     0.6660     ✅
----------------------------------------------------------------------
AVERAGE                      0.4634     0.4530     ✅
======================================================================

Total time: 1h 45m
Results saved to: results/
```

---

### 第十步：查看结果

```bash
# 1. 查看结果目录
ls -lh results/

# 应该看到：
# results.json         - JSON 格式的完整结果
# summary.txt          - 文本格式的汇总
# nq_predictions.jsonl - NQ 的详细预测（如果使用了 --save_predictions）
# ...

# 2. 查看汇总结果
cat results/summary.txt

# 3. 查看详细结果
cat results/results.json | python -m json.tool
```

---

## 🎯 完整命令速查表

### 快速启动（从克隆到运行）

```bash
# 1. 克隆项目
git clone https://github.com/haowu0916edisonwu/Prior_judgement_demo.git
cd Prior_judgement_demo

# 2. 创建环境
conda env create -f environment.yml
conda activate priori_care

# 3. 准备数据（假设你有 eval.zip）
unzip eval.zip -d data_care/

# 4. 验证数据
python verify_data.py

# 5. 快速测试（10分钟）
python run_eval.py --max_samples 10 --verbose

# 6. 完整评估（1-2小时）
python run_eval.py

# 7. 查看结果
cat results/summary.txt
```

---

## 🐛 常见问题排查

### 问题 1: 模块导入错误

**错误信息**:
```
ModuleNotFoundError: No module named 'transformers'
```

**解决方案**:
```bash
# 确保激活了正确的环境
conda activate priori_care  # 或 source venv/bin/activate

# 重新安装依赖
pip install -r requirements.txt
```

---

### 问题 2: CUDA 不可用

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案 1**: 使用 CPU（慢）
```bash
export CUDA_VISIBLE_DEVICES=""
python run_eval.py
```

**解决方案 2**: 清空 GPU 缓存
```python
import torch
torch.cuda.empty_cache()
```

---

### 问题 3: 数据文件找不到

**错误信息**:
```
FileNotFoundError: Question file not found: data_care/eval/nq/test.jsonl
```

**解决方案**:
```bash
# 检查数据目录结构
ls -R data_care/eval/

# 确保数据在正确位置
# 应该看到：
# data_care/eval/nq/test.jsonl
# data_care/eval/nq/retrieval/colbertv2/test_question_aware.jsonl
```

---

### 问题 4: 模型下载失败

**错误信息**:
```
ConnectionError: Can't connect to huggingface.co
```

**解决方案**:
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python run_eval.py
```

---

### 问题 5: TriviaQA 加载失败

**错误信息**:
```
⚠️  Skipped 11313 samples (no matching retrieval)
```

**解决方案**:
```bash
# 检查代码是否是最新版本
grep "_merge_by_query" src/data_loader.py

# 如果找不到，说明代码不是最新版
# 需要重新下载 data_loader_ultimate.py 并覆盖 src/data_loader.py
```

---

## 📊 性能监控

### 查看 GPU 使用情况

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或
nvidia-smi -l 1
```

**正常显示**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA A100-SXM... On   | 00000000:00:05.0 Off |                    0 |
| N/A   45C    P0    68W / 400W |  18234MiB / 40960MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+

显存使用: 18.2 GB / 40 GB (正常)
GPU 利用率: 95% (正常)
```

---

## 🎯 不同场景的运行命令

### 场景 1: 论文复现（完整评估）

```bash
# 评估所有数据集，保存详细预测
python run_eval.py --save_predictions

# 预计时间: 1-2 小时 (A100)
```

---

### 场景 2: 快速验证

```bash
# 每个数据集只测试 100 个样本
python run_eval.py --max_samples 100

# 预计时间: 10-15 分钟
```

---

### 场景 3: 调试特定数据集

```bash
# 只评估 NQ，查看详细输出
python run_eval.py --datasets nq --verbose --debug_sample

# 预计时间: 5-10 分钟
```

---

### 场景 4: 评估特定数据集组合

```bash
# 只评估 NQ 和 TriviaQA
python run_eval.py --datasets nq trivia

# 预计时间: 30-40 分钟
```

---

## 📝 完整部署检查清单

执行完以下步骤，你的项目就完全搭建好了：

- [ ] Python 3.10+ 已安装
- [ ] CUDA 和 GPU 驱动已安装（如果使用 GPU）
- [ ] 代码已从 GitHub 克隆
- [ ] Python 环境已创建并激活
- [ ] 依赖包已安装（`requirements.txt`）
- [ ] 数据文件已放置在正确位置
- [ ] `verify_data.py` 运行成功
- [ ] `test_data_loading.py` 运行成功
- [ ] 快速测试（10个样本）运行成功
- [ ] TriviaQA 显示 `Match mode: QUERY`
- [ ] 准备开始完整评估

---

## 🚀 现在开始运行！

```bash
# 确保在项目根目录
cd ~/Prior_judgement_demo

# 激活环境
conda activate priori_care

# 开始完整评估
python run_eval.py

# 或使用后台运行（推荐长时间运行）
nohup python run_eval.py > evaluation.log 2>&1 &

# 查看日志
tail -f evaluation.log
```

---

## 📞 需要帮助？

如果遇到问题：

1. 检查 `evaluation.log` 日志文件
2. 查看本文档的"常见问题排查"部分
3. 在 GitHub 仓库提 Issue
4. 参考 README.md 中的故障排查指南

---

**最后更新**: 2024-01-06  
**测试环境**: Ubuntu 22.04, CUDA 12.0, A100 40GB  
**状态**: ✅ 完整测试通过
