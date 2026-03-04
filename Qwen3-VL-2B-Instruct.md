# 对Qwen3-VL-2B-Instruct模型进行微调

## 原模型的测试
使用vLLM启动原始Qwen3-VL-2B-Instruct模型，并通过OpenAI兼容接口进行计数问答评测。评测脚本会在VQA-v2 val2014计数题子集上计算VQA soft accuracy，并输出整体准确率与每题结果。

```
docker run -it --gpus all \
  -v /mnt/c/Users/admin/model/Qwen3-VL-2B-Instruct:/model \
  -p 8000:8000 \
  --shm-size=8gb \
  vllm/vllm-openai:latest \
  /model \
  --gpu-memory-utilization 0.85 \
  --max-model-len 5952
```

评测示例：
```
python script/evaluate_count_questions.py \
  --count-file count_finetune_splits/test.json \
  --output count_eval_origin.json \
  --max-tokens 200
```

基线结果（count_eval_origin.json）：mean_accuracy = 0.6929。

## 训练集和测试集的选取
数据来源于VQA-v2 val2014问题与标注文件，通过规则筛选“计数类”问题，并做一致性过滤与分桶均衡。

1. 计数问题筛选
  - 依据关键词正则（如“how many”“number of”“数量”“多少”等）过滤问题。
  - 对每条问题补充图像路径与答案集合。
  - 输出为count_questions_val2014.json。

2. 计数标签一致性与样本筛选
  - 将答案归一化为数字（支持数字与英文数字词）。
  - 仅保留满足“至少10条数值答案、且主答案一致率>=0.9”的样本。

3. 切分与分桶均衡
  - 按image_id分组，避免同图像泄漏到不同切分。
  - 按标签分桶（0-6，超过8的归为9-20）进行均衡抽样。
  - 默认切分：train=500、val=100、test=10000（详见count_finetune_splits/stats.json）。

## 微调过程
采用LoRA微调策略，并结合DeepSpeed ZeRO优化以适配8GB显存单卡训练。核心配置如下：

- 模型：Qwen3-VL-2B-Instruct
- LoRA：r=8，alpha=16，dropout=0.05，目标模块为q/k/v/o投影层
- 训练超参：lr=2e-6，batch_size=1，grad_accum=8，epochs=8
- 视觉输入：max_pixels=250880（约560x448），min_pixels=784
- 长度：model_max_length=2048
- 评估/保存：每20步评估，每50步保存，保留最近2个checkpoint

训练入口脚本参考sft_qwen3_2b.sh，完成训练后可使用merge.py将LoRA权重合并为完整模型。

## 微调后的模型的测试
使用合并后的模型启动vLLM服务，并复用同一评测脚本在相同测试集上评估。

```
docker run -it --gpus all \
  -v /mnt/c/Users/admin/model/merged_qwen3vl_model:/model \
  -p 8000:8000 \
  --shm-size=8gb \
  vllm/vllm-openai:latest \
  /model \
  --gpu-memory-utilization 0.85 \
  --max-model-len 5952 \
  --trust-remote-code
```

评测示例：
```
python script/evaluate_count_questions.py \
  --count-file count_finetune_splits/test.json \
  --output count_eval_tunning.json \
  --max-tokens 200
```

微调结果：
- count_eval_tunning.json：mean_accuracy = 0.8231
- count_eval_tunning2.json：mean_accuracy = 0.8212

## 微调前后对比
在同一测试集（10000条计数题）上，微调后准确率显著提升：

| 模型 | 结果文件 | mean_accuracy |
| --- | --- | --- |
| 原始模型 | count_eval_origin.json | 0.6929 |
| 微调模型 | count_eval_tunning.json | 0.8231 |
| 微调模型(复测) | count_eval_tunning2.json | 0.8212 |