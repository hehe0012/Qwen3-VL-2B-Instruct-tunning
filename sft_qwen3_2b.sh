#!/bin/bash

# ======================
# 单卡配置（8GB显存专用）
# ======================
export CUDA_VISIBLE_DEVICES=0          # 使用第一张卡
NPROC_PER_NODE=1                       # 单卡训练
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20001-29999 -n 1)

# ======================
# DeepSpeed 配置（使用 ZeRO-2 + CPU offload）
# ======================
deepspeed="./zero3.json"       # 需要你准备这个文件（见下文）

# ======================
# 模型路径（根据你的实际路径修改）
# ======================
llm="/mnt/c/Users/admin/model/Qwen3-VL-2B-Instruct"

# ======================
# 数据集配置（确保已注册）
# ======================
datasets="count_finetune"              # 你在数据注册文件中定义的名称

# ======================
# 训练超参数（针对8GB显存优化）
# ======================
lr=2e-6                                 # 降低学习率，防止过拟合
batch_size=1                             # 批次最小单位
grad_accum_steps=8                       # 梯度累积，模拟 batch_size=8
max_pixels=250880                        # 控制图像分辨率（560x448）
min_pixels=784                            # 最小分辨率（28x28）
model_max_length=2048                     # 计数问答不需要长文本，减小显存

# ======================
# 输出路径
# ======================
output_dir="./output"
run_name="qwen3vl_count_8gb"

# ======================
# 训练入口文件（根据你的实际路径）
# ======================
entry_file="/mnt/d/workspace/projects/qwen3-VL-2B-Thinking-FP8/qwen-vl-finetune/Qwen3-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py"

# ======================
# 构建训练参数
# ======================
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --eval_data_path /mnt/d/workspace/projects/qwen3-VL-2B-Thinking-FP8/count_finetune_splits/val.json \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 8 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.3 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj"
    "

# ======================
# 启动训练
# ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}