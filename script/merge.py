import os
import shutil

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

# 基础模型路径（即您微调时使用的原始 Instruct 模型）
base_model_path = "/mnt/c/Users/admin/model/Qwen3-VL-2B-Instruct"
# LoRA 权重路径（训练输出目录，例如 ./output_qwen3vl_count）
lora_path = "./output"

# 加载基础模型（使用正确的类）
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,   # 与训练时一致
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 加载 LoRA 并合并
model = PeftModel.from_pretrained(model, lora_path)
merged_model = model.merge_and_unload()

# 保存合并后的完整模型
save_path = "/mnt/c/Users/admin/model/merged_qwen3vl_model"
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

extra_files = [
    "preprocessor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "generation_config.json",
]

for filename in extra_files:
    target_path = os.path.join(save_path, filename)
    lora_source_path = os.path.join(lora_path, filename)
    base_source_path = os.path.join(base_model_path, filename)
    if os.path.exists(lora_source_path):
        shutil.copyfile(lora_source_path, target_path)
    elif os.path.exists(base_source_path):
        shutil.copyfile(base_source_path, target_path)
    else:
        print(f"未找到 {filename}: {lora_source_path} 或 {base_source_path}")

print(f"合并后的模型已保存到 {save_path}")