# 对Qwen3-VL-2B-Instruct模型进行微调

## 原模型的测试
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

## 微调后的模型的测试
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