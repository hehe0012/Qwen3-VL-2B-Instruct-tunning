# 对Qwen3-VL-2B-Thinking-FP8模型进行测试

## 环境

安装cuda
安装cudnn
```
sudo cp -P include/cudnn*.h /usr/local/cuda-12.4/include/
sudo cp -P lib/libcudnn* /usr/local/cuda-12.4/lib64/
sudo chmod a+r /usr/local/cuda-12.4/lib64/libcudnn* /usr/local/cuda-12.4/include/cudnn*.h
```
cuda12.9.1+python3.10+torch2.10.0
使用运行在wsl2上的docker拉取和运行vllm
在https://www.modelscope.cn/models/Qwen/Qwen3-VL-2B-Thinking-FP8/summary 下载该模型

## 推理测试
启动vllm多模态服务
```
docker run -it --gpus all \
  -v /mnt/c/Users/admin/model/Qwen3-VL-2B-Thinking-FP8:/model \
  -p 8000:8000 \
  --shm-size=8gb \
  vllm/vllm-openai:latest \
  /model \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```
使用test.py测试多模态功能

## 对该模型在计数问题上的表现进行评估
对VQA-v2数据集先筛选出计数问题（约为23000个）
测试结果在count_eval_val2014.json