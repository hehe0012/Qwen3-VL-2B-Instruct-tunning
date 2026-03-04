import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# 读取图片并转换为 base64
with open("/mnt/d/workspace/projects/qwen3-VL-2B-Thinking-FP8/pic1.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# 多模态对话测试
response = client.chat.completions.create(
    model="/model",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "请描述这张图片"
                }
            ]
        }
    ],
    max_tokens=2000
)

print(response)