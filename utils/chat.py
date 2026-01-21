from openai import OpenAI
import os
import json
import base64
import re
import requests

from utils.macro import *
from utils.logger import logger, task_dir


CHAT_LAB = "lab"  # 实验室部署的模型
CHAT_REMOTE = "remote"  # 云服务买的api
CHAT_MT = "mt"  # 从美团调用模型
CHAT_MODE = CHAT_LAB

config_file_path = os.path.join("config", "config.json")
with open(config_file_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# qwen
client = OpenAI(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


#  base 64 编码格式
def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            image_bytes = image_file.read()
    else:
        image_bytes = image
    
    return base64.b64encode(image_bytes).decode('utf-8')


def chat_lab(prompt, image_urls=None, temperature=0.01):
    # 实验室的返回接口已修改为openai风格
    api_url = config.get("lab_url")
    headers = {'Content-Type': 'application/json'}
    content = [{'type':'text', 'text':prompt}]
    if image_urls:
        for image_url in image_urls:
            base64_image = encode_image(image_url)
            content.append({'type':'image_url', 'image_url':f'data:image;base64,{base64_image}'})  # image_url 的 value 是图片数据，不是字典
    messages = [
        {
            'role':'user',
            'content':content
        }
    ]
    # logger.log(prompt, color=COLOR.BLUE)
    response = requests.post(api_url, headers=headers, data=json.dumps(messages))
    data = response.json()
    return data


def chat_remote(prompt, image_urls=None, temperature=0.01):
    model = "qwen-vl-max"
    content = [{'type':'text', 'text':prompt}]
    if image_urls:
        for image_url in image_urls:
            base64_image = encode_image(image_url)
            content.append({'type':'image_url', 'image_url':{'url':f'data:image/png;base64,{base64_image}'}})
    
    messages = [
        {
            'role':'user',
            'content':content
        }
    ]
    completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    response = completion.choices[0].message.content
    logger.log(prompt, color=COLOR.BLUE)
    return response


def chat_with_llm(prompt, image_urls=None, role='user', temperature=0.01):
    if CHAT_MODE == CHAT_REMOTE:
        res = chat_remote(prompt, image_urls, temperature)
    elif CHAT_MODE == CHAT_LAB:
        res = chat_lab(prompt, image_urls, temperature)
    else:
        res = {"error": "Error chat mode."}
    return res


def parse_response(response):
    try:
        # 如果是 dict 且有 openai 风格的 choices
        if isinstance(response, dict) and "choices" in response:
            text = response["choices"][0].get("text", "")
        else:
            text = response
        
        pattern = r"```json\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(0).strip()
            text = text.replace('```json\n', '').replace('```', '').replace('\n```', '')
            text = re.sub(r',\s*}', '}', text)  # 去除可能的最后多余的逗号
        json_message = json.loads(text)
    except Exception as e:
        logger.log(f"Error parsing response: {e}", color=COLOR.RED)
        json_message = response  # 如果解析失败，返回原始响应
    return json_message

