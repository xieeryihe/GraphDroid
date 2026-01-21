import os
from datetime import datetime
from utils.chat import chat_with_llm, parse_response
from utils.utils import write_json_file, get_intent_from_apk, get_package, ensure_dir
from utils.logger import logger, COLOR


class BaseAgent:
    """基础 Agent 类，提供 LLM 调用记录功能"""
    
    def __init__(self, task_dir, intent=None, apk_path=None):
        """
        初始化 BaseAgent
        intent : 起始页面信息，形如 com.dianping.v1/com.dianping.v1.NovaMainActivity
        可通过 adb shell dumpsys window | findstr "mCurrentFocus" (windows) 或 
        adb shell dumpsys window | grep -E 'mCurrentFocus|mFocusedApp' (MacOS) 获取
        """
        self.task_dir = task_dir
        self.chat_json_file = os.path.join(self.task_dir, "chat.json")
        self.all_chat_data = []  # 所有步骤的聊天数据
        self.current_step_chat_data = []  # 当前步骤的聊天数据
        self.current_step = 0
        self.step_start_time = None  # 步骤开始时间
        if intent:
            self.intent = intent
        elif apk_path:
            self.intent = get_intent_from_apk(apk_path)
        else:
            raise Exception("No intent or apk_path provided to Explorer.")
        self.package_name = get_package(self.intent)
        if task_dir is None:
            raise Exception("task_dir is required for BaseAgent.")
        ensure_dir(self.task_dir)
    
    def start_step(self, step=None):
        """
        开始新的步骤，重置当前步骤的聊天数据并记录开始时间
        
        Args:
            step: 步骤编号，如果不提供则使用 self.current_step
        """
        self.current_step_chat_data = []
        self.step_start_time = datetime.now()
    
    def _chat_with_llm_and_record(self, prompt, image_urls=None, role='user', temperature=0.01, purpose=""):
        """
        封装 chat_with_llm，用于记录调用 API 的数据，方便后面统计
        
        Args:
            prompt: 提示词
            image_urls: 图片 URL 列表
            role: 角色
            temperature: 温度参数
            purpose: 调用目的描述
            
        Returns:
            LLM 响应结果
        """
        response = chat_with_llm(prompt=prompt, image_urls=image_urls, role=role, temperature=temperature)
        
        # 记录 API 调用信息
        chat_record = {
            "purpose": purpose,  # 调用目的描述
            "prompt": prompt,
            "image_urls": image_urls if image_urls else [],
            "response": response
        }
        self.current_step_chat_data.append(chat_record)
        
        return response
    
    def end_step(self, step=None):
        """
        结束当前步骤，记录步骤的所有 chat 数据
        """
        if self.step_start_time is None:
            logger.log("Warning: step_start_time is None, please call start_step() first", color=COLOR.YELLOW)
            return
        
        step_end_time = datetime.now()
        step_time = step_end_time - self.step_start_time
        step_chat_record = {
            "step": step if step is not None else self.current_step,
            "step_time": round(step_time.total_seconds(), 2),
            "step_start_time": self.step_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "step_end_time": step_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "step_chat_data": self.current_step_chat_data
        }
        self.all_chat_data.append(step_chat_record)
        
        # 重置步骤开始时间
        self.step_start_time = None
    
    def save_chat_data(self):
        """
        保存所有聊天数据到 JSON 文件
        """
        write_json_file(self.all_chat_data, self.chat_json_file)
        logger.log(f"all_chat_data saved to {self.chat_json_file}", color=COLOR.YELLOW)
