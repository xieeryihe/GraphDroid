import os
from .macro import COLOR, FilePath
from datetime import datetime

class Logger:
    def __init__(self, task_dir,  log_file="log0.txt"):
        self.set_task_dir(task_dir, log_file)

    def set_task_dir(self, task_dir, log_file=None):
        self.task_dir = task_dir
        log_file_name = os.path.basename(log_file) if log_file else os.path.basename(self.log_file)
        self.log_file = os.path.join(task_dir, log_file_name)
        # 如果文件不存在，则创建它
        try:
            os.makedirs(task_dir, exist_ok=True)
        except Exception as e:
            print(f"An error occurred while creating directory '{task_dir}': {e}")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")

    def log(self, *args, color=COLOR.RESET, log_terminal:bool=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] " + " ".join(map(str, args))
        
        # 写入文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
        
        if log_terminal:
            print(color + log_message + COLOR.RESET)

time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
task_dir = os.path.join(FilePath.TASKS_DIR, f"task_{time_stamp}")  # tasks/task_x-x-x_x-x-x

try:
    os.makedirs(task_dir, exist_ok=True)
except Exception as e:
    print(f"An error occurred while creating directory '{task_dir}': {e}")

# 初始化全局logger
logger = Logger(task_dir=task_dir)