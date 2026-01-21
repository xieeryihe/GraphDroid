# 禁止生成 __pycache__ 
import sys
sys.dont_write_bytecode = True

from core.controller import *
from core.utg.explorer import Explorer
from utils.logger import logger
from utils.config_loader import config
from utils.utils import get_pre_made_intent, read_json_file
from core.agent.agent import Agent


def run_utg():
    controller = AndroidController()
    app_name = 'meituan'
    intent = get_pre_made_intent(app_name)
    max_step = 5
    explorer = Explorer(controller=controller, intent=intent, 
                        max_depth=5, max_step=max_step,
                        task_dir=logger.task_dir,
                        default_start=True)
    # explorer.load_data(5)
    explorer.explore()
    pass


def run_utgs():
    controller = AndroidController()
    apks_dir = config['apks_dir']
    apk_list = os.listdir(apks_dir)
    max_step = 50
    base_task_dir = logger.task_dir  # 记录初始的 task_dir
    logger.log(f"批量化运行，发现 {len(apk_list)} 个待测 APK 文件。")
    for apk in apk_list:
        logger.log(f"开始测试 APK 文件：{apk} 。")
        apk_path = os.path.join(apks_dir, apk)
        app_name = apk.rsplit('.', 1)[0]  # 去掉后缀名的 apk 名称
        sub_task_dir=os.path.join(base_task_dir, app_name)  # 再下划一级apk命名的子目录
        logger.set_task_dir(sub_task_dir)  # 切换logger的task_dir
        explorer = Explorer(controller=controller, apk_path=apk_path, 
                            max_depth=5, max_step=max_step,
                            task_dir=logger.task_dir,
                            default_start=True)
        # explorer.load_data(5)
        explorer.explore()


def run_agent():
    controller = AndroidController()
    app_name = 'meituan'
    intent = get_pre_made_intent(app_name)
    task = "点击外卖按钮，点击任意一家店铺，结束。"
    target_step = 2
    # utg_file_path = "bak\\开源\\开源new\\broccoli\\utg.json"
    utg_file_path = "bak\\闭源\\meituan_50\\utg.json"
    agent = Agent(task=task,
                  controller=controller,
                  task_dir=logger.task_dir,
                  intent=intent,
                  use_utg=True,
                  utg_file_path=utg_file_path,
                  default_start=True,
                  target_step=target_step)
    agent.run()

def run_all_agent():
    from utils.utils import get_shortest_path
    cases_file_path = "bak\\消融\\broccoli_tasks\\brief_cases.json"
    cases = read_json_file(cases_file_path)
    base_task_dir = logger.task_dir  # 记录初始的 task_dir
    utg_file_path = "bak\\开源\\开源new\\broccoli\\utg.json"
    start_case_id = 0
    use_utg = False
    app_name = 'Broccoli'
    intent = get_pre_made_intent(app_name)
    for case in cases:
        case_id = case[DictKey.CASE_ID]
        if case_id < start_case_id:
            continue
        case_start_page_id = case[DictKey.CASE_START_PAGE_ID]
        _, edges = get_shortest_path(0, case_start_page_id, utg_file_path)
        case_sentence = case[DictKey.CASE_SENTENCE]
        case_len = int(case[DictKey.CASE_LEN])
        target_steps = len(edges) + case_len  # 前导+任务长度
        sub_task_dir = os.path.join(base_task_dir, str(case_id))  # 再下划一级 case_i 命名的子目录
        logger.set_task_dir(sub_task_dir)  # 切换logger的task_dir

        logger.log(f"开始执行 Case {case_id} ，目标步骤数：{target_steps} 。")
        agent = Agent(task=case_sentence,
                      controller=AndroidController(),
                      task_dir=logger.task_dir,
                      intent=intent,
                      use_utg=use_utg,
                      utg_file_path=utg_file_path,
                      target_step=target_steps,
                      default_start=True)
        agent.run()

def run_experiment():
    from core.experiment.experiment import experiment
    experiment()


def test():
    pass


if __name__ == "__main__":
    import utils.chat as chat
    chat.CHAT_MODE = chat.CHAT_LAB
    # run_utg()
    # run_utgs()
    run_agent()
    # run_all_agent()
    # run_experiment()
    pass