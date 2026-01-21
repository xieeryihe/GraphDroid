import traceback
from time import time
from utils.logger import logger
from core.cog.page_cognition import get_rects
from utils.utils import *
from core.controller import AndroidController
from core.agent.executor import Executor
from core.utg.prompt import ExplorerPrompt
from core.agent.prompt import AgentPrompt
from core.agent.base_agent import BaseAgent
from utils.draw import draw_rects
from utils.chat import parse_response


def filter_rects(rects):
    # 过滤不需要的属性
    filtered_rects = []
    for rect in rects:
        filtered_rect = rect.copy()
        filtered_rect.pop(DictKey.BOUNDS, None)
        filtered_rect.pop(DictKey.CLASS, None)
        filtered_rects.append(filtered_rect)
    return filtered_rects


def filter_utg(utg):
    # 过滤 UTG 中不需要的属性
    old_nodes = utg.get("nodes", [])
    old_edges = utg.get("edges", [])
    node_keys = [DictKey.PAGE_ID, DictKey.SUMMARY]
    edge_keys = [DictKey.EDGE_ID, DictKey.FROM, DictKey.TO, DictKey.ACTION, DictKey.SUMMARY]
    new_nodes = get_sub_dict_list(old_nodes, node_keys)
    new_edges = get_sub_dict_list(old_edges, edge_keys)
    filtered_utg = {"nodes": new_nodes, "edges": new_edges}
    return filtered_utg


class Agent(BaseAgent):
    def __init__(self, task,
                 controller: AndroidController,
                 task_dir="temp_task",
                 intent=None, apk_path=None,
                 use_utg=False, utg_file_path=None,
                 default_start=True, 
                 target_step=5, max_steps=10):
        super().__init__(task_dir=task_dir, intent=intent, apk_path=apk_path)
        self.task = task
        self.action_history = []
        self.action_data = []  # 用于存储操作数据
        self.controller = controller
        self.executor = Executor(task_dir=task_dir, controller=controller)
        self.default_start = default_start  # 默认冷启动App
        self.max_steps = target_step + 5 if target_step is not None else max_steps
        self.use_utg = use_utg
        if self.use_utg:
            if utg_file_path is None:
                raise ValueError("<use_utg> is True but <utg_file_path> is None.")
            utg = read_json_file(utg_file_path)
            self.utg = filter_utg(utg)  # 只使用过滤后的UTG
        else:
            self.utg = None


    def pre_run(self):
        # 交互前准备
        current_intent = self.controller.get_current_intent()
        if self.default_start:
            self.controller.kill_app(current_intent)  # 尝试杀掉前台 app
            self.controller.start_app(self.intent)

    def post_run(self):
        self.save_chat_data()
        action_data_file = os.path.join(self.task_dir, "action_data.json")
        write_json_file(self.action_data, action_data_file)

    def run(self):
        logger.log(f"{LOG_SEPARATOR} Start Run Agent {LOG_SEPARATOR}")
        logger.log(f"Current task is: {self.task}.", color=COLOR.YELLOW)

        self.pre_run()

        while(self.current_step < self.max_steps):
            try:
                logger.log(f"{LOG_SEPARATOR} Step {self.current_step} {LOG_SEPARATOR}", color=COLOR.GREEN)

                self.start_step(step=self.current_step)  # 开始新步骤，重置聊天数据并记录开始时间
                self.current_image_path = os.path.join(self.task_dir, f"{self.current_step}.png")
                self.current_SoM_image_path = os.path.join(self.task_dir, f"{self.current_step}_SoM.png")

                # 获取并检查当前页面状态
                current_intent = self.controller.get_current_intent()
                logger.log(f"Current intent: {current_intent}", color=COLOR.BLUE)
                current_package_name = get_package(current_intent)

                screenshot_path = self.controller.get_screenshot(prefix=self.current_step, save_dir=self.task_dir)
                xml_path = self.controller.get_u2_xml(prefix=self.current_step, save_dir=self.task_dir)
                if current_package_name != self.package_name:
                    logger.log(f"Package Error: Current package is {current_package_name}, while target package is {self.package_name}", color=COLOR.RED)
                    last_image_path = os.path.join(self.task_dir, f"{self.current_step-1}.png")
                    res = self._chat_with_llm_and_record(ExplorerPrompt.APP_IDENTIFIER, [last_image_path, self.current_image_path], purpose="检查是否在目标 APP 中")
                    flag = str(parse_response(res))
                    logger.log(f"LLM判断flag: {flag}", color=COLOR.YELLOW)
                    if "0" in flag or self.other_app_times >= 2:
                        # LLM判断已经脱离APP或者连续两次操作都未返回原APP，则重新启动APP
                        logger.log(f"flag:{flag}, other_app_times:{self.other_app_times}\n不在目标 APP 中，重新启动。", color=COLOR.YELLOW)
                        self.controller.start_app(self.intent)
                        self.current_step += 1
                        continue
                    else:
                        logger.log("实际上仍在目标 APP 中，可能发生系统弹窗遮挡", color=COLOR.YELLOW)
                        self.other_app_times += 1
                    # 之后继续逻辑，试图通过探索交互回到目标 APP
                rects = self.draw_rects()
                filtered_rects = filter_rects(rects)

                path_chain = None
                if self.use_utg:
                    current_page_id, target_page_id = self.map_page()
                    logger.log(f"Mapped: current page {current_page_id}, target page {target_page_id}", color=COLOR.YELLOW)
                    if current_page_id >= 0 and target_page_id >= 0:
                        path_chain = self.generate_path_chain(current_page_id, target_page_id)
                logger.log(f"Path chain to target page: {path_chain}", log_terminal=False)

                info = self.generate_action(
                    element_info=filtered_rects,
                    image_path=self.current_SoM_image_path,
                    path_chain=path_chain
                )
                action = info.get(DictKey.ACTION, None)
                if action == ACTION.STOP:
                    logger.log(">>> Stop action received, stopping exploration. <<<")
                    break
                node_id = info.get(DictKey.NODE_ID, None)
                if node_id is not None:
                    node = get_node_by_id(rects, int(node_id))
                    info = {**node, **info}  # 合并字典
                else:
                    logger.log("Warning: Generated action has no node_id.", color=COLOR.RED)
                info[DictKey.IMAGE_PATH] = self.current_image_path
                logger.log(f"execute info:\n{info}")
                self.executor.execute(info)
                
                self.action_data.append(info)  # 保存操作数据
                action_info = get_sub_dict(info, [DictKey.ACTION, DictKey.CONTENT_DESC, DictKey.CLASS, DictKey.RESOURCE_ID, DictKey.REASON])
                self.action_history.append(action_info)  # 保存action信息，用于prompt
                self.current_step += 1
                self.end_step()

            except KeyboardInterrupt:  # 捕获 ctrl + C
                logger.log("KeyboardInterrupt, exiting...", color=COLOR.RED)
                break
            except Exception as e:
                logger.log(f'Error occured when runing: {e}')
                logger.log(traceback.format_exc())
                if 'Arrearage' in str(e):
                    break
        
        self.post_run()

    def map_page(self):
        logger.log(">>> Mapping pages...", color=COLOR.YELLOW)
        prompt = AgentPrompt.PageMapping.format(
            task=self.task,
            utg=str(self.utg)
        )
        response = self._chat_with_llm_and_record(prompt, image_urls=[self.current_image_path], purpose="Map page")
        response = parse_response(response)
        logger.log(f"Formatted response:\n{response}")
        current_page_id = response.get("current_page_id", None)
        target_page_id = response.get("target_page_id", None)
        return int(current_page_id), int(target_page_id)
    
    def generate_path_chain(self, from_page_id, to_page_id):
        # 生成路径链
        pages_id, edges_id = get_shortest_path(from_page_id, to_page_id, utg=self.utg)
        page_list = self.utg.get(DictKey.NODES, [])
        edge_list = self.utg.get(DictKey.EDGES, [])
        pages_info, edges_info = [], []
        for page_id in pages_id:
            page_info = get_page_by_id(page_list, page_id)
            pages_info.append(page_info)
        for edge_id in edges_id:
            edge_info = get_edge_by_id(edge_list, edge_id)
            edges_info.append(edge_info)
        path_chain = []
        for i in range(len(pages_info)-1):
            path_chain.append(edges_info[i])
            path_chain.append(pages_info[i+1])
        return path_chain
    
    def generate_action(self, element_info, image_path, path_chain=None):
        addition_utg = ""
        if self.utg is not None and path_chain is not None:
            addition_utg = AgentPrompt.AdditionUTG.format(path_chain=str(path_chain))

        prompt = AgentPrompt.ActionGenerator.format(
            task=self.task,
            element_info=element_info,
            action_history=self.action_history,
            addition_utg=addition_utg
        )
        logger.log(">>> Generating action...", color=COLOR.YELLOW)
        logger.log(prompt, log_terminal=False)
        response = self._chat_with_llm_and_record(prompt, image_urls=[image_path], purpose="Generate action")
        response = parse_response(response)
        logger.log(f"Formatted response:\n{response}", log_terminal=False)
        return response

    def draw_rects(self):
        # 获取 rects 并绘制 SoM
        rects = get_rects(image_path=self.current_image_path)

        # 为每个 rect 增加 node_id 和 SoM_tag 属性
        for index, rect in enumerate(rects):
            rect[DictKey.NODE_ID] = index
            rect[DictKey.SOM_TAG] = index
            
        # 写入 SoM 元素信息到文件
        widgets_path = self.current_SoM_image_path.replace(".png", "_elements.txt")
        with open(widgets_path, "w", encoding='utf-8') as file:
            for rect in rects:
                file.write(f"{rect}\n")
        # 绘制 SoM 图片
        draw_rects(src_image_path=self.current_image_path, dst_image_path=self.current_SoM_image_path, rects=rects)
        
        rects = format_rects(rects)  # 格式化 rects，增添相关属性
        return rects