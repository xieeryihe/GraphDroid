import ast
import os
import re
import sys
import traceback
from time import time, sleep

from .scripts import prompts
from .scripts.controller import list_all_devices, AndroidController, traverse_tree
from .scripts.model import parse_explore_rsp, parse_grid_rsp, OpenAIModel, QwenModel
from .scripts.utils import print_with_color, draw_bbox_multi, draw_grid

# 我的项目引入的
from utils.logger import Logger, logger, task_dir
from utils.utils import read_json_file, write_json_file, ensure_dir, get_node_by_id, get_edge_by_id, get_rects, get_pre_made_intent, get_shortest_path
from utils.macro import DictKey, COLOR, ACTION
from utils.chat_mt import construct_message, inference_with_message
from core.controller import AndroidController as my_controller
from core.controller import get_center
from utils.draw import draw_rects


device = "fb298d8e"

REQUEST_INTERVAL = 10
MIN_DIST = 30

# auto_docs_dir = os.path.join(case_dir, "auto_docs")  # 自动探索的文档
# demo_docs_dir = os.path.join(case_dir, "demo_docs")  # 手工录制的文档

no_doc = True  # 目前不用文档
rows, cols = 0, 0

controller = AndroidController(device)
width, height = controller.get_device_size()
if not width and not height:
    logger.error("Failed to get device size.")
    sys.exit(1)

def get_case_by_id(case_id, cases):
    """ 根据 case_id 获取 case """
    for case in cases:
        if case[DictKey.CASE_ID] == case_id:
            return case
    return None


class AppAgent:
    def __init__(self, app_name, src_utg_path, cases_file_path, auto_docs_dir=None, task_dir=None):
        self.app_name = app_name
        self.intent = get_pre_made_intent(app_name)  # 获取预设的 intent
        self.src_utg_file_path = src_utg_path
        self.utg = read_json_file(src_utg_path)
        self.cases = read_json_file(cases_file_path)
        self.auto_docs_dir = auto_docs_dir
        self.task_dir = task_dir
        self.controller = my_controller()
        ensure_dir(task_dir)

        self.current_step = 0
    
    def pre_run_single(self):
        # 探索前准备
        self.controller.kill_app(self.intent)  # 尝试杀掉之前已经运行的 app
        self.controller.start_app(self.intent)

    def capture_page_state(self, prefix, save_dir):
        # 获取并检查当前页面状态
        screenshot_file = self.controller.get_screenshot(prefix=prefix, save_dir=save_dir)
        xml_file = self.controller.get_u2_xml(prefix=prefix, save_dir=save_dir)
        return screenshot_file, xml_file
    
    def execute_action(self, edge):
        # edge 只需要包含 action 和 bounds 就行
        action = edge[DictKey.ACTION]
        if action == ACTION.CLICK:
            center_x, center_y = get_center(edge)
            self.controller.click(center_x, center_y)
            dst_image_path = self.current_image_path.replace(".png", f"_{action}.png")
            draw_rects(src_image_path=self.current_image_path, dst_image_path=dst_image_path, rects=[edge])
        elif action == ACTION.BACK:
            self.controller.back()
        else:
            self.case_logger.log(f"Unknown action: {action}", color=COLOR.RED)
    
    def run_single(self, case, pre_edge_ids=[]):
        case_id = case[DictKey.CASE_ID]
        task = case[DictKey.CASE_SENTENCE]

        # 类属性
        case_dir = os.path.join(self.task_dir, str(case_id))
        ensure_dir(case_dir)  # 确保目录存在
        self.case_logger = Logger(task_dir=case_dir)
        self.case_logger.log(f"Running case {case_id}, pre_edge_ids: {pre_edge_ids}", color=COLOR.YELLOW)
        self.case_logger.log(f"Task:\n{task}", color=COLOR.BLUE)
        self.current_step = 0

        # 从属于当前case的临时变量
        case_step_list = []  # 当前 case 的步骤列表，用于保存详细统计数据
        case_file_path = os.path.join(case_dir, f"case_data.json")
        prompt_tokens, completion_tokens, cost = 0, 0, 0

        # 运行前准备
        self.pre_run_single()

        # 执行前缀操作，如果没有前缀操作则为空
        for edge_id in pre_edge_ids:
            pre_step_time_start = time()
            self.case_logger.log(f">>>>> Now start pre step {self.current_step}.", color=COLOR.GREEN)
            edges = self.utg[DictKey.EDGES]
            edge = get_edge_by_id(edges, edge_id)
            self.current_image_path = os.path.join(case_dir, f"{self.current_step}_pre.png")
            self.capture_page_state(f"{self.current_step}_pre", case_dir)
            self.case_logger.log(f"Pre step {self.current_step}, edge info:\n{edge}")
            self.execute_action(edge)
            pre_step_time_end = time()
            case_step = {**edge, DictKey.TIME: pre_step_time_end - pre_step_time_start}
            self.current_step += 1

        # 执行真正的case
        execute_start_time = time()  # 单纯的执行时间

        task_complete = False
        grid_on = False
        last_act = ""
        max_step = len(pre_edge_ids) + case[DictKey.CASE_LEN] + 5
        try: 
            while self.current_step < max_step:
                step_start_time = time()
                self.current_image_path = os.path.join(case_dir, f"{self.current_step}.png")
                # 换用自己的xml拉取方式
                screenshot_path = self.controller.get_screenshot(self.current_step, save_dir=case_dir)
                xml_path = self.controller.get_u2_xml(self.current_step, save_dir=case_dir)
                # screenshot_path = controller.get_screenshot(self.current_step, case_dir)
                # xml_path = controller.get_xml(self.current_step, case_dir)
                if screenshot_path == "ERROR" or xml_path == "ERROR":
                    break
                if grid_on:
                    global rows, cols
                    rows, cols = draw_grid(screenshot_path, os.path.join(case_dir, f"{self.current_step}_grid.png"))
                    image = os.path.join(case_dir, f"{self.current_step}_grid.png")
                    prompt = prompts.task_template_grid
                else:
                    clickable_list = []
                    focusable_list = []
                    traverse_tree(xml_path, clickable_list, "clickable", True)
                    traverse_tree(xml_path, focusable_list, "focusable", True)
                    elem_list = clickable_list.copy()
                    for elem in focusable_list:
                        bbox = elem.bbox
                        center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                        close = False
                        for e in clickable_list:
                            bbox = e.bbox
                            center_ = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                            dist = (abs(center[0] - center_[0]) ** 2 + abs(center[1] - center_[1]) ** 2) ** 0.5
                            if dist <= MIN_DIST:
                                close = True
                                break
                        if not close:
                            elem_list.append(elem)
                    draw_bbox_multi(screenshot_path, os.path.join(case_dir, f"{self.current_step}_labeled.png"), elem_list)
                    image = os.path.join(case_dir, f"{self.current_step}_labeled.png")
                    self.case_logger.log(f"prompt image_path:{image}")
                    if no_doc:  # 不用文档
                        prompt = re.sub(r"<ui_document>", "", prompts.task_template)
                    else:
                        ui_doc = ""
                        for i, elem in enumerate(elem_list):
                            doc_path = os.path.join(self.auto_docs_dir, f"{elem.uid}.txt")
                            if not os.path.exists(doc_path):
                                continue
                            ui_doc += f"Documentation of UI element labeled with the numeric tag '{i + 1}':\n"
                            doc_content = ast.literal_eval(open(doc_path, "r").read())
                            if doc_content["tap"]:
                                ui_doc += f"This UI element is clickable. {doc_content['tap']}\n\n"
                            if doc_content["text"]:
                                ui_doc += f"This UI element can receive text input. The text input is used for the following " \
                                        f"purposes: {doc_content['text']}\n\n"
                            if doc_content["long_press"]:
                                ui_doc += f"This UI element is long clickable. {doc_content['long_press']}\n\n"
                            if doc_content["v_swipe"]:
                                ui_doc += f"This element can be swiped directly without tapping. You can swipe vertically on " \
                                        f"this UI element. {doc_content['v_swipe']}\n\n"
                            if doc_content["h_swipe"]:
                                ui_doc += f"This element can be swiped directly without tapping. You can swipe horizontally on " \
                                        f"this UI element. {doc_content['h_swipe']}\n\n"
                        print_with_color(f"Documentations retrieved for the current interface:\n{ui_doc}", "magenta")
                        ui_doc = """
                        You also have access to the following documentations that describes the functionalities of UI 
                        elements you can interact on the screen. These docs are crucial for you to determine the target of your 
                        next action. You should always prioritize these documented elements for interaction:""" + ui_doc
                        prompt = re.sub(r"<ui_document>", ui_doc, prompts.task_template)
                prompt = re.sub(r"<task_description>", task, prompt)
                prompt = re.sub(r"<last_act>", last_act, prompt)
                print_with_color("Thinking about what to do in the next step...", "yellow")
                # status, rsp = mllm.get_model_response(prompt, [image])
                message = construct_message(prompt, [image])
                sleep(3)  # 每次请求前 sleep 3s，避免请求过快
                response = inference_with_message(message)  #全部输出，包括统计数据
                prompt_tokens += response.get(DictKey.PROMPT_TOKENS, 0)
                completion_tokens += response.get(DictKey.COMPLETION_TOKENS, 0)
                cost += response.get(DictKey.COST, 0)
                rsp = response[DictKey.RESPONSE_CONTENT]
                self.case_logger.log(prompt, log_terminal=False)
                self.case_logger.log(f"LLM rep:\n{rsp}")

                if grid_on:
                    res = parse_grid_rsp(rsp)
                else:
                    res = parse_explore_rsp(rsp)
                
                self.case_logger.log(f"Parse rsp: {res}", color=COLOR.BLUE)

                act_name = res[0]
                if act_name == "FINISH":
                    task_complete = True
                    break
                if act_name == "ERROR":
                    break
                last_act = res[-1]
                res = res[:-1]
                if act_name == "tap":
                    _, area = res
                    tl, br = elem_list[area - 1].bbox# print(elem_list[area - 1].bbox)  # ((432, 2120), (648, 2296))
                    
                    bbox = elem_list[area - 1].bbox
                    bounds = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
                    dst_image_path = os.path.join(case_dir, f"{self.current_step}_tap.png")
                    rect = {DictKey.BOUNDS: bounds}
                    draw_rects(self.current_image_path, dst_image_path, [rect])
                    
                    x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                    ret = controller.tap(x, y)
                    if ret == "ERROR":
                        print_with_color("ERROR: tap execution failed", "red")
                        break
                elif act_name == "text":
                    _, input_str = res
                    ret = controller.text(input_str)
                    if ret == "ERROR":
                        print_with_color("ERROR: text execution failed", "red")
                        break
                elif act_name == "long_press":
                    _, area = res
                    tl, br = elem_list[area - 1].bbox
                    x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                    ret = controller.long_press(x, y)
                    if ret == "ERROR":
                        print_with_color("ERROR: long press execution failed", "red")
                        break
                elif act_name == "swipe":
                    _, area, swipe_dir, dist = res
                    tl, br = elem_list[area - 1].bbox
                    x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                    ret = controller.swipe(x, y, swipe_dir, dist)
                    if ret == "ERROR":
                        print_with_color("ERROR: swipe execution failed", "red")
                        break
                elif act_name == "grid":
                    grid_on = True
                elif act_name == "tap_grid" or act_name == "long_press_grid":
                    _, area, subarea = res
                    x, y = area_to_xy(area, subarea)
                    if act_name == "tap_grid":
                        ret = controller.tap(x, y)
                        if ret == "ERROR":
                            print_with_color("ERROR: tap execution failed", "red")
                            break
                    else:
                        ret = controller.long_press(x, y)
                        if ret == "ERROR":
                            print_with_color("ERROR: tap execution failed", "red")
                            break
                elif act_name == "swipe_grid":
                    _, start_area, start_subarea, end_area, end_subarea = res
                    start_x, start_y = area_to_xy(start_area, start_subarea)
                    end_x, end_y = area_to_xy(end_area, end_subarea)
                    ret = controller.swipe_precise((start_x, start_y), (end_x, end_y))
                    if ret == "ERROR":
                        print_with_color("ERROR: tap execution failed", "red")
                        break
                if act_name != "grid":
                    grid_on = False
                
                sleep(REQUEST_INTERVAL)

                step_end_time = time()
                case_step = {
                    DictKey.STEP_ID: self.current_step,
                    **response,
                    DictKey.TIME: step_end_time - step_start_time
                }
                case_step_list.append(case_step)
                self.current_step += 1
            
            if task_complete:
                self.case_logger.log("Task completed successfully", color=COLOR.YELLOW)
            elif self.current_step == max_step:
                self.case_logger.log("Task finished due to reaching max rounds", color=COLOR.YELLOW)
            else:
                self.case_logger.log("Task finished unexpectedly", color=COLOR.RED)
        except Exception:
            self.case_logger.log(traceback.format_exc(), color=COLOR.RED)
        finally:
            execute_end_time = time()
            execute_time = execute_end_time - execute_start_time
            step_num = self.current_step - len(pre_edge_ids)  # 实际执行的步数
            step_num = max(step_num, 1)  # 防止除零，因为有的case可能第一部就错了，进入不了 current_step++ 的地方
            case_data = {
                DictKey.CASE_ID: case_id,
                DictKey.PRE_EDGE_IDS: pre_edge_ids,
                DictKey.STEP_NUM: step_num,
                DictKey.PROMPT_TOKENS: prompt_tokens,
                DictKey.COMPLETION_TOKENS: completion_tokens,
                DictKey.COST: cost,
                DictKey.CASE_TIME: execute_time,
                DictKey.AVERAGE_STEP_TIME : execute_time / step_num
            }
            write_json_file(case_step_list, case_file_path)
        return case_data

    def run_all(self, start_case_id=0, finished_cases_file=""):
        cases_data = []
        cases_data_path = os.path.join(self.task_dir, "cases_data.json")  # 执行的数据
        finished_cases = read_json_file(finished_cases_file)
        try:
            for case in self.cases:
                case_id = case[DictKey.CASE_ID]
                if case_id < start_case_id:
                    continue
                total_time_start = time()
                finished_case = get_case_by_id(case_id, finished_cases)
                pre_edge_ids = finished_case[DictKey.PRE_EDGE_IDS]
                case_data = self.run_single(case, pre_edge_ids)
                total_time_end = time()

                total_time = total_time_end - total_time_start
                case_data[DictKey.TOTAL_TIME] = total_time
                case_data[DictKey.CASE_LEN] = case[DictKey.CASE_LEN]
                # case_data[DictKey.AVERAGE_STEP_TIME] = case_data[DictKey.CASE_TIME] / case[DictKey.CASE_LEN]
                cases_data.append(case_data)
                logger.log(f"Finished case {case_id}, cost time: {total_time:.1f} seconds", color=COLOR.GREEN)
        
        except KeyboardInterrupt:
            logger.log("KeyboardInterrupt, exiting...", color=COLOR.RED)
        except Exception as e:
            logger.log(f"Error occurred while running cases: {e}", color=COLOR.RED)
            logger.log(traceback.format_exc(), color=COLOR.RED)
        finally:
            write_json_file(cases_data, cases_data_path)
        
        logger.log("All cases finished", color=COLOR.GREEN)
# configs = load_config()


def area_to_xy(area, subarea):
    area -= 1
    row, col = area // cols, area % cols
    x_0, y_0 = col * (width // cols), row * (height // rows)
    if subarea == "top-left":
        x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 4
    elif subarea == "top":
        x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 4
    elif subarea == "top-right":
        x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) // 4
    elif subarea == "left":
        x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 2
    elif subarea == "right":
        x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) // 2
    elif subarea == "bottom-left":
        x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) * 3 // 4
    elif subarea == "bottom":
        x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) * 3 // 4
    elif subarea == "bottom-right":
        x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) * 3 // 4
    else:
        x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 2
    return x, y

