import os
import traceback
from core.controller import *
from utils.logger import Logger, logger, task_dir
from utils.chat import chat_with_llm, parse_response
from utils.macro import ACTION, DictKey
from utils.draw import draw_rects
from utils.macro import DictKey, LOG_SEPARATOR
from utils.utils import read_json_file, write_json_file, ensure_dir, get_node_by_id, get_edge_by_id, get_rects, get_pre_made_intent, get_shortest_path
from core.experiment.task_generator import lcp
from core.experiment.prompt import PROMPT
from time import time, sleep


def longest_lcp(case, cases):
    """ 计算 case 与 cases 中所有 case 的最大最长公共前缀"""
    target_lcp = []
    for c in cases:
        edge_list = lcp(case, c)
        if len(edge_list) > len(target_lcp):
            target_lcp = edge_list
    return target_lcp


def get_case_by_id(case_id, cases):
    """ 根据 case_id 获取 case """
    for case in cases:
        if case[DictKey.CASE_ID] == case_id:
            return case
    return None


class TaskExecutor:
    # 目前是专用于 mt 实验的执行者
    def __init__(self, task_dir, cases_file_path, src_utg_path, app_name=None):
        # 所有 cases 综合需要的
        self.app_name = app_name
        self.intent = get_pre_made_intent(app_name)  # 获取预设的 intent
        self.task_dir = task_dir

        self.cases = read_json_file(cases_file_path) if cases_file_path else []
        self.src_utg_path = src_utg_path
        self.src_utg = read_json_file(src_utg_path)  # read only
        self.dst_utg_path = os.path.join(task_dir, "utg.json")  # runtime update
        self.finished_case_ids = set()  # 已完成的case_id
        self.reachable_page_ids = set([0])  # 目前可达页面 id
        # self.reachable_pages = []
        self.reachable_edge_ids = set()  # 目前可达边 id
        self.reachable_edges = []
        self.controller = AndroidController()  # agent每次都重新创建，controller 共用
        
        # 每个 case 要用的
        self.current_step = 0
        self.current_image_path = None
    
    def capture_page_state(self, prefix, save_dir):
        # 获取并检查当前页面状态
        screenshot_file = self.controller.get_screenshot(prefix=prefix, save_dir=save_dir)
        xml_file = self.controller.get_u2_xml(prefix=prefix, save_dir=save_dir)
        return screenshot_file, xml_file
    
    def pre_run_single(self):
        # 探索前准备
        self.controller.kill_app(self.intent)  # 尝试杀掉之前已经运行的 app
        self.controller.start_app(self.intent)

    def run_all(self):
        cases_data = []
        cases_data_path = os.path.join(self.task_dir, "cases_data.json")  # 执行的数据
        try:
            while(len(self.finished_case_ids) < len(self.cases)):
                total_time_start = time()
                case_id, pre_edge_ids = self.get_next_case_with_prefix()
                if case_id < 0:
                    logger.log("No more reachable cases to run.", color=COLOR.RED)
                    return
                case = get_case_by_id(case_id, self.cases)
                case_data = self.run_single(case, pre_edge_ids)
                total_time_end = time()

                total_time = total_time_end - total_time_start
                case_data[DictKey.TOTAL_TIME] = total_time
                case_data[DictKey.CASE_LEN] = case[DictKey.CASE_LEN]
                case_data[DictKey.AVERAGE_STEP_TIME] = case_data[DictKey.CASE_TIME] / case[DictKey.CASE_LEN]
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

    def run_single(self, case, pre_edge_ids=[]):
        """运行单个 case，包括前缀操作和真正case"""
        # 检查有没有相同前缀的 case
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
        action_history = []
        case_step_list = []  # 当前 case 的步骤列表，用于保存详细统计数据
        case_file_path = os.path.join(case_dir, f"case_data.json")
        prompt_tokens, completion_tokens, cost = 0, 0, 0

        # 运行前准备
        self.pre_run_single()

        # 执行前缀操作，如果没有前缀操作则为空
        for edge_id in pre_edge_ids:
            pre_step_time_start = time()
            self.case_logger.log(f">>>>> Now start pre step {self.current_step}.", color=COLOR.GREEN)
            edge = get_edge_by_id(self.reachable_edges, edge_id)
            self.current_image_path = os.path.join(case_dir, f"{self.current_step}_pre.png")
            self.capture_page_state(f"{self.current_step}_pre", case_dir)
            self.case_logger.log(f"Pre step {self.current_step}, edge info:\n{edge}")
            self.execute_action(edge)
            pre_step_time_end = time()
            case_step = {**edge, DictKey.TIME: pre_step_time_end - pre_step_time_start}
            self.current_step += 1

        # 执行真正的case
        execute_start_time = time()  # 单纯的执行时间

        for step in case[DictKey.CASE_STEPS]:
            # 这里用 for 而不是靠 LLM 输出 stop 标志等方式是因为：我们已经将整个 case 映射到图中了
            step_time_start = time()
            self.case_logger.log(f">>>>> Now start step {self.current_step}.", color=COLOR.GREEN)
            # 这里如何理解呢？用每个 step 的 edge_id 来鉴别之前是否已经探索过，其实和正常运行，比对交互过的case中，每个step的要求是一样的
            # 比如case1和case2的第一步都是 a 操作，那么等价于 case1 和 case2 的第一步都是 edge_id 为 a 操作对应 edge_id 的边
            # 因此，这里用 case_id 作为操作的唯一标识
            edge_id = step[DictKey.EDGE_ID]

            response = {}
            self.current_image_path = os.path.join(case_dir, f"{self.current_step}.png")
            self.capture_page_state(self.current_step, case_dir)
            if edge_id in self.reachable_edge_ids:  # 回放
                edge = get_edge_by_id(self.reachable_edges, edge_id)
            else:  # 请求LLM，一定是加入新边
                sleep(3)
                SoM_image_path = os.path.join(case_dir, f"{self.current_step}_SoM.png")
                rects = get_rects(self.current_image_path)
                # 为每个 rect 增加 node_id 和 SoM_tag 属性
                for index, rect in enumerate(rects):
                    rect[DictKey.NODE_ID] = index
                    rect[DictKey.SOM_TAG] = index
                draw_rects(src_image_path=self.current_image_path, dst_image_path=SoM_image_path, rects=rects)
                response = self.get_action_by_llm(task, action_history, current_image_path=SoM_image_path, element_info=rects)
                response_log = response.copy()
                response_log.pop(DictKey.PROMPT)
                self.case_logger.log("New edge, LLM response:\n", response_log, color=COLOR.BLUE)
                prompt_tokens += response.get(DictKey.PROMPT_TOKENS, 0)
                completion_tokens += response.get(DictKey.COMPLETION_TOKENS, 0)
                cost += response.get(DictKey.COST, 0)
                
                edge = {
                    DictKey.EDGE_ID: edge_id,
                    DictKey.FROM: step[DictKey.FROM],
                    DictKey.TO: step[DictKey.TO],
                    DictKey.ACTION: response[DictKey.ACTION],
                    DictKey.REASON: response[DictKey.REASON]
                }
                node_id = int(response.get(DictKey.NODE_ID, -1))
                if node_id >= 0:
                    node = get_node_by_id(rects, node_id)
                    edge[DictKey.NODE_ID] = node_id
                    edge[DictKey.BOUNDS] = node.get(DictKey.BOUNDS, None)

                # 全局记录保存
                self.reachable_edge_ids.add(edge_id)
                self.reachable_edges.append(edge)
                self.reachable_page_ids.add(step[DictKey.TO])

            self.case_logger.log(f"Step {self.current_step}, edge:\n{edge}")
            self.execute_action(edge)
            
            # 每一步 action 保存 case 记录保存
            action_history.append({
                DictKey.STEP_ID: self.current_step,
                # DictKey.NODE_ID: edge.get(DictKey.NODE_ID, -1),
                DictKey.ACTION: edge[DictKey.ACTION],
                DictKey.REASON: edge.get(DictKey.REASON, "")
            })
            step_time_end = time()
            step_cost_time = step_time_end - step_time_start
            case_step = {**edge, **response, DictKey.TIME: step_cost_time}  # 合并响应内容
            case_step_list.append(case_step)  # 加入当前case步骤序列
            self.current_step += 1
        self.capture_page_state(self.current_step, case_dir)  # case执行完再保存一次图片
        
        execute_end_time = time()
        execute_time = execute_end_time - execute_start_time
        self.case_logger.log(f"Case {case_id} finished")
        
        # 与本case执行无关的附属信息更新放在下面
        self.finished_case_ids.add(case_id)
        write_json_file(case_step_list, case_file_path)
        self.update_utg()  # 更新当前case集的UTG
        case_data = {
            DictKey.CASE_ID: case_id,
            DictKey.PRE_EDGE_IDS: pre_edge_ids,
            DictKey.PROMPT_TOKENS: prompt_tokens,
            DictKey.COMPLETION_TOKENS: completion_tokens,
            DictKey.COST: cost,
            DictKey.CASE_TIME: execute_time
        }
        return case_data

    def get_action_by_llm(self, task, action_history, element_info, current_image_path):
        """使用 LLM 执行操作"""
        # 这里可以实现使用 LLM 来生成操作
        prompt = PROMPT.ActionGenerator.format(
            task=task,
            element_info=element_info,
            action_history=action_history
        )
        self.case_logger.log(">>> Generating action...", color=COLOR.YELLOW)
        response = chat_with_llm(prompt, image_urls=[current_image_path])
        self.case_logger.log(f"Original response:\n{response}", log_terminal=False)
        ans = parse_response(response)
        return ans

    def get_next_case_with_prefix(self):
        """获取下一个可达的 case id 和 前缀 edge_ids"""
        for case in self.cases:
            case_id = case[DictKey.CASE_ID]
            start_page_id = case[DictKey.CASE_START_PAGE_ID]
            if start_page_id in self.reachable_page_ids and case_id not in self.finished_case_ids:
                if start_page_id != 0:
                    _, edges = get_shortest_path(0, start_page_id, self.dst_utg_path)
                    if edges:
                        return case_id, edges
                else:
                    return case_id, []
        return -1, []

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

    def update_utg(self):
        """更新UTG"""
        utg_data = {
            DictKey.NODES: list(self.reachable_page_ids),
            DictKey.EDGES: self.reachable_edges
        }
        write_json_file(utg_data, self.dst_utg_path)
    
    def check_cases(self):
        """检查 cases 是否可以顺次直接执行完，不行的话需要自己手动补充前缀"""
        while(len(self.finished_case_ids) < len(self.cases)):
            case_id, pre_edge_ids = self.get_next_case_with_prefix()
            if case_id < 0:
                logger.log("No more reachable cases to run.", color=COLOR.RED)
                return False
            case = get_case_by_id(case_id, self.cases)
            logger.log(f"Running case {case_id} with pre_edge_ids: {pre_edge_ids}")
            for step in case[DictKey.CASE_STEPS]:
                self.reachable_edges.append(step)
                to_page_id = step[DictKey.TO]
                if to_page_id not in self.reachable_page_ids:
                    self.reachable_page_ids.add(to_page_id)
            self.finished_case_ids.add(case_id)
            # self.pre_run_single()
            # self.run_single(case, pre_edge_ids)
            self.update_utg()
        return True

    def test_pre_execute(self):
        """测试前缀执行"""
        edge = {'edge_id': 4, 'from': 0, 'to': 3, 'action': 'click', 'bounds': [216, 2120, 432, 2296], 'node_id': 7}
        self.reachable_page_ids.add(4)
        self.reachable_edge_ids.add(4)
        self.reachable_edges.append(edge)
        self.pre_run_single()
        target_case_id = 0
        target_case = get_case_by_id(target_case_id, self.cases)
        self.run_single(target_case, pre_edge_ids=[4])
