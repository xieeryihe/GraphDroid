import os
import traceback
from datetime import datetime
from core.controller import *
from utils.chat import *
from utils.macro import *
from utils.utils import *
from core.cog.page_cognition import get_rects
from utils.draw import draw_rects, crop_image
from utils.logger import Logger, logger, task_dir
from core.utg.prompt import ExplorerPrompt
from utils.page_comparator import draw_layout_image
from core.agent.base_agent import BaseAgent


def mark_page_finished(page_list, id):
    # 根据 id 标记页面为探索完成
    for page in page_list:
        if page[DictKey.PAGE_ID] == id:
            page[DictKey.FINISHED] = True
            return True
    return False


def get_unfinished_page(page_list):
    # 获取未完成的页面
    # 也可以根据 utg 由近及远顺序获取
    for page in page_list:
        if not page[DictKey.FINISHED]:
            return page
    return None


class Explorer(BaseAgent):
    def __init__(self, controller: AndroidController,
                task_dir="temp",
                max_step=10, max_depth=1,
                intent=None, apk_path=None,
                default_start=True):
        """
        default_start: True 表示默认从 launch activity 开始，False 表示从当前页面开始执行
        """
        
        super().__init__(task_dir=task_dir, intent=intent, apk_path=apk_path)
        
        # 初始化时完成目录创建相关工作
        self.apk_path = apk_path  # apk路径
        self.controller = controller
        self.default_start = default_start  # 冷启动 app
        self.explore_mode = EXPLORE_MODE.TRAVERSAL  # 默认为探索模式
        self.max_step = max_step
        self.max_depth = max_depth
        self.current_image_path = ""  # 当前页面截图路径
        self.current_utg_page_id = 0  # 当前页面在 UTG 中的 id
        self.page_list = []  # full_data 中页面信息列表，每一步都是一个 item
        self.node_list = []  # utg 中所有页面的信息列表，因为在utg中，所以称作node
        self.edge_list = []  # utg 中边信息列表
        self.last_utg_page_id = -1  # 上一个页面在 UTG 中的 id
        self.last_event = None  # 上一个操作
        self.last_action_image_path = "" # 上一个交互组件图片路径
        self.directed_utg_page_id = -1  # 需要直达的 utg 页面 id
        self.new_page = False
        self.other_app_times = 0  # 记录切换到其他app的步数次数

        self.full_data_dir = os.path.join(self.task_dir, "full_data")  # 所有探索过程的记录（截图，xml）
        self.utg_data_dir = os.path.join(self.task_dir, "utg_data")  # 只加入 utg 的记录
        
        ensure_dir(self.full_data_dir)
        ensure_dir(self.utg_data_dir)
        self.utg_json_file = os.path.join(self.task_dir, "utg.json")
        self.full_json_file = os.path.join(self.task_dir, "full.json")  # 记录每个页面的info
        self.image_mapping_file = os.path.join(self.task_dir, "image_mapping.txt")  # 实际执行的图片和 utg 中图片的映射
        if not default_start:
            pass 
            # self.load_data()
        
    # 从当前页开始执行, 并加载之前探索的数据
    # 如果只需要指定页面开始探索，则不需要调用该函数
    def load_data(self, utg_page_id=-1):
        if utg_page_id < 0:
            return
        task_dir = self.task_dir
        log_file_index = len(glob(os.path.join(self.task_dir, 'log*.txt')))
        logger = Logger(task_dir=task_dir, log_file=f"log{log_file_index}.txt")  # 新建一个 logger
        logger.log(f'继续执行，从记录{self.task_dir}中读取数据')
        self.last_utg_page_id = utg_page_id  # 当前页面就当做上一次的页面，相当于上一个action什么都不做
        self.current_step = len(glob(os.path.join(self.full_data_dir, '*_SoM.png')))
        if not os.path.exists(self.utg_json_file):
            return
        utg = read_json_file(self.utg_json_file)
        full_page = read_json_file(self.full_json_file)
        self.node_list = utg["nodes"]
        self.edge_list = utg["edges"]
        self.page_list = full_page
    
    def pre_explore(self):
        # 探索前准备
        # 如果有apk_path，且没有安装，则安装apk
        if self.apk_path and not self.controller.is_app_installed(self.package_name):
            logger.log(f"App {self.package_name} not installed, installing from {self.apk_path}...")
            self.controller.install_app(self.apk_path)
            logger.log(f"App {self.package_name} installed.")
        current_intent = self.controller.get_current_intent()
        if self.default_start:
            self.controller.kill_app(current_intent)  # 尝试杀掉前台 app
            self.controller.start_app(self.intent)
    
    def post_explore(self):
        # 探索后处理
        generate_utg(work_dir=self.task_dir)
        logger.log("\n>>>Task finished.")
    
    def should_stop_explore(self):
        # 外部停止条件（内部的停止条件由循环中的过程决定）
        # 超过最大步数则停止
        if self.current_step >= self.max_step:
            logger.log(f"\n>>> Stop! Reach max step {self.max_step}.")
            return True
        
        # 任意一个页面未探索完，认为不应该结束
        ret = bool(self.page_list) and all(page[DictKey.FINISHED] for page in self.page_list)
        return ret
    
    def explore(self):
        """探索"""
        logger.log(f"{LOG_SEPARATOR} Start Explore {LOG_SEPARATOR}")
        logger.log(f"Target intent: {self.intent}", color=COLOR.YELLOW)
        logger.log(f"Current chat mode is: {CHAT_MODE}.", color=COLOR.YELLOW)
        logger.log(f"Current device is: {self.controller.device}.", color=COLOR.YELLOW)

        self.pre_explore()

        while True:
            try:
                # 停止条件
                if self.should_stop_explore():
                    break
                logger.log(f"\n\n>>> Now start step {self.current_step} <<<\n\n")
                self.start_step(step=self.current_step)  # 开始新步骤，重置聊天数据并记录开始时间
                self.current_image_path = os.path.join(self.full_data_dir, f"{self.current_step}.png")
                self.current_SoM_image_path = os.path.join(self.full_data_dir, f"{self.current_step}_SoM.png")
                # 获取并检查当前页面状态
                current_intent = self.controller.get_current_intent()
                logger.log(f"Current intent: {current_intent}", color=COLOR.BLUE)
                current_package_name = get_package(current_intent)

                screenshot_path = self.controller.get_screenshot(prefix=self.current_step, save_dir=self.full_data_dir)
                xml_path = self.controller.get_u2_xml(prefix=self.current_step, save_dir=self.full_data_dir)
                if current_package_name != self.package_name:
                    logger.log(f"Package Error: Current package is {current_package_name}, while target package is {self.package_name}", color=COLOR.RED)
                    last_image_path = os.path.join(self.full_data_dir, f"{self.current_step-1}.png")
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
                else:
                    self.other_app_times = 0  # 重置计数器
                
                rects, summary = self.draw_and_summarize()
                # back_key = ['login', 'authentic', 'error', 'advertisement', 'game']
                # for key in back_key:
                #     if key in summary['type'].lower():
                #         self.find_back = True
                #         break
                
                page_info = {
                    DictKey.PAGE_ID: self.current_step,
                    DictKey.IMAGE_PATH: self.current_image_path,
                    DictKey.PACKAGE: current_package_name,
                    DictKey.ACTIVITY: get_activity(current_intent),
                    DictKey.FRAGMENT: self.controller.getcurfrag(),
                    DictKey.NODE_COUNT: len(rects),
                    DictKey.SUMMARY: summary,
                    DictKey.FINISHED: False,
                }
                
                utg_page_id = self.find_same_utg_page(page_info)  # 检查是否有相同页面
                # 要先标记节点访问，再替换页面，不然会因为节点数不一致出bug
                # 标记上一次交互的节点
                
                # 页面增添/替换
                if utg_page_id >= 0:  # 存在相同页面，则旧页面补充进新页面
                    self.new_page = False
                    self.current_utg_page_id = utg_page_id
                    self.replace_page(old_page_num=utg_page_id, new_page_rects=rects)
                    logger.log(f"Enter same page {utg_page_id}.")
                else:  # 如果没有相同页面，则认为进入新的页面
                    self.new_page = True
                    self.add_page(page_info, rects)
                    logger.log(f"Enter new utg page {self.current_utg_page_id}.")

                # 无论是不是探索过的页面，只要发生页面转移，就加入边
                if self.current_step > 0 and self.current_utg_page_id != self.last_utg_page_id:
                    self.add_edge(self.last_utg_page_id, self.current_utg_page_id, node, action)
                
                # 特殊页面情况检查处理
                # 顺序是：软键盘检查 - 弹窗检查 - 最大深度检查
                node, action = self.special_page_check(rects, page_info)
                if not action:
                    # 如果无特殊页面，则正常选择元素交互
                    node, action = self.get_node_with_interaction()
                
                logger.log(f"交互方式: {action}, 交互组件:\n{node}", color=COLOR.GREEN)
                        
                # 交互
                self.interaction(node=node, action=action)
                mark_node_visited(self.utg_data_dir, self.current_utg_page_id, node)
                # 记录信息（认为进入新的页面）
                self.page_list.append(page_info)
                
                # 保存本轮step的所有chat llm数据
                self.end_step()  # 结束步骤，记录本步的所有 chat 数据
                
                self.last_utg_page_id = self.current_utg_page_id  # 记录上一个utg页面编号
                self.current_step += 1 

            except KeyboardInterrupt:  # 捕获 ctrl + C
                logger.log("KeyboardInterrupt, exiting...", color=COLOR.RED)
                break
            except Exception as e:
                logger.log(f'Error occured when exploring: {e}')
                logger.log(traceback.format_exc())
                if 'Arrearage' in str(e):
                    break
                
        write_json_file(self.page_list, self.full_json_file)  # 记录每一个step的 page_info
        logger.log(f"page_list saved to {self.full_json_file}", color=COLOR.YELLOW)
        self.save_chat_data()  # 保存所有聊天数据

        self.post_explore()
        self.controller.kill_app(self.intent)  # 最后终止APP
    
    def draw_and_summarize(self):
        # 绘制SoM 和 layout文件
        current_image_path = os.path.join(self.full_data_dir, f"{self.current_step}.png")
        self.current_SoM_image_path = os.path.join(self.full_data_dir, f"{self.current_step}_SoM.png")
        rects = get_rects(image_path=current_image_path)
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
        draw_rects(src_image_path=current_image_path, dst_image_path=self.current_SoM_image_path, rects=rects)
        # 绘制布局图片
        current_image_layout_path = os.path.join(self.full_data_dir, f"{self.current_step}_layout.png")
        draw_layout_image(rects, self.controller.width, self.controller.height, current_image_layout_path)
        # 总结页面
        logger.log('正在总结页面-----', color=COLOR.YELLOW)
        summary_time_start = time.time()
        response = self._chat_with_llm_and_record(ExplorerPrompt.PAGE_SUMMARIZOR, [current_image_path], purpose="页面总结")
        summary = parse_response(response)
        summary_time_end = time.time()
        logger.log(f"总结页面耗时: {summary_time_end - summary_time_start}")
        logger.log(str(summary), color=COLOR.GREEN)
        
        # TODO: 过滤组件
        rects = format_rects(rects)  # 格式化 rects，增添相关属性
        return rects, summary

    def special_page_check(self, widgets, page_info):
        """
        特殊页面处理，三种情况只处理一种，按顺序优先，命中后统一调用mark_page_finished并返回。
        """
        node, action = None, None

        # 1.软键盘检查
        logger.log('正在检查软键盘-----', color=COLOR.YELLOW)
        response = self._chat_with_llm_and_record(prompt=ExplorerPrompt.KEYBOARD_CHECKER, image_urls=[self.current_image_path], purpose="检查是否存在软键盘")
        res = str(parse_response(response))
        if "1" in res:
            logger.log(f"Image: {self.current_image_path} contains keyboard.", color=COLOR.BLUE)
            action = ACTION.BACK
            mark_page_finished(self.node_list, self.current_utg_page_id)
            return node, action

        # 2.弹窗检测
        logger.log('正在检查弹窗-----', color=COLOR.YELLOW)
        response = self._chat_with_llm_and_record(ExplorerPrompt.POP_WINDOW_IDENTIFIER, image_urls=[self.current_image_path], purpose="检查是否存在弹窗")
        res = str(parse_response(response))
        if "1" in res:
            logger.log(f"Image: {self.current_image_path} contains pop-up window.", color=COLOR.BLUE)
            prompt = ExplorerPrompt.POP_WINDOW_CLOSER.format(widgets=widgets)
            response = self._chat_with_llm_and_record(prompt, [self.current_SoM_image_path], purpose="选择关闭弹窗组件")
            res = parse_response(response)
            node = get_node_by_id(widgets, res.get(DictKey.NODE_ID, -1))
            action = ACTION.CLICK if node else ACTION.BACK
            if not node:
                logger.log(f"Error: No valid node selected to close pop-up window.", color=COLOR.RED)
            mark_page_finished(self.node_list, self.current_utg_page_id)
            return node, action

        # 3.最大深度检查
        logger.log('正在检查页面深度-----', color=COLOR.YELLOW)
        edge, dis = get_back_edge(0, self.current_utg_page_id, self.utg_json_file)
        logger.log(f"当前页面: {self.current_step}, 页面深度:{dis}, 最大深度{self.max_depth}", color=COLOR.GREEN)
        if dis > self.max_depth:
            logger.log(f"当前页面深度 {dis} 超过最大深度 {self.max_depth}，寻找返回按钮尝试返回。", color=COLOR.YELLOW)
            prompt = ExplorerPrompt.BACK_SELECTOR.format(widgets=widgets)
            response = self._chat_with_llm_and_record(prompt, [self.current_SoM_image_path], purpose="选择返回按钮")
            res = parse_response(response)
            node = get_node_by_id(widgets, res.get(DictKey.NODE_ID, -1))
            action = ACTION.CLICK if node else ACTION.BACK
            if not node:
                logger.log(f"Error: No valid widget selected to go back.", color=COLOR.RED)
            mark_page_finished(self.node_list, self.current_utg_page_id)
            return node, action

        # 没有特殊情况
        return None, None
        
    def get_node_with_interaction(self):
        # 根据探索模式获取下一个要交互的节点
        
        def get_directed_node_with_interaction(node_list):
            edge, dis = get_back_edge(self.current_utg_page_id, self.directed_utg_page_id, self.utg_json_file)
            if edge:
                node = get_node_by_id(node_list, edge.get(DictKey.NODE_ID, None))
                action = edge.get(DictKey.ACTION, None)
            else:
                logger.log(f"没有找到从 {self.current_utg_page_id} 到 {self.directed_utg_page_id} 的边，选择默认操作 Back", color=COLOR.YELLOW)
                node, action = None, ACTION.BACK
                self.explore_mode = EXPLORE_MODE.TRAVERSAL  # 切换为遍历模式
            return node, action
        
        logger.log(f"准备选择节点，当前交互模式: {self.explore_mode}", color=COLOR.YELLOW)

        target_node, action = None, ACTION.BACK  # Back 兜底
        node_list_file = os.path.join(self.utg_data_dir, f"{self.current_utg_page_id}.json")
        node_list = read_node_list(node_list_file)
        
        # 直达模式
        if self.explore_mode == EXPLORE_MODE.DIRECTED:
            return get_directed_node_with_interaction(node_list)

        # 遍历模式
        random.shuffle(node_list)
        logger.log('正在决定下一个交互组件及操作-----', color=COLOR.YELLOW)
        
        for node in node_list:
            shouldVisit = node[DictKey.SHOULD_VISIT]
            isVisited = node[DictKey.IS_VISITED]
            if not shouldVisit or isVisited:
                continue
            logger.log(str(node), color=COLOR.GREEN)
            src_image_path = os.path.join(self.utg_data_dir, f'{self.current_utg_page_id}.png')
            dst_image_path = "debug.png"
            # image_bytes = draw_widget(raw_image_path, node[DictKey.BOUNDS])
            draw_rects(src_image_path, dst_image_path,[{DictKey.BOUNDS: node[DictKey.BOUNDS]}])
            prompt = ExplorerPrompt.CLICK_CHECKER.format(widget=node)
            purpose = "检查可点击性"
            response = self._chat_with_llm_and_record(prompt, [dst_image_path], purpose=purpose)
            ans = parse_response(response)
            if ans and ans.get(DictKey.CLICKABLE) == True:
                target_node = node
                action = ACTION.CLICK
                break
            elif ans and ans.get(DictKey.CLICKABLE) == False:
                node[DictKey.SHOULD_VISIT] = False
                logger.log(f"unclickable ele judge by LLM: {node}", color=COLOR.BLUE)
                continue
            else:
                logger.log(f"Error response from LLM: {ans}", color=COLOR.RED)
                continue
        
        # 如果没有要交互的节点，说明页面已经探索完毕
        if target_node is None:
            logger.log(f"没有找到可交互的节点，当前页面 {self.current_utg_page_id} 已经探索完毕，切换为直达模式", color=COLOR.YELLOW)
            mark_page_finished(self.node_list, self.current_utg_page_id)  # 标记页面探索完成
            self.explore_mode = EXPLORE_MODE.DIRECTED  # 切换为直达模式
            self.directed_utg_page_id = get_unfinished_page(self.page_list)[DictKey.PAGE_ID]  # 获取未完成的页面
            logger.log(f"下一个直达页面: {self.directed_utg_page_id}", color=COLOR.BLUE)
            target_node, action = get_directed_node_with_interaction(node_list)  # 获取下一个要交互的节点，并且肯定是交互过的
            
        # node_list 按照 node_id 重排
        node_list.sort(key=lambda node: node.get('node_id', 0))
        write_node_list(node_list, node_list_file)
        return target_node, action

    def add_page(self, page_info, formatted_rects):
        """ 添加新的页面信息到 utg 中 """
        # 找到最大的 utg_page_id，然后复制图片
        png_files = glob(os.path.join(self.utg_data_dir, '*.png'))
        png_files = [f for f in png_files if re.match(r'\d+\.png$', os.path.basename(f))]  # 匹配所有数字命名的图片
        # 此时，列表元素是诸如 'tasks/task_2024-11-21_17-07-10/utg_data/0.png' 的字符串

        new_page_info = page_info.copy()  # 复制一份，避免修改原始 info
        max_utg_page_id = -1
        for png_file in png_files:  # 获取当前最大的 utg 序号
            file_name = os.path.basename(png_file)
            num = int(os.path.splitext(file_name)[0])
            max_utg_page_id = max(max_utg_page_id, num)
        
        self.current_utg_page_id = max_utg_page_id + 1  # 序号 +1
        copy_files(
            src_dir=self.full_data_dir,
            src_page_id=self.current_step,
            dst_dir=self.utg_data_dir,
            dst_page_id=self.current_utg_page_id
        )

        # 覆写为 utg 相关属性
        new_page_info[DictKey.PAGE_ID] = self.current_utg_page_id
        new_page_info[DictKey.IMAGE_PATH] = os.path.join(self.utg_data_dir, f"{self.current_utg_page_id}.png")
        with open(self.image_mapping_file, "a") as f:  # 记录初始映射关系
            f.write(f"{self.current_step} -> {self.current_utg_page_id}\n")
        
        # 只有新的页面才需要生成
        rects_json = os.path.join(self.utg_data_dir, f"{self.current_utg_page_id}.json")
        write_json_file(formatted_rects, rects_json)  # 写入新的页面信息
        self.node_list.append(new_page_info)
        self.update_utg()

    def add_edge(self, from_node_id, to_node_id, node, action):
        # 向 utg 中添加边
        edge = {
            DictKey.EDGE_ID: len(self.edge_list),
            DictKey.FROM: from_node_id,
            DictKey.TO: to_node_id,
            DictKey.NODE_ID: node.get(DictKey.NODE_ID, None) if node else None,
            DictKey.BOUNDS: node.get(DictKey.BOUNDS, None) if node else None,
            DictKey.ACTION: action
        }
        
        edge_exists = False
        for e in self.edge_list:
            if (
                e[DictKey.FROM] == edge[DictKey.FROM]
                and e[DictKey.TO] == edge[DictKey.TO]
                and e[DictKey.ACTION] == edge[DictKey.ACTION]
                and bounds_iou(e[DictKey.BOUNDS], edge[DictKey.BOUNDS])>0.8
                ):
                # 加入iou指标，当两个起点和终点相同的边的bounds过于重叠，则算作一条边
                edge_exists = True
                logger.log(f"Edge {edge} already exists.")
                break
        
        if not edge_exists:
            last_page_summary = get_page_by_id(self.node_list, from_node_id)[DictKey.SUMMARY]
            current_page_summary = get_page_by_id(self.node_list, to_node_id)[DictKey.SUMMARY]
            page_summaries = {DictKey.FROM:last_page_summary, DictKey.TO:current_page_summary}
            from_image_path = os.path.join(self.utg_data_dir, f'{from_node_id}.png')
            to_image_path = os.path.join(self.utg_data_dir, f'{to_node_id}.png')
            if node:
                logger.log('正在总结组件-----', color=COLOR.YELLOW)
                summary_time_start = time.time()
                edge[DictKey.IMAGE_PATH] = self.last_action_image_path
                action_summarizor_prompt = ExplorerPrompt.ACTION_SUMMARIZOR.format(action=action, summaries=page_summaries)
                # logger.log(action_summarizor_prompt, log_terminal=False)
                marked_image_path = self.last_action_image_path.replace('.png', '_marked.png')  # 将 rect 在 from_image 上画出，减少幻觉
                node[DictKey.SOM_TAG] = action
                # 在from_image 上画出交互组件，只有页面跳转时才画
                draw_rects(src_image_path=from_image_path, dst_image_path=marked_image_path, rects=[node])
                response = self._chat_with_llm_and_record(action_summarizor_prompt, [self.last_action_image_path, marked_image_path, to_image_path], purpose="总结动作")
                summary = parse_response(response)
                summary_time_end = time.time()
                logger.log(f'总结组件耗时: {summary_time_end - summary_time_start}')
                logger.log(str(summary), color=COLOR.GREEN)
                edge[DictKey.SUMMARY] = summary
            else:
                edge[DictKey.IMAGE_PATH] = None
                edge[DictKey.SUMMARY] = "[NO NODE TO SUMMARY]"
            logger.log(f"add edge:\n{edge}", color=COLOR.BLUE)
            self.edge_list.append(edge)
            self.update_utg()

    def interaction(self, node: str, action):
        if action == ACTION.BACK:  # 最先判断返回操作，该操作不需要 node
            self.controller.back()
            return
        
        src_image_dir = self.full_data_dir
        image_num = self.current_step
        bounds = get_bounds(node)
        center = get_center(node)
        action_image_dir = os.path.join(self.utg_data_dir, f"page_{self.current_utg_page_id}") # 交互元素图片保存目录
        if not os.path.exists(action_image_dir):
            os.makedirs(action_image_dir)
        src_image_path = os.path.join(src_image_dir, f"{image_num}.png")
        if action == ACTION.CLICK:
            # 画框，方便查看操作的点
            draw_rectangle(src_image_dir=src_image_dir, image_num=image_num, bounds=bounds)
            self.last_action_image_path = crop_image(src_image_path, action_image_dir, bounds)
            self.controller.click(center[0], center[1])

    def find_same_utg_page(self, current_page_info, top_k=1):
        # 检查是否有相同页面，有：返回相同utg_page_id，没有，返回-1
        # top_k 原来设置为 3 的时候，容易选中最后一个认为是相同的页面，可能跟模型能力不足，先设为 1
        logger.log(f'正在确定是否探索过页面,当前页面 num:{self.current_step}', color=COLOR.YELLOW)

        utg_data = read_json_file(self.utg_json_file)
        
        # 1.首先筛选出package、activity、fragment相同的页面
        utg_pages_info = []
        for utg_page_info in utg_data.get("nodes", []):
            if (
                utg_page_info[DictKey.PACKAGE] == current_page_info[DictKey.PACKAGE]
                and utg_page_info[DictKey.ACTIVITY] == current_page_info[DictKey.ACTIVITY]
                and utg_page_info[DictKey.FRAGMENT] == current_page_info[DictKey.FRAGMENT]
                and utg_page_info[DictKey.PAGE_ID] != current_page_info[DictKey.PAGE_ID]
            ):
                utg_pages_info.append(utg_page_info)
        
        if not utg_pages_info:
            logger.log(f'当前页面 {self.current_step}, 没有与之相似的页面', color=COLOR.YELLOW)
            return -1
        
        # 2.页面 layout 对比，根据阈值筛选 top_k 个候选图片,然后交给llm判断。
        candidates = []
        current_image_layout = current_page_info[DictKey.IMAGE_PATH].replace(".png", "_layout.png")
        current_image_layout_cv = cv2.imread(current_image_layout, cv2.IMREAD_GRAYSCALE)

        for utg_page_info in utg_pages_info:
            utg_image_num = utg_page_info[DictKey.PAGE_ID]
            utg_image_layout = os.path.join(self.utg_data_dir, f"{utg_image_num}_layout.png")
            utg_image_layout_cv = cv2.imread(utg_image_layout, cv2.IMREAD_GRAYSCALE)
            
            score, diff = ssim(current_image_layout_cv, utg_image_layout_cv, full=True)  # 用 SSIM 计算布局相似度
            
            if score >= SAME_THRESHOLD:
                # 布局及其相似，认为是相同页面
                logger.log(f'页面 {self.current_step} 与 utg 页面 {utg_image_num} 布局相似度 {score}，认为是相同页面', color=COLOR.YELLOW)
                return utg_page_info[DictKey.PAGE_ID]
            
            if score > SIMILAR_THRESHOLD:
                logger.log(f'页面 {self.current_step} 与 utg 页面 {utg_image_num} 布局相似度 {score}，作为候选页面', color=COLOR.YELLOW)
                temp_can = {**utg_page_info, DictKey.SCORE: score}
                candidates.append(temp_can)

        # 按相似度降序排序，取前 top_k 个
        candidates = sorted(candidates, key=lambda x: x[DictKey.SCORE], reverse=True)[:top_k]
        can_pages = [  # 只挑选特定属性
            {DictKey.PAGE_ID: can[DictKey.PAGE_ID], DictKey.SUMMARY: can[DictKey.SUMMARY]} for can in candidates
        ]
        image_list = [current_page_info[DictKey.IMAGE_PATH]] + [can[DictKey.IMAGE_PATH] for can in candidates]

        if not can_pages:
            logger.log('没有找到相似页面布局', color=COLOR.RED)
            return -1

        # 用 LLM 从候选页面中选出相同页面
        logger.log('找到相似页面布局，LLM正在区分页面-----', color=COLOR.YELLOW)
        logger.log(f'候选页面:\n{can_pages}\n所有图片列表:\n{image_list}', color=COLOR.YELLOW)

        identify_prompt = ExplorerPrompt.PAGE_IDENTIFIER.format(
            target_page_summary=current_page_info[DictKey.SUMMARY],
            candidate_page_summaries=can_pages)
        response = self._chat_with_llm_and_record(identify_prompt, image_list, purpose="检查相似页面")
        choice = parse_response(response)
        logger.log(f'LLM选择结果: {choice}', color=COLOR.YELLOW)
        return int(choice[DictKey.PAGE_ID]) if choice else -1


    def update_utg(self):
        # 更新 utg（覆写节点和边信息到utg.json）
        utg_data = {
            "nodes": self.node_list,
            "edges": self.edge_list
        }
        write_json_file(utg_data, self.utg_json_file)

    def check_page_complete(self, target_utg_page_id):  # 检验一个page所有node是否探索完
        node_list_file = os.path.join(self.utg_data_dir, f'{target_utg_page_id}.json')
        node_list = read_node_list(node_list_file)
        for node in node_list:
            if not node['isVisited']:
                return False
        return True 

    # TODO: 可以和寻找最短路的函数合并，调用时参数调换位置即可
    def is_circle_exits(self):
        if not self.new_page:
            utg_file_path = os.path.join(self.task_dir, 'utg.json')
            edge, dis = get_back_edge(self.current_utg_page_id, self.last_utg_page_id, utg_file_path)
            if edge:
                return True
        return False
        
    def replace_page(self, old_page_num, new_page_rects):
        # 对于相似页面，old中已经visited的元素，如果和new中rect重合度高，则替换
        logger.log(f"当前页面: {self.current_step}, 与utg页面: {old_page_num} 重合度高，进行替换", color=COLOR.YELLOW)
        old_page_json_path = os.path.join(self.utg_data_dir, f"{old_page_num}.json")
        old_page_rects = read_node_list(old_page_json_path)
        for old_node in old_page_rects:
            if not old_node[DictKey.IS_VISITED]:
                continue
            else:  # isVisited == True
                for new_node in new_page_rects:
                    if is_same_node(old_node, new_node):
                        new_node[DictKey.IS_VISITED] = old_node[DictKey.IS_VISITED]
                        new_node[DictKey.SHOULD_VISIT] = old_node[DictKey.SHOULD_VISIT]
        copy_files(
            src_dir=self.full_data_dir,
            src_page_id=self.current_step,
            dst_dir=self.utg_data_dir,
            dst_page_id=old_page_num
        )
        with open(self.image_mapping_file, "a") as f:  # 记录替换后的映射关系
            f.write(f"{self.current_step} -> {old_page_num}\n")
        write_json_file(new_page_rects, old_page_json_path)  # 写入新的页面信息

