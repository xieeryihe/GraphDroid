import os
import random
from utils.chat import chat_with_llm, parse_response
from utils.utils import read_json_file, write_json_file, get_page_by_id, get_edge_by_id, get_edges_by_ids, get_sub_dict
from utils.macro import DictKey, COLOR, ACTION
from core.experiment.prompt import *
from utils.logger import logger, task_dir

POLICY_PURE_RANDOM = "pure_random"  # 纯随机策略
POLICY_APPOINT_RANDOM = "appoint_random"  # 指定起点随机
POLICY_APPOINT = "appoint"  # 指定起始id列表
MIN_LENGTH = 3  # 最小 case 长度
MAX_LENGTH = 7  # 最大 case 长度

# =============== 简单工具 =================


def get_sub_page(page, add_keys=[]):  # 精简 page 信息
    base_keys = [DictKey.PAGE_ID, DictKey.SUMMARY]
    return get_sub_dict(page, base_keys + add_keys)


def get_sub_edge(edge, add_keys=[]):  # 精简 edge 信息
    base_keys = [DictKey.EDGE_ID, DictKey.FROM, DictKey.TO, DictKey.ACTION, DictKey.SUMMARY]
    return get_sub_dict(edge, base_keys + add_keys)


def preprocess_utg(src_utg_file, dst_utg_file):
    # 在这里总结 UTG 的边信息，以自然语言的形式添加到原始UTG信息中
    src_utg = read_json_file(src_utg_file)
    nodes, edges = src_utg[DictKey.NODES], src_utg[DictKey.EDGES]
    new_edges = []
    for edge in edges:
        from_page = get_page_by_id(nodes, edge[DictKey.FROM])
        to_page = get_page_by_id(nodes, edge[DictKey.TO])
        edge_desc_prompt = PROMPT.EDGE_DESCRIPTION.format(
            from_page_info=get_sub_page(from_page),
            to_page_info=get_sub_page(to_page),
            edge=get_sub_edge(edge)
        )
        res = chat_with_llm(prompt=edge_desc_prompt)["choices"][0]["text"]
        logger.log(f"edge {edge[DictKey.EDGE_ID]}, description:\n{res}")
        new_edge = { **edge, DictKey.DESCRIPTION: res }
        new_edges.append(new_edge)

    src_utg[DictKey.EDGES] = new_edges
    write_json_file(src_utg, dst_utg_file)


class TaskGenerator:
    def __init__(self, src_utg_path, logger, task_dir,
                 policy=POLICY_PURE_RANDOM, case_num=10, base_ids=None, start_page_ids=[]):
        self.start_page_ids = start_page_ids  # 起始页面 ID 列表
        self.src_utg_path = src_utg_path  # utg 数据源
        self.logger = logger
        self.task_dir = task_dir
        self.full_cases_path = os.path.join(task_dir, "full_cases.json")  # 全部测试用例保存路径
        self.brief_cases_path = os.path.join(task_dir, "brief_cases.json")  # 精简测试用例保存路径
        self.image_file_path = os.path.join(task_dir, "cases_heatmap.png")  # 热力图路径
        self.distribution_file_path = os.path.join(task_dir, "cases_distribution.json")  # 统计分布路径

        self.utg = read_json_file(src_utg_path)
        self.nodes = self.utg[DictKey.NODES]
        self.edges = self.utg[DictKey.EDGES]
        self.policy = policy  # 生成策略
        self.full_steps = []  # 当前case 已生成的 step 的全部信息
        self.generated_steps = []  # 当前 case 已生成的sub_step, 用于prompt
        self.full_cases = []  # 目前全部已生成的测试用例
        self.cases_sentence = []  # 目前全部已生成的测试用例的短句，用于prompt
        self.adj_list = self._build_adj_list()  # 构建邻接表缓存，避免重复计算

        # 根据生成策略，生成起始页面 ID 列表
        if self.policy == POLICY_PURE_RANDOM:
            self.start_page_ids = [random.choice(self.nodes)[DictKey.PAGE_ID] for _ in range(case_num)]
        elif self.policy == POLICY_APPOINT_RANDOM:
            if base_ids is None:
                raise ValueError("base_ids must be provided for POLICY_APPOINT_RANDOM.")
            for i in range(len(base_ids)):
                case_count = random.randint(MIN_LENGTH, MAX_LENGTH)  # 随机生成 3 到 7 之间的整数，作为以该id为起始的case数量
                self.start_page_ids.extend([base_ids[i]] * case_count)  # 扩展列表
            random.shuffle(self.start_page_ids)  # 打乱顺序
        # 否则，就用指定的 start_page_ids
        
        logger.log(f"Using policy: {self.policy}", color=COLOR.YELLOW)
        logger.log(f"length of start_page_ids: {len(self.start_page_ids)}", color=COLOR.BLUE)
        logger.log(f"start_page_ids: {self.start_page_ids}", color=COLOR.BLUE)

    def get_case_len(self):
        # sample 一个 [1/x, 1/y] * 页面数的整数，作为 case 长度
        page_count = len(self.nodes)
        lower_bound = page_count // 5  # 向下取整
        upper_bound = page_count // 2  # 向下取整
        random_int = random.randint(lower_bound, upper_bound)
        return random_int
    
    def _build_adj_list(self):
        # 构建邻接表（仅在初始化时调用一次）
        adj_list = {node[DictKey.PAGE_ID]: [] for node in self.nodes}
        for edge in self.edges:
            adj_list[edge[DictKey.FROM]].append(edge[DictKey.EDGE_ID])
        return adj_list

    def get_step_sentence(self, edge_id, description):
        edge = get_edge_by_id(self.edges, edge_id)
        page = get_page_by_id(self.nodes, edge[DictKey.FROM])
        # action = edge[DictKey.ACTION]
        # if action == ACTION.BACK:
        #     step_sentence = description
        # else:
        #     step_sentence = f"在{page[DictKey.SUMMARY][DictKey.TYPE]}{description}"
        step_sentence = f"在{page[DictKey.SUMMARY][DictKey.TYPE]}{description}"
        return step_sentence

    def generate_case_step(self, current_page_id: int):
        # 生成case中的单步
        incident_edges_id = self.adj_list[current_page_id]
        incident_edges = get_edges_by_ids(self.edges, incident_edges_id)  # 只加入当前节点的邻接边为信息
        incident_edges = [get_sub_edge(edge, [DictKey.DESCRIPTION]) for edge in incident_edges]  # 精简信息
        all_pages = [get_sub_page(node) for node in self.nodes]
        case_step_prompt = PROMPT.CASE_STEP.format(
            all_pages=all_pages, incident_edges=incident_edges, current_page_id=current_page_id,
            generated_steps=self.generated_steps)
        logger.log(f"case_step_prompt:\n{case_step_prompt}\n", log_terminal=False)
        response = chat_with_llm(prompt=case_step_prompt)
        ans = parse_response(response)
        return ans

    def generate_test_case(self, start_page_id, k=4):
        # 生成一条 test case，长度由 k 指定
        if start_page_id < 0:
            raise ValueError("start_page_id must be a non-negative integer.")
        current_page_id = start_page_id
        self.full_steps = []  # 重置已生成的步骤
        self.generated_steps = []  # 重置已生成的sub_step, 用于prompt
        error_time = 0
        case_sentence = ""
        while(len(self.generated_steps) < k):
            try:
                logger.log(f"generating step {len(self.generated_steps)}, target case len:{k}")
                logger.log(f"current page id: {current_page_id}")

                step = self.generate_case_step(current_page_id=current_page_id)
                edge_id = int(step[DictKey.EDGE_ID])
                if edge_id < 0:  # 如果进行到某个阶段，导致无边可选，则停止当次生成
                    logger.log("no edge available, stop case generation.")
                    break

                # edge 全部信息
                edge = get_edge_by_id(self.edges, edge_id)
                edge = {**edge, **step}
                self.full_steps.append(edge)  # 保存全部步骤信息

                # edge 精简信息, 用于prompt
                sub_edge = get_sub_edge(edge, [DictKey.DESCRIPTION])  # 精简信息
                step_sentence = self.get_step_sentence(edge_id, edge[DictKey.DESCRIPTION])
                case_sentence += step_sentence  # 修改prompt后，默认每个step有句号
                self.generated_steps.append(sub_edge)
                
                current_page_id = edge[DictKey.TO]
                logger.log(f"generated step: {step}")
            except Exception as e:
                error_time += 1
                print(e)
            if error_time >= 3:
                raise ValueError("Too many errors occurred while generating case.")
        case = {  # case id 将在外层调用补上
            DictKey.CASE_START_PAGE_ID: start_page_id,
            DictKey.CASE_LEN: len(self.generated_steps),
            DictKey.CASE_SENTENCE: case_sentence,
            DictKey.CASE_STEPS: self.full_steps  # 使用step的全部信息
        }
        return case

    def generate_test_cases(self):
        # 随机生成 case，必填 case_num，指定起点生成 case，必填 start_page_ids
        # 生成的 case 在时间戳目录中
        i = 0
        regenerate_times = 0
        while i < len(self.start_page_ids):
            k = self.get_case_len()  # 随机生成 case 长度
            logger.log(f"\n>>> Generating case: {i+1}, start_page_id: {self.start_page_ids[i]}, k: {k} <<<\n")
            case = self.generate_test_case(start_page_id=self.start_page_ids[i], k=k)
            logger.log(f"Generated case {i+1}:\n{case}")
            if case[DictKey.CASE_LEN] < k:
                logger.log(f"case {i+1} is too short, regenerate.")
                regenerate_times += 1
                if regenerate_times >= 3:  # 如果连续生成失败，跳过
                    logger.log(f"Too many failed attempts, skip case {i+1}.")
                    i += 1
                    regenerate_times = 0
                continue
            self.full_cases.append({ DictKey.CASE_ID: i, **case })  # 加上 case_id 属性
            self.cases_sentence.append({ DictKey.CASE_ID: i, DictKey.CASE_SENTENCE: case[DictKey.CASE_SENTENCE] })
            i += 1
        logger.log(f"Generated {len(self.full_cases)} cases.")
        
        # 生成完毕，写入文件
        write_json_file(self.full_cases, self.full_cases_path)
        brief_cases = get_brief_cases(self.full_cases)
        write_json_file(brief_cases, self.brief_cases_path)
        distribution = get_data_distribution(self.full_cases)
        write_json_file(distribution, self.distribution_file_path)
        draw_cases_heatmap(self.full_cases, self.image_file_path)


def generate_cases_by_human(processed_utg_file, cases_ids_file):
    utg = read_json_file(processed_utg_file)
    edges = utg[DictKey.EDGES]
    pages = utg[DictKey.NODES]
    origin_steps_ids = read_json_file(cases_ids_file)
    full_cases = []
    for id, step_ids in enumerate(origin_steps_ids):
        case_sentence = ""
        full_steps = []
        start_step = get_edge_by_id(edges, step_ids[0])
        start_page_id = start_step[DictKey.FROM]
        for step_id in step_ids:
            edge = get_edge_by_id(utg[DictKey.EDGES], step_id)
            description = edge[DictKey.DESCRIPTION]
            page = get_page_by_id(pages, edge[DictKey.FROM])
            # action = edge[DictKey.ACTION]
            # if action == ACTION.BACK:
            #     step_sentence = description
            # else:
            #     step_sentence = f"在{page[DictKey.SUMMARY][DictKey.TYPE]}{description}"
            step_sentence = f"在{page[DictKey.SUMMARY][DictKey.TYPE]}{description}"
            case_sentence += step_sentence
            full_steps.append(edge)
        case = {  # case id 将在外层调用补上
            DictKey.CASE_ID: id,
            DictKey.CASE_START_PAGE_ID: start_page_id,
            DictKey.CASE_LEN: len(step_ids),
            DictKey.CASE_SENTENCE: case_sentence,
            DictKey.CASE_STEPS: full_steps  # 使用step的全部信息
        }
        full_cases.append(case)
    return full_cases


# =============== 数据统计 =================


def lcp(case1, case2):
    # 计算最大前缀
    steps1 = case1[DictKey.CASE_STEPS]
    steps2 = case2[DictKey.CASE_STEPS]
    min_len = min(len(steps1), len(steps2))
    edge_list = []
    for i in range(min_len):
        if steps1[i][DictKey.EDGE_ID] == steps2[i][DictKey.EDGE_ID]:
            edge_list.append(steps1[i])
        else:
            break
    return edge_list


def get_same_steps(case1, case2):
    # 获取两个 case 重叠的 step（顺序可不同，一条边只能用一次）
    steps1 = case1[DictKey.CASE_STEPS]
    steps2 = case2[DictKey.CASE_STEPS]
    same_steps = []
    used_idx2 = set()  # 记录steps2中已匹配的下标

    for step1 in steps1:
        for idx2, step2 in enumerate(steps2):
            if idx2 in used_idx2:
                continue
            if step1[DictKey.EDGE_ID] == step2[DictKey.EDGE_ID]:
                same_steps.append(step1)
                used_idx2.add(idx2)
                break  # 一条边只能用一次，找到就break

    return same_steps


def draw_cases_heatmap(cases, image_file_path='cases_heatmap.png'):
    # 计算 LCP 矩阵
    num_cases = len(cases)
    same_step_matrix = [[0] * num_cases for _ in range(num_cases)]
    for i in range(num_cases):
        for j in range(i + 1, num_cases):
            same_step_num = len(get_same_steps(cases[i], cases[j]))
            same_step_matrix[i][j] = same_step_num
            same_step_matrix[j][i] = same_step_num  # 对称填充

    # 绘制热力图
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(same_step_matrix, annot=True, fmt="g", cmap="YlGnBu",
                     xticklabels=[f'{i+1}' for i in range(num_cases)],
                     yticklabels=[f'{i+1}' for i in range(num_cases)])
    plt.title('Heatmap')
    plt.xlabel('case_id')
    plt.ylabel('case_id')

    # 刻度尺对象
    cbar = ax.collections[0].colorbar

    min_val = 0
    max_val = max(max(row) for row in same_step_matrix) # 取最值

    # 设置整数刻度
    integer_ticks = np.arange(min_val, max_val + 1)
    cbar.set_ticks(integer_ticks)
    cbar.set_ticklabels([str(int(t)) for t in integer_ticks])  # 确保刻度标签为整数的字符串

    plt.savefig(image_file_path)
    plt.show()


def get_data_distribution(cases):
    # 统计 case 的分布情况
    # 计算包含前缀的 case 对
    num_cases = len(cases)
    average_case_len = 0
    edge_id_set = set()  # 所有操作的种类
    from collections import Counter
    edge_id_counter = Counter()
    max_case_len = max([case[DictKey.CASE_LEN] for case in cases])
    min_case_len = min([case[DictKey.CASE_LEN] for case in cases])
    for i in range(num_cases):
        average_case_len += cases[i][DictKey.CASE_LEN]
        for step in cases[i][DictKey.CASE_STEPS]:
            edge_id_set.add(step[DictKey.EDGE_ID])
            edge_id_counter[step[DictKey.EDGE_ID]] += 1
    average_case_len /= num_cases
    # 统计出现2次及以上的边
    redundant_edge_ids = [edge_id for edge_id, count in edge_id_counter.items() if count >= 2]
    redundant_edge_type_ratio = len(redundant_edge_ids) / len(edge_id_set)

    # 统计：
    # 包含了出现2次及以上的边的case数量以及占总 case 数量的比例
    # 包含了重复步骤的case中，每条case的重复步骤占整条case长度的比例，再求这些比例的平均
    redundant_case_num = 0
    redundant_case_ratio = 0
    redundant_edge_ratio = 0
    for case in cases:
        redundant_step_num = 0
        for step in case[DictKey.CASE_STEPS]:
            if step[DictKey.EDGE_ID] in redundant_edge_ids:
                redundant_step_num += 1
        if redundant_step_num > 0:
            redundant_case_num += 1
            redundant_edge_ratio += redundant_step_num / case[DictKey.CASE_LEN]
            print(f"case {case[DictKey.CASE_ID]} has {redundant_step_num} redundant steps.")
    redundant_case_ratio = redundant_case_num / num_cases
    redundant_edge_ratio /= redundant_case_num

    logger.log(f"Total cases: {num_cases}")
    logger.log(f"Average case length: {average_case_len:.2f}")
    logger.log(f"Max case length: {max_case_len}")
    logger.log(f"Min case length: {min_case_len}")
    logger.log(f"Edge type num: {len(edge_id_set)}")
    logger.log(f"Redundant edge type num: {len(redundant_edge_ids)}")
    logger.log(f"Redundant edge type ratio: {redundant_edge_type_ratio:.2f}")
    logger.log(f"Contain redundant case num: {redundant_case_num}")
    logger.log(f"Contain redundant case ratio: {redundant_case_ratio:.2f}")
    logger.log(f"Contain redundant edge ratio: {redundant_edge_ratio:.2f}")

    distribution = {
        "num_cases": num_cases,
        "average_case_len": average_case_len,
        "max_case_len": max_case_len,
        "min_case_len": min_case_len,
        "edge_type_num": len(edge_id_set),
        "redundant_edge_type_num": len(redundant_edge_ids),
        "redundant_edge_type_ratio": redundant_edge_type_ratio,
        "contain_redundant_case_num": redundant_case_num,
        "contain_redundant_case_ratio": redundant_case_ratio,
        "contain_redundant_edge_ratio": redundant_edge_ratio
    }
    return distribution


def get_brief_cases(full_cases):
    brief_cases = []
    for case in full_cases:
        brief_case = get_sub_dict(case, [DictKey.CASE_ID, DictKey.CASE_START_PAGE_ID, DictKey.CASE_LEN, DictKey.CASE_SENTENCE])
        case_steps = case[DictKey.CASE_STEPS]
        edge_ids = [step[DictKey.EDGE_ID] for step in case_steps]
        brief_case[DictKey.CASE_STEPS] = edge_ids  # 只保留边 ID
        brief_cases.append(brief_case)
    return brief_cases


# =============== deprecated =================


def translate(src_lang, dst_lang, text):
    # 调用大模型翻译
    prompt = PROMPT.TRANSLATE.format(src_lang=src_lang, dst_lang=dst_lang, text=text)
    response = chat_with_llm(prompt=prompt)
    logger.log(f"tanslate \n{text}\n from '{src_lang}' to '{dst_lang}':\n{response}\n")
    return response
