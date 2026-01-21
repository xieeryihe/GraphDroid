import json
import os
import time
from utils.utils import read_json_file, write_json_file, get_edges_by_ids, get_pages_by_ids, remove_keys
from utils.macro import DictKey, LOG_SEPARATOR
from utils.logger import task_dir, logger
from utils.chat import chat_with_llm, parse_response
# 用于将 case 映射到 utg 上

PROMPT = """
    You are a professional Android application test engineer.
    I will provide you with a UI transition graph (UTG) and a test case.
    The UTG is composed of nodes and edges.
    Each node represents a page in the application, and each edge represents a transition(action) from one page to another.
    The test case is a sequence of actions with information of page and element.
    
    Your task is to map each action of the test case to an edge of the UTG, and respond a list of 'edge_id'.
    
    The UTG is as follows:
    <UTG>
    {utg}
    </UTG>
    The test cases are as follows:
    
    <Test Case>
    {case}
    </Test Case>

    <Note>
    1. The mapped edges in UTG mast be continuous.
    2. Give -1 if an action cannot be mapped to any edge of the UTG.
    3. Return the list with the following format: [edge_id1, edge_id2, ...], do not return other words.
"""


def get_sub_utg(utg_file_path, cases_file_path):
    # 先获取 cases 中包含的所有 edge 和 page 的 id，并过滤无用信息
    cases = read_json_file(cases_file_path)
    edge_ids, page_ids = set(), set()
    for case in cases:
        case_steps = case[DictKey.CASE_STEPS]
        for step in case_steps:
            edge_ids.add(step[DictKey.EDGE_ID])
            page_ids.add(step[DictKey.FROM])
            page_ids.add(step[DictKey.TO])
    logger.log(f"sub utg info:\nedge_ids: {edge_ids}\npage_ids: {page_ids} ")
    # 再从 utg 中过滤所需要的 edge 和 page
    utg = read_json_file(utg_file_path)
    sub_utg = {}
    pages = get_pages_by_ids(utg[DictKey.NODES], page_ids)
    remove_page_keys = [DictKey.IMAGE_PATH, DictKey.NODE_COUNT, DictKey.FINISHED]
    _ = [remove_keys(page, remove_page_keys) for page in pages]

    edges = get_edges_by_ids(utg[DictKey.EDGES], edge_ids)
    remove_edge_keys = [DictKey.IMAGE_PATH, DictKey.NODE_ID, DictKey.BOUNDS]
    _ = [remove_keys(edge, remove_edge_keys) for edge in edges]
    
    sub_utg[DictKey.NODES] = pages
    sub_utg[DictKey.EDGES] = edges
    return sub_utg


# 获取实际case id 的 oracle
def get_oracle(case):
    case_steps = case[DictKey.CASE_STEPS]
    oracle = []
    for step in case_steps:
        oracle.append(step[DictKey.EDGE_ID])
    return oracle


def cal_correct_ratio(target, oracle):
    if not isinstance(target, list):
        return 0.0
    n = min(len(target), len(oracle))
    for i in range(n):
        if target[i] != oracle[i]:
            return i / n
    return 1.0


def run_case_map(utg_file_path, cases_file_path, start_case_id):
    data_file_path = os.path.join(task_dir, "case_map.json")
    # utg = get_sub_utg(utg_file_path, cases_file_path)
    utg = read_json_file(utg_file_path)
    cases = read_json_file(cases_file_path)
    data = []
    try:
        for case in cases:
            time.sleep(7)
            case_time_start = time.time()
            if case[DictKey.CASE_ID] < start_case_id:
                continue
            case_id = case[DictKey.CASE_ID]
            case_sentence = case[DictKey.CASE_SENTENCE]
            logger.log(f"{LOG_SEPARATOR}case id: {case_id}{LOG_SEPARATOR}\ncase_sentence: {case_sentence}\n")
            prompt = PROMPT.format(utg=utg, case=case_sentence)
            response = chat_with_llm(prompt)
            ans = parse_response(response)
            res = {DictKey.CASE_ID: case_id, **ans}
            try:
                response_content = json.loads(response_content)
                res[DictKey.RESPONSE_CONTENT] = response_content
                oracle = get_oracle(case)
                res[DictKey.ORACLE] = oracle
                correct_ratio = cal_correct_ratio(response_content, oracle)
                res[DictKey.CORRECT_RATIO] = correct_ratio
            except Exception as e:
                logger.log(f"Error: {e}")
                logger.log(f"Can not convert response_content to list: {response_content}")
            case_time_end = time.time()
            res[DictKey.TIME] = case_time_end - case_time_start
            logger.log(f"case map result:\n{res}\n\n", log_terminal=False)  # 带 prompt 的太长了就不打印到终端了
            res.pop(DictKey.PROMPT)
            logger.log(f"case map result:\n{res}\n\n")
            data.append(res)
    except Exception as e:
        logger.log(f"Error: {e}")
    finally:
        write_json_file(data, data_file_path)
    return data
