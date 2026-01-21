from core.experiment.task_generator import *
from core.experiment.task_executor import TaskExecutor
from utils.logger import logger, task_dir


def run_distribution():
    from utils.utils import read_json_file
    src_dir = "bak\\broccoli"
    full_cases_file = os.path.join(src_dir, "full_cases.json")
    distribution_file_path = os.path.join(src_dir, "cases_distribution.json")
    image_file_path = os.path.join(src_dir, "cases_heatmap.png")
    full_cases = read_json_file(full_cases_file)
    distribution = get_data_distribution(full_cases)
    write_json_file(distribution, distribution_file_path)
    draw_cases_heatmap(full_cases, image_file_path)


def run_preprocess():
    # 预处理，一般来讲需要单独调用，直接生成在目标目录
    src_utg_path = "bak\\broccoli\\utg.json"
    dst_utg_file = src_utg_path.replace(".json", "_processed.json")
    preprocess_utg(src_utg_path, dst_utg_file)


def run_cases():
    # 生成对应case，需要设置生成策略，设定对应的起始id，生成在对应时间戳目录下
    src_utg_path = "bak\\broccoli\\utg_processed.json"
    case_num = 20
    base_ids = [0, 3, 6, 7]
    start_page_ids = [0, 0, 0, 0, 0, 3, 0, 6, 0, 7, 0, 0, 0, 3]
    task_generator = TaskGenerator(
        src_utg_path=src_utg_path, logger=logger, task_dir=task_dir,
        policy=POLICY_PURE_RANDOM, case_num=case_num, base_ids=base_ids, start_page_ids=start_page_ids)
    task_generator.generate_test_cases()


def run_human_cases():
    processed_utg_file = "bak/utg_processed.json"
    cases_ids_file = "bak/cases_human.json"

    full_cases = generate_cases_by_human(processed_utg_file, cases_ids_file)
    full_cases_path = os.path.join(task_dir, "full_cases.json")
    distribution_file_path  = os.path.join(task_dir, "distribution.json")
    image_file_path = os.path.join(task_dir, "cases_heatmap.png")
    write_json_file(full_cases, full_cases_path)
    distribution = get_data_distribution(full_cases)
    write_json_file(distribution, distribution_file_path)
    draw_cases_heatmap(full_cases, image_file_path)


def run_executor():
    app_name = 'meituan'
    full_cases_file = "bak/full_cases.json"
    utg_file_path = "bak/utg.json"

    import utils.chat as chat
    chat.CHAT_MODE = chat.CHAT_MT
    executor = TaskExecutor(
        task_dir=task_dir, cases_file_path=full_cases_file,
        src_utg_path=utg_file_path, app_name=app_name)
    executor.run_all()
    # cases = read_json_file(full_cases_file)
    # executor.run_single(cases[6], [])
    # print(executor.check_cases())
    # executor.test_pre_execute()


def run_app_agent():
    from core.experiment.baseline.AppAgent.AppAgent import AppAgent
    app_name = 'meituan'
    full_cases_file = "bak/full_casesjson"
    utg_file_path = "bak/utg.json"
    finished_cases_file = "bak/exe_anki/cases_data.json"

    executor = AppAgent(
        task_dir=task_dir, cases_file_path=full_cases_file,
        src_utg_path=utg_file_path, app_name=app_name)
    start_case_id = 0
    executor.run_all(start_case_id=start_case_id, finished_cases_file=finished_cases_file)


def run_case_map():
    # 边映射
    from core.experiment.case_map import run_case_map
    import utils.chat as chat
    chat.CHAT_MODE = chat.CHAT_MT

    utg_file_path = "bak/utg.json"
    full_cases_file = "bak/full_cases.json"

    start_case_id = 0
    run_case_map(utg_file_path, full_cases_file, start_case_id)


def write_excel():
    # 映射结果写入excel
    from utils.utils import write_json_to_excel, read_json_file, write_json_file
    print("write")
    data_file_path = "bak/case_map.json"
    
    excel_file_path = data_file_path.replace('.json', '.xlsx')
    data = read_json_file(data_file_path)
    # 按照 case_id 排序
    data.sort(key=lambda x: x["case_id"])
    write_json_file(data, data_file_path)
    write_json_to_excel(data_file_path, excel_file_path)


def experiment():
    # run_preprocess()
    run_cases()
    # run_human_cases()
    # run_distribution()
    # run_executor()
    # run_app_agent()
    # run_case_map()
    # write_excel()
    pass
