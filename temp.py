# 禁止生成 __pycache__
import sys
sys.dont_write_bytecode = True

import os
from time import time
from utils.macro import DictKey

def test():
    from core.cog.page_cognition import get_cog_rects
    image_path = "temp/1.png"
    image_path = "tasks/task_2025-02-13_11-37-12/utg_data/1.png"
    rects = get_cog_rects(image_path=image_path)
    print(rects)
    pass


def test_yolo():
    from ultralytics import YOLO
    from utils.draw import draw_rects
    from core.yolo.yolo_process import yolo_detection
    image_path = "temp/temp_zy.png"
    model_path_yolo = "core/yolo/yolo_mdl.pt"
    model_det = YOLO(model_path_yolo, task='detect')  # Yolov8n
    boxes = yolo_detection(image_path=image_path, model=model_det)
    draw_rects(origin_image_path=image_path, target_image_path='temp', rects=boxes)


def test_cog():
    import logging
    logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
    logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
    from core.cog.page_cognition import get_cog_rects
    from utils.draw import draw_rects

    file_name = "0.png"
    input_dir = os.path.join("dataset", "Meituan")
    input_image_path = os.path.join(input_dir, file_name)
    output_dir = os.path.join("output", "Meituan")

    start_time = time()
    rects = get_cog_rects(image_path=input_image_path)
    draw_rects(image_path=input_image_path, output_dir=output_dir, rects=rects)
    end_time = time()
    print(f"test ui finish, cost {end_time - start_time} seconds")
    print(rects)


def test_shortest_path():
    from utils.utils import get_shortest_path
    utg_file_path = "utg.json"
    a, b = get_shortest_path(0, 5, utg_file_path)
    print(a,b)


def test_filter():
    image_path = "0.png"
    from utils.draw import draw_rects
    from core.cog.page_cognition import get_cog_rects
    rects = get_cog_rects(image_path=image_path)
    Som_file_path = draw_rects(image_path=image_path, output_dir=".", rects=rects)


def test_xml():
    from core.controller import AndroidController
    controller = AndroidController()
    xml_file = controller.get_xml(prefix="debug", save_dir=".")


def test_utg():
    from utils.utils import generate_utg
    work_dir = "tasks/task_2025-07-17_14-35-22"
    generate_utg(work_dir)


def test_get_rects():
    from core.cog.page_cognition import get_rects
    from utils.draw import draw_rects
    from core.controller import AndroidController
    controller = AndroidController()
    image_path = controller.get_screenshot(prefix="test", save_dir=".")
    xml_file = controller.get_u2_xml(prefix="test", save_dir=".")
    # image_path = "test.png"
    # rects = get_rects(image_path=image_path)
    # draw_rects(src_image_path=image_path, dst_image_path=image_path.
    # #replace(".png", "_rects.png"), rects=rects)


def test_u2():
    import uiautomator2 as u2
    device = "fb298d8e"
    u2_device = u2.connect_usb(device)
    xml_data = u2_device.dump_hierarchy()
    with open("xml.xml", "w", encoding="utf-8") as f:
        f.write(xml_data)


def test_draw():
    from utils.draw import draw_X, draw_arrow, draw_rect
    src_image_path = "test.png"
    dst_image_path = "test_x.png"
    point = (100, 100)
    start_point = (100, 100)
    end_point = (100, 200)
    rect = (100, 100, 200, 200)
    # draw_X(src_image_path, dst_image_path, point)
    # draw_arrow(src_image_path, dst_image_path, start_point, end_point)
    draw_rect(src_image_path, dst_image_path, rect)


def test_case_gen():
    from core.experiment.task_generator import get_adj_list, get_case_step
    from utils.utils import read_json_file, get_edges_by_ids
    utg_path = "utg.json"
    utg_data = read_json_file(utg_path)
    edges = utg_data["edges"]
    adj_list = get_adj_list(utg_data)
    start_node_id = 0
    incident_edges_id = adj_list[start_node_id]
    incident_edges = get_edges_by_ids(edges, incident_edges_id)
    
    step = get_case_step(utg=utg_data, current_page_id=start_node_id, generated_steps=[], generated_test_cases=[])
    print(adj_list)
    print(incident_edges)
    print(step)


def test_gen_test_case():
    from core.experiment.task_generator import get_test_case
    utg_file_path = "utg.json"
    case = get_test_case(utg_file_path, [], k=4)
    print(case)


def test_formatted_cases():
    from core.experiment.task_generator import format_cases
    from utils.utils import write_json_file
    full_cases_path = "full_cases.json"
    brief_cases_path = "brief_cases.json"
    utg_file_path = "utg.json"
    formatted_cases = format_cases(utg_file_path, full_cases_path)
    write_json_file(formatted_cases, brief_cases_path)


def test_heatmap():
    from core.experiment.task_generator import draw_cases_heatmap
    from utils.utils import read_json_file, write_json_file
    from utils.logger import task_dir
    cases_path = "full_cases.json"
    cases = read_json_file(cases_path)
    heatmap_path = os.path.join(task_dir, "heatmap.png")
    draw_cases_heatmap(cases, heatmap_path)


def test_case_distribution():
    from core.experiment.task_generator import get_data_distribution
    from utils.utils import write_json_file, read_json_file
    task_dir = "tasks/task"
    cases_path = os.path.join(task_dir, "full_cases.json")
    distribution_path = os.path.join(task_dir, "cases_distribution.json")
    cases = read_json_file(cases_path)
    distribution = get_data_distribution(cases)
    write_json_file(distribution, distribution_path)
    print(distribution)


def test_brief_cases():
    from core.experiment.task_generator import get_brief_cases
    full_cases_file = "full_cases.json"
    brief_cases_file = "brief_cases.json"
    get_brief_cases(full_cases_file, brief_cases_file)


def test_cal_avage():
    cases_file_path = "full_cases.json"
    data_file_path = "cases_data.json"
    from utils.utils import read_json_file, write_json_file
    from core.experiment.task_executor import get_case_by_id
    cases_data = read_json_file(data_file_path)
    cases = read_json_file(cases_file_path)
    for case_data in cases_data:
        case_id = case_data["case_id"]
        case = get_case_by_id(case_id, cases)
        case_time = case_data[DictKey.CASE_TIME]
        case_len = case[DictKey.CASE_LEN]
        average_step_time = case_time / case_len
        case_data[DictKey.AVERAGE_STEP_TIME] = average_step_time
    write_json_file(cases_data, data_file_path)


def test():
    from core.experiment.baseline.AppAgent.scripts.controller import execute_adb
    command = "adb -s fb298d8e shell uiautomator dump /sdcard/temp.xml"
    execute_adb(command)

def test_apk():
    from utils.utils import get_intent_from_apk
    apk_path = "apks\\Calendar.apk"
    intent = get_intent_from_apk(apk_path)
    print(intent)

def temp_write_result():
    from utils.utils import read_json_file, write_json_file
    task_dir = "bak\\开源new\\task_2025-12-30_15-59-31"
    results = []
    for filename in os.listdir(task_dir):
        if not os.path.isdir(os.path.join(task_dir, filename)):
            continue
        print("处理应用:", filename)
        work_dir = os.path.join(task_dir, filename)
        utg_file_path = os.path.join(work_dir, "utg.json")
        utg = read_json_file(utg_file_path)
        results.append({
            "app_name": filename,
            "node_count": len(utg["nodes"]),
            "edge_count": len(utg["edges"])
        })
    result_json_path = "GraphDroid.json"
    result_file_path = "GraphDroid.xlsx"
    write_json_file(results, result_json_path)
    from utils.utils import write_json_to_excel
    write_json_to_excel(result_json_path, result_file_path)


def test_page_mapping():
    from utils.utils import write_page_mapping
    work_dir = "bak\\meituan_100_3"
    excel_file_path = os.path.join(work_dir, "page_mapping.xlsx")
    write_page_mapping(work_dir, excel_file_path)
    

def test_get_task_data_dirs():
    base_dir = "bak\\消融\\消融2"
    excel_file = os.path.join(base_dir, "all_data.xlsx")
    from utils.utils import get_task_data_dirs, write_json_to_excel
    all_data = get_task_data_dirs(base_dir, start_index=0, task_num=20)
    write_json_to_excel(json_file=None, json_data=all_data, excel_file=excel_file)


if __name__ == "__main__":
    # test_cal_jsonl()
    # test_utg()
    test_get_task_data_dirs()
    pass