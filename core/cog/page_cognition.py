import cv2
import os
import json
import numpy as np
import Levenshtein
import torch  # 要有这个，否则PaddleOCR报错
from paddleocr import PaddleOCR
from .Rect import Rect
from .XML import XML
import logging
from core.yolo.yolo_process import get_yolo_rects

# 关闭ocr的logging
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)

# 获取当前脚本文件目录
script_dir = os.path.dirname(os.path.abspath(__file__))
blank = 5
text_widths, text_height = [20, 40, 60], 35
font_path = os.path.join(script_dir, "Assets", "Arial.ttf")
OVERLAP_THRESHOLD = 0.9  # 重叠度阈值


# ==================== utils ====================


def calculate_levenshtein_similarity(str1, str2):
    # 计算编辑距离
    distance = Levenshtein.distance(str1, str2)
    # 根据编辑距离计算相似度
    similarity = 1 - (distance / max(len(str1), len(str2)))
    return similarity


def relative_overlap_area(a, b):
    # 对于a,b两个rect[x1, y1, x2, y2], 计算 a 和 b 重叠区域占 a 面积的百分比
    a_x1, a_y1, a_x2, a_y2 = a
    b_x1, b_y1, b_x2, b_y2 = b
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)

    # 计算 a 和 b 的重叠区域
    overlap_x1 = max(a_x1, b_x1)
    overlap_y1 = max(a_y1, b_y1)
    overlap_x2 = min(a_x2, b_x2)
    overlap_y2 = min(a_y2, b_y2)

    # 如果没有重叠区域，返回 0
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0

    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    overlap_percentage = overlap_area / a_area
    return overlap_percentage


def get_ocr_result(image, lang="ch"):
    """
    利用 paddleocr 识别文本信息
    对于 paddleocr, image 参数可以是[图片路径字符串, PIL图片, cv2 图片, numpy数组], 及其混合组成的列表
    return: List[Dict]
    { "text": "xxx", "bounds": [x1, y1, x2, y2]}
    """
    ocr = PaddleOCR(use_angle_cls=False, lang=lang, enable_mkldnn=False)
    ocr_results = ocr.ocr(image, cls=False)[0]  # 一次只有一张图片
    ocr_res = []
    if not ocr_results:
        return []
    
    for ocr_result in ocr_results:
        # 单个 ocr_result 形式为 [coordinates, text]
        text = ocr_result[1][0]  # ocr_result[1] 形式为: (text:str, conf:float), conf 为置信度
        conf = ocr_result[1][1]
        if conf < 0.9:
            continue
        coordinates = ocr_result[0]  # coordinates 形式为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], 为左上,右上,右下,左下绝对坐标
        x1 = round(min(coordinates[0][0], coordinates[3][0]))
        y1 = round(min(coordinates[0][1], coordinates[1][1]))
        x2 = round(max(coordinates[1][0], coordinates[2][0]))
        y2 = round(max(coordinates[2][1], coordinates[3][1]))
        bounds = [x1, y1, x2, y2]
        ocr_res.append({"class": "OCRText", "text": text, "bounds": bounds})
    
    return ocr_res


# ==================== 主流程 ====================


def filter_status_bar(image_path, rects):
    # 过滤状态栏, 以屏幕高度的 0.04 为界限, 元素下沿小于该限的认为是状态栏元素
    filtered_rects = []
    image = cv2.imread(image_path)
    height = image.shape[0]
    height_limit = round(0.04 * height)
    for rect in rects:
        if rect['bounds'][3] < height_limit:
            continue
        filtered_rects.append(rect)
    return filtered_rects


def filter_invisible_xml_rects(image_path, rects):
    # 利用 ocr, 过滤不可见的组件
    ret_rects = []
    for rect in rects:
        rect_img = Rect.crop_image(image_path, rect['bounds'])
        if rect_img is None:
            ret_rects.append(rect)
            continue
        
        # 注意，由于是识别的组件截图，所以坐标是相对坐标，需要转换
        ocr_results = get_ocr_result(image=rect_img)
        for ocr_result in ocr_results:  # for 循环自动忽略空列表
            text = ocr_result['text']
            rel_bounds = ocr_result['bounds']
            x1 = rect['bounds'][0] + rel_bounds[0]
            y1 = rect['bounds'][1] + rel_bounds[1]
            x2 = rect['bounds'][0] + rel_bounds[2]
            y2 = rect['bounds'][1] + rel_bounds[3]   
            bounds = [x1, y1, x2, y2]
            ocr_result['bounds'] = bounds  # 覆写原始结果，相对坐标转换为绝对坐标

        # 如果 xml 框本身就包含 text 属性, 则用 ocr 结果去匹配, 看看是否需要加入结果列表(相当于过滤)
        if 'text' in rect.keys():
            # 如果xml 有文本，则过滤掉和 OCR 结果相似度低的框
            if isinstance(rect['text'], str):  # xml_rect['text'] 可能为 str 或 list
                for result in ocr_results:
                    # 如果 编辑距离 超过阈值, 则加入结果列表(保留)
                    if calculate_levenshtein_similarity(rect['text'], result['text']) > 0.5:
                        ret_rects.append(rect)
                        break
            else:  # 文本为列表
                for text in rect['text']:
                    for result in ocr_results:
                        if calculate_levenshtein_similarity(text, result['text']) > 0.5 or text in result['text']:
                            if rect not in ret_rects:  # 防止重复添加
                                ret_rects.append(rect)
                            break
        # 如果 xml 框本身不包含文本, 则直接加入结果列表(保留)
        else:
            ret_rects.append(rect)
    return ret_rects


def get_unmatched_text(image_path, rects):
    # 识别其余未在 xml 框中匹配的文本信息
    unmatched_text = []
    ocr_results = get_ocr_result(image_path)
    for ocr_result in ocr_results:
        add_ocr = True
        bounds = ocr_result['bounds']
        for rect in rects:
            # 如果 OCR 框包含了目前已有的框，或者OCR 框的 70% 以上被包含，则忽略。
            if Rect.is_containing(bounds, rect['bounds']) or Rect.intersection_over_second_area(rect['bounds'], bounds) > 0.7:
                add_ocr = False
                break
        if add_ocr:
            unmatched_text.append(ocr_result)

    return unmatched_text


def filter_rects_by_edge_detection(image_path, rects):
    """
    使用边缘检测过滤矩形框。
    如果边缘不存在/距离 bounds 较远，则认为被遮挡/实际不存在
    """
    low_threshold = 50
    high_threshold = 100
    edge_presence_threshold = 100

    filtered_rects = []
    img_cv = cv2.imread(image_path)

    for rect in rects:
        bounds = rect['bounds']
        x1, y1, x2, y2 = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        element_img = img_cv[y1:y2, x1:x2]  # 裁剪目标元素
        gray_img = cv2.cvtColor(element_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, low_threshold, high_threshold)

        if np.sum(edges) > edge_presence_threshold:
            edge_positions = np.column_stack(np.where(edges > 0))
            
            # 垂直方向的坐标值从上往下增加。也就是说，y 坐标值越小，位置越靠近图像的顶部。
            edge_top = np.min(edge_positions[:, 0])  # 边缘在垂直方向上的最小坐标值。
            edge_bottom = np.max(edge_positions[:, 0])  # 边缘在垂直方向上的最大坐标值。
            edge_left = np.min(edge_positions[:, 1])  # 边缘在水平方向上的最小坐标值。
            edge_right = np.max(edge_positions[:, 1])   # 边缘在水平方向上的最大坐标值。

            # TODO:
            # 如果线条在图像边界上被检测到，并且仅有极少数 pixel 出现在边界上，说明可能存在贯穿元素的线条，则表示该元素被可能被遮挡
            # 边缘像素阈值设置为总边缘像素数量的 1%
            edge_pixel_threshold = edges.shape[0] * edges.shape[1] * 0.01
            
            filtered_rects.append(rect)
    return filtered_rects


def get_cog_rects(image_path):
    """
    识别页面组件。
    returns: List[Dict], 诸如: 
    {'class': 'View', 'content_desc': '酒店民宿', 'bounds': [436, 328, 645, 516], 'id_bounds': [436, 293, 461, 328]}
    """
    try:
        # 从 xml 获取元素框
        xml_path = os.path.splitext(image_path)[0] + ".xml"
        xml = XML(xml_path)
        xml_rects = xml.group_interactive_nodes()  # time_cost < 0.01s
    except:
        xml_rects = []

    # 过滤不可见组件，主要用时在这里，因为对每个 rect 都要 OCR 一次
    # filtered_xml_rects = filter_invisible_xml_rects(image_path=image_path, rects=xml_rects)  # OCR
    filtered_xml_rects = xml_rects  # 先跳过最耗时的过滤这一步
    # 补充未匹配的文本框
    ocr_rects = get_unmatched_text(image_path=image_path, rects=xml_rects)  # OCR
    rects = filtered_xml_rects + ocr_rects

    # 边缘检测过滤
    rects = filter_rects_by_edge_detection(image_path=image_path, rects=rects)  # time_cost: 0.1~0.2s
    # 过滤状态栏
    rects = filter_status_bar(image_path=image_path, rects=rects)

    rects = sorted(rects, key=lambda r: (r['bounds'][1], r['bounds'][0]))
    return rects


def get_rects(image_path):
    rects, webviews = [], []
    cog_rects = get_cog_rects(image_path)
    yolo_rects = get_yolo_rects(image_path)
    for rect in cog_rects:
        if rect['class'] == 'WebView':
            webviews.append(rect)

    # 如果 cog 识别出的元素不包含在任何一个 webview 中，则保留
    for rect in cog_rects:
        if (rect['class'] != 'WebView' and
            all(relative_overlap_area(rect['bounds'], webview['bounds']) < OVERLAP_THRESHOLD for webview in webviews)):
                rects.append(rect)

    # 将 yolo 识别出的位于 webview 中的元素加入
    for rect in yolo_rects:
        if any(relative_overlap_area(rect['bounds'], webview['bounds']) > OVERLAP_THRESHOLD for webview in webviews):
            rects.append(rect)

    return rects


# ==================== 数据处理相关 ====================


def format_rects(rects):
    # 格式化 rects, 增加 text 属性
    for i, rect in enumerate(rects):
        rect['shouldVisit'] = True
        rect['isVisited'] = False
        # rect = {'node_id': i, **rect}  # 使用解包将 node_id 放在第一顺位
        rects[i] = rect
    return rects


def write_node_list(node_list, output_json: str):
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(node_list, f, ensure_ascii=False, indent=2)


def read_node_list(node_list_file: str):
    with open(node_list_file, 'r', encoding='utf-8') as f:
        node_list = json.load(f)
    return node_list


def get_node_by_id(utg_data_dir, utg_page_id: int, node_id: int):
    # 根据 node_id 获取对应的节点
    node_list_file = os.path.join(utg_data_dir, f"{utg_page_id}.json")
    node_list = read_node_list(node_list_file)
    return node_list[node_id]


def mark_node_visited(utg_data_dir, utg_page_id, node):
    # 标记节点已访问
    if not node:
        return -1
    node_id = int(node['node_id'])
    node_list_file = os.path.join(utg_data_dir, f"{utg_page_id}.json")
    node_list = read_node_list(node_list_file)
    node_list[node_id]['isVisited'] = True
    write_node_list(node_list, node_list_file)
    
    
def write_node_summary(utg_data_dir, utg_page_id, node, summary):
    # 写入llm对node操作的总结
    node_list_file = os.path.join(utg_data_dir, f"{utg_page_id}.json")
    node_list = read_node_list(node_list_file)
    node_id = int(node['node_id'])
    node_list[node_id]['summary'] = summary
    write_node_list(node_list, node_list_file)
