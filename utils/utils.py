# 禁止生成 __pycache__ 
import sys
sys.dont_write_bytecode = True

import json
import os
import shutil
import re
import cv2
import random
import copy
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image, ImageDraw
from glob import glob
from utils.macro import *
from collections import deque, defaultdict
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scient.image import hash
from scient.algorithm import similar
from core.cog.page_cognition import *
from utils.page_comparator import cal_layout_similarity, cal_page_iou
from core.utg.prompt import ExplorerPrompt
from utils.logger import logger
from datetime import datetime
import xml.etree.ElementTree as ET


SIMILAR_THRESHOLD = 0.7  # 图片布局相似度阈值
SAME_THRESHOLD = 0.95  # 图片布局相同阈值
IOU_THRESHOLD = 0.7  # 元素重叠阈值

# 手机尺寸
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 2340


# ==================== 参数配置相关 ====================


def get_intent_from_apk(apk_path):
    # 使用 androguard 从 APK 文件中提取package和主Activity，返回格式为 "package/主Activity"
    try:
        package_name = None
        main_activity = None
        if not (package_name and main_activity):
            try:
                from androguard.misc import AnalyzeAPK
                apk, _, _ = AnalyzeAPK(apk_path)
                package_name = apk.get_package()
                
                # 获取 main activity
                main_activities = apk.get_main_activities()
                if main_activities:
                    main_activity = list(main_activities)[0]
                    
            except Exception as e:
                logger.log(f"Warning: Failed to parse APK with androguard: {e}")
        
        if package_name and main_activity:
            # 如果 activity 不是全限定名，则补全为全限定名
            # 情况1: 以'.'开头的相对名称 -> package_name + main_activity
            # 情况2: 不含'.'的简单名称 -> package_name + '.' + main_activity
            # 情况3: 已含'.'的全限定名 -> 直接使用
            if main_activity.startswith('.'):
                main_activity = package_name + main_activity
            elif '.' not in main_activity:
                main_activity = package_name + '.' + main_activity
            # else: main_activity 已经是全限定名，直接使用
            
            intent = f"{package_name}/{main_activity}"
            logger.log(f"Successfully extracted intent from {apk_path}: \n{intent}")
            return intent
        else:
            logger.log(f"Warning: Could not extract package_name ({package_name}) or main_activity ({main_activity}) from {apk_path}")
            # 返回至少提取到的信息
            if package_name:
                return f"{package_name}/com.example.MainActivity"  # 使用默认 activity
            return None
        
    except Exception as e:
        logger.log(f"Error parsing APK file {apk_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_pre_made_intent(app_name):
    intent = ""
    if app_name == "meituan":
        intent = APPInfo.INTENT_MEITUAN
    elif app_name == "xhs":
        intent = APPInfo.INTENT_XHS
    elif app_name == "dianping":  # 大众点评
        intent = APPInfo.INTENT_DIANPING
    elif app_name == "Broccoli":
        intent = APPInfo.INTENT_BROCCOLI
    elif app_name == "note":
        intent = APPInfo.INTENT_NOTE
    elif app_name == 'google_news':
        intent = APPInfo.INTENT_NEWS
    elif app_name == 'AntennaPod':
        intent = APPInfo.INTENT_ANTENNA_POD
    elif app_name == 'anki':
        intent = APPInfo.INTENT_ANKI
    elif app_name == 'amaze':
        intent = APPInfo.INTENT_AMAZE
    elif app_name == 'memo':
        intent = APPInfo.INTENT_MEMO
    elif app_name == 'brain':
        intent = APPInfo.INTENT_BRAIN
    return intent


def get_package(intent):
    return intent.split("/")[0]


def get_activity(intent):
    return intent.split("/")[1]


def ensure_dir(path):
    """确保目录存在"""
    try:
        os.makedirs(path, exist_ok=True)  # exist_ok=True 表示如果目录已经存在，则什么都不做
        # print(f"Directory '{path}' created successfully.")
    except Exception as e:
        print(f"An error occurred while creating directory '{path}': {e}")


# ==================== 主流程相关 ====================


def copy_file_if_exists(src_path, dst_path):
    """如果源文件存在，则复制到目标路径"""
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        return dst_path
    return None


def copy_files(src_dir, src_page_id, dst_dir, dst_page_id):
    # 复制相关文件"""
    suffix = ["png", "layout.png", "SoM.png"]  # 后缀是除了数字编号和'_'的内容，比如1_layout.png

    for file_type in suffix:
        src_path = os.path.join(src_dir, f"{src_page_id}_{file_type}" if "." in file_type else f"{src_page_id}.{file_type}")
        dst_path = os.path.join(dst_dir, f"{dst_page_id}_{file_type}" if "." in file_type else f"{dst_page_id}.{file_type}")
        copied_file = copy_file_if_exists(src_path, dst_path)
        if not copied_file:
            logger.log(f"Copy file failed: {src_path}.")


def filter_edge_detection(image_path, node_list_file):
    # 通过边缘检测过滤图片
    node_list = read_node_list(node_list_file)
    for node in node_list:
        bounds = node['attributes']['bounds']
        if bounds == '[635,699][784,760]':
            print(node)
        if not should_save_by_edge_detection(image_path=image_path, bounds=bounds):
            node['shouldVisit'] = False
    # 写回修改后的数据
    write_node_list(node_list, node_list_file)
    return node_list


def should_save_by_edge_detection(image_path, bounds):
    """
    使用边缘检测过滤矩形框，如果边缘不存在/距离 bounds 较远，则认为被遮挡/实际不存在。
    bounds 是形如 "[0,0][1440,3200]" 的字符串
    return: True, 保留. False, 过滤.
    """
    if not image_path or not bounds:
        return False

    low_threshold = 50
    high_threshold = 100
    edge_presence_threshold = 100

    img_cv = cv2.imread(image_path)

    x1, y1, x2, y2 = map(int, re.findall(r'\d+', bounds))
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return False
    element_img = img_cv[y1:y2, x1:x2]  # 裁剪目标元素
    gray_img = cv2.cvtColor(element_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)

    if np.sum(edges) > edge_presence_threshold:
        # 如果存在边缘，则认为该元素正常显示
        edge_positions = np.column_stack(np.where(edges > 0))
        
        # 垂直方向的坐标值从上往下增加。也就是说，y 坐标值越小，位置越靠近图像的顶部。
        edge_top = np.min(edge_positions[:, 0])  # 边缘在垂直方向上的最小坐标值。
        edge_bottom = np.max(edge_positions[:, 0])  # 边缘在垂直方向上的最大坐标值。
        edge_left = np.min(edge_positions[:, 1])  # 边缘在水平方向上的最小坐标值。
        edge_right = np.max(edge_positions[:, 1])   # 边缘在水平方向上的最大坐标值。

        # TODO: 边缘检测优化
        # 如果线条在图像边界上被检测到，并且仅有极少数 pixel 出现在边界上，说明可能存在贯穿元素的线条，则表示该元素被可能被遮挡
        # 边缘像素阈值设置为总边缘像素数量的 1%
        edge_pixel_threshold = edges.shape[0] * edges.shape[1] * 0.01
        
        return True
    return False


def handle_recyclerView(node, node_list):
    def setShouldVisitFalse(node):
        # 递归设置 shouldVisit 为 False
        node['shouldVisit'] = False
        for child in node.get('children', []):
            child_node = node_list[child]
            setShouldVisitFalse(child_node)

    def setShouldVisitTrue(node):
        # 递归设置 shouldVisit 为 True
        node['shouldVisit'] = True
        for child in node.get('children', []):
            child_node = node_list[child]
            setShouldVisitTrue(child_node)
    
    # # 保留第一个子节点及其后代的 shouldVisit 为 True，其余节点及其后代的 shouldVisit 为 False
    # children = node.get('children', [])
    # # 处理 children 为空的情况
    # if not children:
    #     # 如果没有子节点，将当前节点的 shouldVisit 设置为 True
    #     node['shouldVisit'] = True
    # else:
    #     # 有子节点的情况
    #     if len(children) > 1:
    #         for child in children[1:]:
    #             child_node = node_list[child]
    #             setShouldVisitFalse(child_node)
    #     first_child = node_list[children[0]]
    #     setShouldVisitTrue(first_child)
    #
    # return node_list
    '''保留recyclerview中结构不同的子节点'''
    # 布局集合
    view_groups = ["android.support.v7.widget.LinearLayoutCompat", "android.widget.HorizontalScrollView",
                   "android.widget.GridView", "androidx.drawerlayout.widget.DrawerLayout",
                   "android.widget.RelativeLayout", "androidx.recyclerview.widget.RecyclerView",
                   "com.google.android.material.card.MaterialCardView", "android.view.ViewGroup",
                   "android.widget.FrameLayout", "android.widget.LinearLayout",
                   "android.support.v7.widget.RecyclerView"]
    # 比较节点结构，相同返回true
    def cmp_structure(node1, node2):
        attributes1 = node1['attributes']
        attributes2 = node2['attributes']
        if attributes1['class'] != attributes2['class'] or attributes1['resource-id'] != attributes2['resource-id']:
            return False
        if attributes1['class'] in view_groups:  #若节点是布局类，比较他们是否是最后一个布局节点（非叶节点）
            if is_last_container(node1, node_list) and is_last_container(node2, node_list):
                return True

        child_ids1 = node1.get('children', [])
        child_ids2 = node2.get('children', [])
        if len(child_ids1) != len(child_ids2):
            return False

        for id1, id2 in zip(child_ids1, child_ids2):
            child1 = node_list[id1]
            child2 = node_list[id2]
            if not cmp_structure(child1, child2):
                return False
        return True
    def is_last_container(node, node_list):
        for i in node.get('children', []):
            child = node_list[i]
            if len(child.get('children', []))>0:
                return False
        return True
    # 筛选recyclerView的子节点
    remained_nodes = [] # 保留节点
    child_ids = node.get('children', [])
    for id in child_ids:
        child = node_list[id]
        if child['attributes']['class'].split('.')[-1] == 'RecyclerView':
            handle_recyclerView(child, node_list)
        flag = False
        for node in remained_nodes:
            if cmp_structure(node, child):
                flag = True
                setShouldVisitFalse(node) # 剔除与已有节点结构相同的节点
                break
        if not flag:
            setShouldVisitTrue(child)
            remained_nodes.append(child)
    return node_list


def get_back_edge(from_page_id, to_page_id, utg_file_path):
    """ 获取从 from 到 to 的第一条边，以及距离目标节点的距离 """
    # 构建邻接表
    adjacency_list = {}
    try:
        utg = read_json_file(utg_file_path)
    except FileNotFoundError:
        return None, -1
    edges = utg.get('edges', [])
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        if from_node not in adjacency_list:
            adjacency_list[from_node] = []
        adjacency_list[from_node].append(edge)
    
    # BFS 初始化
    queue = deque([(from_page_id, None, 0)])  # (当前节点, 第一条边, 距离)
    visited = set()
    
    while queue:
        current_node, first_edge, distance = queue.popleft()
        if current_node in visited:
            continue
        visited.add(current_node)
        
        # 如果到达目标节点，返回第一条边和距离
        if current_node == to_page_id:
            return first_edge, distance
        
        # 遍历相邻节点
        for edge in adjacency_list.get(current_node, []):
            next_node = edge["to"]
            if next_node not in visited:
                # 如果是起点，记录第一条边
                if current_node == from_page_id:
                    queue.append((next_node, edge, distance + 1))
                else:
                    queue.append((next_node, first_edge, distance + 1))
    
    # 如果未找到路径，返回 None 和 -1（表示没有找到路径）
    return None, -1


def get_shortest_path(from_page_id, to_page_id, utg_file_path=None, utg=None):
    # BFS 获取从 from_page_id 到 to_page_id 的最短路径, 返回 pages 和 edges 的 id
    # 优先使用传入的 utg 参数，否则从文件读取
    if utg_file_path is None and utg is None:
        raise ValueError("Either <utg_file_path> or <utg> must be provided.")
    utg = utg if utg is not None else read_json_file(utg_file_path)

    # 构建邻接表
    adjacency_list = {}
    edges = utg.get(DictKey.EDGES, [])
    for edge in edges:
        from_node = edge[DictKey.FROM]
        to_node = edge[DictKey.TO]
        if from_node not in adjacency_list:
            adjacency_list[from_node] = []
        adjacency_list[from_node].append(edge)

    queue = deque([(from_page_id, [from_page_id], [])])  # (当前节点, 当前路径的页面, 当前路径的边)
    visited = set()

    while queue:
        current_node, current_pages, current_edges = queue.popleft()
        if current_node in visited:
            continue
        visited.add(current_node)

        # 如果到达目标节点，返回当前路径的页面和边
        if current_node == to_page_id:
            edge_ids = [edge[DictKey.EDGE_ID] for edge in current_edges]
            return current_pages, edge_ids

        # 遍历相邻节点
        for edge in adjacency_list.get(current_node, []):
            next_node = edge[DictKey.TO]
            if next_node not in visited:
                # 更新路径
                next_pages = current_pages + [next_node]
                next_edges = current_edges + [edge]
                queue.append((next_node, next_pages, next_edges))
    return [], []


def is_same_node(node1, node2):
    node1_bounds = node1[DictKey.BOUNDS]
    node1_class = node1[DictKey.CLASS]
    node2_bounds = node2[DictKey.BOUNDS]
    node2_class = node2[DictKey.CLASS]
    iou = bounds_iou(node1_bounds, node2_bounds)
    # TODO: 这里的阈值可以调大一点，或者增加判断条件，比如要求 text/description 相同等
    if iou > IOU_THRESHOLD and node1_class == node2_class:
        return True
    return False


# =============== 绘图记录相关 =================


def draw_image(src_image_dir, image_name, action, position):
    """
    绘制操作的图片
    src_image_dir: 原始图片的目录，image_name 是纯名字，不带后缀，一般是 current_step 纯数字，比如 1,2，表示执行的步骤
    """

    # coordinates 一般形式为 [a,b] 这样的坐标
    def draw_X(draw, coordinates, cross_length=18, width=10, fill='yellow'):
        # 画X形标记
        draw.line(
            [(coordinates[0] - cross_length, coordinates[1] - cross_length), (coordinates[0] + cross_length, coordinates[1] + cross_length)],
            fill=fill,
            width=width
        )
        draw.line(
            [(coordinates[0] - cross_length, coordinates[1] + cross_length), (coordinates[0] + cross_length, coordinates[1] - cross_length)],
            fill=fill,
            width=width
        )

    def draw_arrow(draw, coordinates, fill='yellow'):
        # 画一个箭头
        # 画一条线然后画一个叉，表示方向即可
        draw.line(coordinates, fill=fill, width=7)
        draw_X(draw=draw, coordinates=coordinates[1], fill=fill)

    src_image_path = os.path.join(src_image_dir, f"{image_name}.png")
    dst_image_path = os.path.join(src_image_dir, f"{image_name}_opt.png")


    image = Image.open(src_image_path)
    draw = ImageDraw.Draw(image)
    if action == ACTION.CLICK:
        draw_X(draw=draw, coordinates=position)
    # elif data[DictKey.ACTION] == "long_click":
    #     if is_relative:
    #         coordinates = [int(coordinates[0]*SCREEN_WIDTH), int(coordinates[1]*SCREEN_HEIGHT)]
    #     draw_X(draw=draw, coordinates=coordinates, fill='blue')
    # elif data[DictKey.ACTION] == "scroll":
    #     if is_relative:
    #         start = (int(coordinates[0][0]*SCREEN_WIDTH), int(coordinates[0][1]*SCREEN_HEIGHT))  # 转换为元组
    #         end = (int(coordinates[1][0]*SCREEN_WIDTH), int(coordinates[1][1]*SCREEN_HEIGHT))
    #     else:
    #         start = (coordinates[0][0], coordinates[0][1])
    #         end = (coordinates[1][0], coordinates[1][1])
    #     draw_arrow(draw=draw, coordinates=(start, end))
    # else:
    #     action = data[DictKey.ACTION]
    #     raise ValueError(f"Error: action {action} is invalid.")

    image.save(dst_image_path)


def draw_rectangle(src_image_dir, image_num, bounds):
    # 绘制矩形框, bounds 为 [x1, y1, x2, y2] 形式
    src_image_path = os.path.join(src_image_dir, f"{image_num}.png")
    dst_image_path = os.path.join(src_image_dir, f"{image_num}_rect.png")
    image = Image.open(src_image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bounds, outline='red', width=5)  # 绘制矩形
    image.save(dst_image_path)  # 保存图片


def quick_draw(src_image, dst_image, bounds):
    """快速绘制矩形框"""
    def test_quick_draw():
        src_image = "1.png"
        dst_image = "show_temp.png"
        bounds = "[776,213][929,299]"
        quick_draw(src_image, dst_image, bounds)
    matches = re.findall(r'\[(\d+),(\d+)\]', bounds)
    bounds = [[int(x), int(y)] for x, y in matches]  # 诸如 [776,213][929,299] 的形式
    image = Image.open(src_image)
    draw = ImageDraw.Draw(image)
    x1, y1 = bounds[0]  # 左上角坐标
    x2, y2 = bounds[1]  # 右下角坐标
    draw.rectangle([x1, y1, x2, y2], outline='red', width=5)  # 绘制矩形
    image.save(dst_image)  # 保存图片


def generate_utg(work_dir):
    # 生成 utg 相关文件
    # 复制html文件到本地目录下
    src_file = os.path.join('resources', 'index.html')
    dst_file = os.path.join(work_dir, 'index.html')
    shutil.copy(src_file, dst_file)
    
    # 读取 utg.json，写入成 utg.js
    utg_json_file = os.path.join(work_dir, 'utg.json')
    if not os.path.exists(utg_json_file):
        return
    with open(utg_json_file, 'r', encoding='utf-8') as f:
        utg_json = json.load(f)
    node_num = len(utg_json['nodes'])
    edge_num = len(utg_json['edges'])
    utg_json['info'] = {'nodeNum': node_num, 'edgeNum': edge_num}

    for index, edge in enumerate(utg_json['edges']):
        edge_id = edge.pop(DictKey.EDGE_ID)  # 先删除 'edge_id' 并获得其值
        image_path = edge.pop(DictKey.IMAGE_PATH, '')
        if image_path:
            image = image_path[image_path.find('utg_data'):].replace('\\', '/')
        else:
            image = ''
        edge = {'id': edge_id, 'label': str(edge_id), 'image': image, **edge}  # 重构字典，确保顺序
        utg_json['edges'][index] = edge

    for index, page in enumerate(utg_json['nodes']):
        node_id = page.pop('page_id')  # 先删除 'page_id' 并获得其值
        activity = page.get('activity', '')
        match = re.search(r'\.([^.]+)$', activity)
        if match:
            page['label'] = f"{node_id}: {match.group(1)}"  # page 的 label 是 activity
        else:
            page['label'] = f"{node_id}"  # 如果没有匹配到 activity，只显示 node_id

        image_path = page.pop('image_path', '')  # node 必有 image_path
        image = image_path[image_path.find('utg_data'):].replace('\\', '/')
        page = {'id': node_id, 'image': image, **page}
        utg_json['nodes'][index] = page

    utg_js_file = os.path.join(work_dir, 'utg.js')
    with open(utg_js_file, 'w', encoding='utf-8') as f:
        f.write(f"utg = {json.dumps(utg_json, indent=2, ensure_ascii=False)};")
        
        
def bounds_iou(bounds1, bounds2):  # 计算两个元素bounds的交并比
    if bounds1 is None and bounds2 is None:
        return 1.0
    elif bounds1 is None or bounds2 is None:
        return 0.0
    l1, t1, r1, b1 = bounds1
    l2, t2, r2, b2 = bounds2
    
    if l1 >= r2 or l2 >= r1 or t1 >= b2 or t2 >= b1:
        return 0
    
    intersection = (min(r1, r2) - max(l1, l2)) * (min(b1, b2) - max(t1, t2))
    union = (r1 - l1)*(b1 - t1) + (r2 - l2)*(b2 - t2) - intersection
    
    return intersection/union


def build_component_library(utg_data_dir, components_dir, threshold=0.98):
    # 加载预训练的 ViT 模型
    model = vit_b_16(pretrained=True)
    # 移除分类头以获取嵌入
    model.heads = torch.nn.Identity()
    model.eval()

    # 定义图像预处理转换
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not os.path.exists(components_dir):
        os.makedirs(components_dir)
    images_dir = os.path.join(components_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    with open(os.path.join(utg_data_dir, "utg.json"), 'r') as f:
        utg = json.load(f)
    edges = utg.get('edges', [])
    components = []
    for edge in edges:
        component_image = Image.open(edge["image"]).convert('RGB')
        component_image = preprocess(component_image).unsqueeze(0)
        with torch.no_grad():
            component_embedding = model(component_image).numpy().flatten()
        edge["embedding"] = component_embedding
        edge["is_add"] = False
    count = 0
    for i, edge1 in enumerate(edges):
        if edge1["is_add"]:
            continue
        edge1["is_add"] = True
        component_path = os.path.join(images_dir,f'{count}.png')
        shutil.copy(edge1["image"], component_path)
        component_embedding1 = edge1["embedding"]
        functions = [edge1["summary"]["function"]]
        for j in range(i+1, len(edges)):
            edge2 = edges[j]
            component_embedding2 = edge2["embedding"]
            if cosine_similarity([component_embedding1], [component_embedding2])[0][0] > threshold:
                edge2["is_add"] = True
                functions.append(edge2["summary"]["function"])
        
        components.append({"component_id":count, "image":component_path,"functions":functions})
        count += 1
    
    with open(os.path.join(components_dir, 'components.json'), 'w') as f:
        json.dump(components, f, indent=4, ensure_ascii=False)


# =============== 基础工具相关 =================


# 获取子字典
def get_sub_dict(d: dict, keys: list):
    sub_d = {}
    for key in keys:
        value = d.get(key, None)
        if value is not None:
            sub_d[key] = value
    return sub_d


# 获取字典列表的子字典列表
def get_sub_dict_list(dict_list: list, keys: list):
    ret = []
    for d in dict_list:
        sub_d = get_sub_dict(d, keys)
        ret.append(sub_d)
    return ret


# 删除指定 keys
def remove_keys(d: dict, keys: list):
    ret = {}
    for key in keys:
        ret[key] = d.pop(key, None)
    return ret


def get_page_by_id(page_list, id):
    if id < 0:
        return None
    for page in page_list:
        if page[DictKey.PAGE_ID] == id:
            return page
    return None


def get_pages_by_ids(page_list, ids):
    pages = []
    for id in ids:
        page = get_page_by_id(page_list, id)
        pages.append(page)
    return pages


def get_edge_by_id(edge_list, id):
    if id < 0:
        return None
    for edge in edge_list:
        if edge[DictKey.EDGE_ID] == id:
            return edge
    return None


def get_edges_by_ids(edge_list, ids):
    edges = []
    for id in ids:
       edge = get_edge_by_id(edge_list, id)
       edges.append(edge)
    return edges


# 交互节点
def get_node_by_id(node_list, id):
    if id < 0:
        return None
    for node in node_list:
        if node[DictKey.NODE_ID] == id:
            return node
    return None


def write_json_file(data, file_path, mode='w', indent=2, separators=None):
    # 写入 JSON 文件
    with open(file_path, mode, encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, separators=separators)


def read_json_file(file_path, mode='r'):
    # 读取 JSON 文件
    try:
        with open(file_path, mode, encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {}


def write_json_to_excel(json_file, json_data=None, excel_file="temp.xlsx"):
    # json_data 需要是字典列表
    if json_data is None:
        data = read_json_file(json_file)
    else:
        data = json_data
    # 将 JSON 数据转换为 DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        raise ValueError("JSON 数据格式不支持，仅支持列表或字典格式")
    
    # 写入 Excel 文件到 sheet1，保留其它 sheet
    if os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='sheet1', index=False)
    else:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='sheet1', index=False)
    print(f"数据已成功写入到 {excel_file}")


def write_page_mapping(task_dir, xlsx_file_path):
    # 读取 task_dir 下的 image_mapping.txt，写入到 excel 文件中
    mapping_file_path = os.path.join(task_dir, 'image_mapping.txt')
    if not os.path.exists(mapping_file_path):
        logger.log(f"Error: {mapping_file_path} does not exist.")
        return
    mapping_data = []
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        utg_num = 0
        for line in lines:
            parts = line.strip().split(' -> ')
            if len(parts) == 2:
                step_num = int(parts[0]) + 1
                utg_num = max(utg_num, int(parts[1]) + 1)
                mapping_data.append({'x': step_num, 'y': utg_num})
    df = pd.DataFrame(mapping_data)
    # 确保目标目录存在
    dir_name = os.path.dirname(xlsx_file_path)
    if dir_name:
        ensure_dir(dir_name)

    # 如果文件存在，则以追加模式打开并替换 sheet；如果不存在则创建新文件并写入
    if os.path.exists(xlsx_file_path):
        with pd.ExcelWriter(xlsx_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='page_mapping', index=False)
    else:
        with pd.ExcelWriter(xlsx_file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='page_mapping', index=False)

    logger.log(f"页面映射已保存到 {xlsx_file_path}")


def get_task_data(task_dir):
    # 读取 task_dir 下的 chat.json和log.txt，写入到 excel 文件中
    task_data_file_path = os.path.join(task_dir, 'chat.json')
    log_file_path = os.path.join(task_dir, 'log0.txt')
    
    # 从日志计算task总时间
    # 每一行的时间戳形式为 [2026-01-13 21:39:25]
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            start_time_str = re.search(r'\[(.*?)\]', lines[0]).group(1)
            end_time_str = re.search(r'\[(.*?)\]', lines[-1]).group(1)
            time_format = "%Y-%m-%d %H:%M:%S"
            start_time = datetime.strptime(start_time_str, time_format)
            end_time = datetime.strptime(end_time_str, time_format)
            total_time = (end_time - start_time).total_seconds()
        else:
            total_time = 0
    
    # 读取 chat.json, task_data 是一个字典列表
    task_data = read_json_file(task_data_file_path)
    step_num = len(task_data)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for step in task_data:
        step_chat_data = step[DictKey.STEP_CHAT_DATA]
        for sub_step in step_chat_data:
            usage = sub_step[DictKey.RESPONSE][DictKey.USAGE]
            prompt_tokens = usage.get(DictKey.PROMPT_TOKENS, 0)
            total_prompt_tokens += prompt_tokens
            completion_tokens = usage.get(DictKey.COMPLETION_TOKENS, 0)
            total_completion_tokens += completion_tokens
    
    return {
        DictKey.STEP_NUM: step_num,
        DictKey.TOTAL_TIME: total_time,
        DictKey.PROMPT_TOKENS: total_prompt_tokens,
        DictKey.COMPLETION_TOKENS: total_completion_tokens
    }


def get_task_data_dirs(base_dir, start_index=0, task_num=-1):
    # 获取 base_dir 下所有任务目录的路径列表
    all_data = []
    for i in range(start_index, start_index + task_num):
        task_dir = os.path.join(base_dir, str(i))
        if not os.path.exists(task_dir):
            raise FileNotFoundError(f"Task directory {task_dir} does not exist.")
            
        task_data = get_task_data(task_dir)
        data = {DictKey.CASE_ID: i, **task_data}
        all_data.append(data)
    return all_data


