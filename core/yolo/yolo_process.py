# 存放 yolo 处理的相关函数
import json
import cv2
import os
from ultralytics import YOLO


CLASSES = ['status bar', 'navigation bar', 'buttonicon', 'edittext',
            'image', 'page indicator', 'progress bar', 'rating bar',
            'check box', 'options', 'spinner', 'autocompletetextview', 'switch']


def write_bbox_to_json(bbox, filename="temp_yolo.json"):
    with open(filename, "w") as f:
        json.dump(bbox, f, indent=2)


def read_bbox_from_json(filename="temp_yolo.json"):
    list_read = []
    with open(filename, 'r') as f:
        list_read = json.load(f)
    return list_read


def is_contained(a_bounds, b_bounds):
    # 判断a是否包含b
    x1_a, y1_a, x2_a, y2_a = a_bounds
    x1_b, y1_b, x2_b, y2_b = b_bounds
    return x1_a <= x1_b and y1_a <= y1_b and x2_a >= x2_b and y2_a >= y2_b


def yolo2dict(boxes, origin_shape):
    # 将 yolo 的输出转换为 dict
    # origin_shape 形如 (2340, 1080)
    boxes_dict = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bounds = [x1, y1, x2, y2]
        bounds_rel = abs2rel(bounds, origin_shape)
        cls = int(cls)
        class_name = CLASSES[cls]
        dict_data = {
            "class": class_name,
            "bounds":  bounds,
            "bounds_rel": bounds_rel,
            "conf": conf,
            "cls": cls,
            "source": "yolo"
        }
        boxes_dict.append(dict_data)
    return boxes_dict


def abs2rel(bounds, origin_shape):
    # 将绝对坐标转换为相对坐标（相对于图像左上角）
    # origin_shape 形如 (2340, 1080)
    h, w = origin_shape
    relative_data = []
    
    x1, y1, x2, y2 = bounds
    x1_rel = round(x1 / w, 3)  # 保留三位小数
    y1_rel = round(y1 / h, 3)
    x2_rel = round(x2 / w, 3)
    y2_rel = round(y2 / h, 3)

    relative_data = [x1_rel, y1_rel, x2_rel, y2_rel]
    return relative_data


# ==================== 过滤逻辑 ====================


def filter_yolo(boxes: list[dict], drop_class_id=[0,1]):
    # yolo 基础过滤规则
    filtered_boxes = []
    for box in boxes:
        bounds_rel = box["bounds_rel"]
        cls = box["cls"]
        
        x1_rel, y1_rel, x2_rel, y2_rel = bounds_rel
        width = x2_rel - x1_rel
        height = y2_rel - y1_rel
        area = width * height

        keep_flag = True

        if cls in drop_class_id:  # 过滤不要的类框
            keep_flag = False
        if y1_rel < 0.04:  # 过滤状态栏(有些顶部状态栏里的元素被识别为 icon)
            keep_flag = False
        if area >= 0.3:  # 过滤超大框
            keep_flag = False
        if x2_rel-x1_rel < 0.01 or y2_rel-y1_rel < 0.01:  # 过滤过小框
           keep_flag = False
        if keep_flag:
            filtered_boxes.append(box)
    
    return filtered_boxes


def iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集的面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # 计算两个边界框的并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def NMS(boxes: list[dict], iou_threshold: float = 0.5):
    # 按照置信度降序排序边界框
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    picked_boxes = []
    while boxes:
        # 选取置信度最高的边界框
        current_box = boxes.pop(0)
        picked_boxes.append(current_box)
        # 计算当前边界框与其他边界框的IoU
        boxes = [box for box in boxes if iou(current_box['bounds'], box['bounds']) < iou_threshold]
    
    return picked_boxes


def draw_yolo_boxes(boxes, input_image_path, output_image_path="temp_yolo.png"):
    # 简单绘制框到图片上
    image = cv2.imread(input_image_path)
    height, width, _ = image.shape
    for index, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box
        # 将相对坐标转换为绝对坐标
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        # 绘制框
        color = (0, 255, 0)  # 绿色
        thickness = 2
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 在框左上角绘制文本，显示置信度和类别
        label = f"id:{index},conf:{conf:.2f},cls:{int(cls)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_x = x1
        text_y = y1 - 10
        image = cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
    
    cv2.imwrite(output_image_path, image)


def full_contained_filter_by_conf(boxes):
    # 当 a 包含且仅包含 b 时，取置信度高的框，否则不动
    def filter(boxes):  # 单次调用 filter 只过滤一轮
        remove_id_list = []
        for i, a in enumerate(boxes):
            contained_list = []
            for j, b in enumerate(boxes):
                a_bounds = a["bounds"]
                b_bounds = b["bounds"]
                if i != j and is_contained(a_bounds, b_bounds):
                    contained_list.append(j)
            # 针对仅包含一个框的情况，置信度低的加入删除列表
            if contained_list and len(contained_list) == 1:
                j = contained_list[0]
                b = boxes[j]
                a_conf = a["conf"]
                b_conf = b["conf"]
                if a_conf < b_conf:
                    remove_id_list.append(i)
                else:
                    remove_id_list.append(j)

        filtered_boxes = [box for idx, box in enumerate(boxes) if idx not in remove_id_list]
        return filtered_boxes
    
    filtered_boxes = boxes
    # 不断过滤，直到 filtered_boxes 不再减少
    previous_length = len(filtered_boxes)
    while True:
        filtered_boxes = filter(filtered_boxes)
        current_length = len(filtered_boxes)
        if current_length == previous_length:
            break
        previous_length = current_length

    return filtered_boxes


def partly_contained_filter(boxes):
    # 当 b 的绝大部分被 a 包含时, 剔除 b
    def filter(boxes):
        remove_id_list = []
        for i, a in enumerate(boxes):
            for j, b in enumerate(boxes):
                if i != j:
                    a_bounds = a["bounds"]
                    b_bounds = b["bounds"]
                    x1_inter = max(a_bounds[0], b_bounds[0])
                    y1_inter = max(a_bounds[1], b_bounds[1])
                    x2_inter = min(a_bounds[2], b_bounds[2])
                    y2_inter = min(a_bounds[3], b_bounds[3])
                    
                    # 计算交集的面积
                    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
                    b_area = (b_bounds[2] - b_bounds[0]) * (b_bounds[3] - b_bounds[1])
                    if inter_area / b_area > 0.8:  # 如果 b 的 0.8 以上部分被 a 包含
                        if i not in remove_id_list:  # a,b 重叠度较高时，保留一个，防止都被删除
                            remove_id_list.append(j)
        
        filtered_boxes = [box for idx, box in enumerate(boxes) if idx not in remove_id_list]
        return filtered_boxes
    
    filtered_boxes = boxes
    previous_length = len(filtered_boxes)
    while True:
        filtered_boxes = filter(filtered_boxes)
        current_length = len(filtered_boxes)
        if current_length == previous_length:
            break
        previous_length = current_length
    return filtered_boxes


def filter_contain_boxes(boxes: list[dict]):
    # 过滤存在一定程度包含关系的框（谨慎使用）
    filtered_boxes = full_contained_filter_by_conf(boxes)
    # filtered_boxes = partly_contained_filter(filtered_boxes)
    return filtered_boxes


def yolo_detection(image_path, model, conf_infer=0.02, drop_class_id=[0, 1]):
    """
    yolo 推理总方法，
    return: 格式化的 boxes 列表
    e.g. {  
        "class": class_name_en, 
        "bounds": xxx, 
        "bounds_rel": 相对坐标,
        "conf": 置信度,
        "cls:" yolo 推理的原始类序号(0~12)
        "source": "yolo" (用于多源合并区分来源)
    }
    
    """
    results = model([image_path], conf=conf_infer, verbose=False)
    origin_shape = results[0].orig_shape  # (h, w), 例如 (2340, 1080),因为 cv 中看的是行×列
    names = results[0].names  # name 是字典: {0: 'statusbar', 1: 'navigationbar'...
    cls = results[0].boxes.cls  # 推理得到类的索引: tensor([ 0.,  1., ...
    conf = results[0].boxes.conf  # 推理置信度: tensor([0.8401, 0.7482, ...
    data = results[0].boxes.data  # data 是二维向量，每一行有 6 个元素，分别是(x1,y1,x2,y2,conf,cls) 坐标为绝对坐标
    # print("data.shape", data.shape)
    boxes = data.tolist()  # 转为列表形式
    boxes = yolo2dict(boxes=boxes, origin_shape=origin_shape)
    # write_bbox_to_json(bbox=boxes)

    # ========= 之后的 boxes 均为字典形式处理 ===========

    boxes = filter_yolo(boxes=boxes, drop_class_id=drop_class_id)
    boxes = NMS(boxes)
    boxes = filter_contain_boxes(boxes=boxes)  # 过滤包含框关系
    return boxes


def get_yolo_rects(image_path):
    model_path_yolo = os.path.join("core", "yolo", "yolo_mdl.pt")
    model_det = YOLO(model_path_yolo, task='detect')  # Yolov8n
    rects = yolo_detection(image_path=image_path, model=model_det)
    return rects


def test_pipline(image_path="temp_zy1.png"):
    from utils.draw import draw_rects
    from core.yolo.yolo_process import yolo_detection
    image_path = "temp/temp_zy.png"
    model_path_yolo = "yolo/yolo_mdl.pt"
    model_det = YOLO(model_path_yolo, task='detect')  # Yolov8n
    boxes = yolo_detection(image_path=image_path, model=model_det)
    draw_rects(image_path=image_path, output_dir='temp', rects=boxes)

