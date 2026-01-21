# 参考 UIAnalyzer/UITestAgent 的绘制框代码

import os
import glob
import io
import shutil
import platform
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple
from utils.macro import DictKey


blank = 5
text_widths, text_height = [25, 50, 75], 35
text_width_unit = 25  # 每个数字占用的宽度

RED = '\033[91m'
BLUE = '\033[94m'
GREEN = '\033[32m'
END = '\033[0m'


def get_bounds_average_color(image_path: str, bounds: List) -> Tuple[int, int, int]:
    """获得边框的平均颜色"""
    image = Image.open(image_path)
    w, h = image.size[0], image.size[1]
    pixels = image.load()
    r, g, b = 0, 0, 0
    count = 0

    for x in range(bounds[0], min(bounds[2], w)):
        for y in [bounds[1], min(bounds[3]-1, h-1)]:  # 只取边框的上下边界像素
            r += pixels[x, y][0]
            g += pixels[x, y][1]
            b += pixels[x, y][2]
            count += 1

    for y in range(bounds[1], min(bounds[3], h)):
        for x in [bounds[0], min(bounds[2] - 1, w-1)]:  # 只取边框的左右边界像素
            r += pixels[x, y][0]
            g += pixels[x, y][1]
            b += pixels[x, y][2]
            count += 1

    return r // count, g // count, b // count


def get_line_average_color(image_path: str, start_point: List, end_point: List) -> Tuple[int, int, int]:
    """获得线段的平均颜色"""
    image = Image.open(image_path)
    pixels = image.load()
    r, g, b = 0, 0, 0
    count = 0

    x1, y1 = start_point
    x2, y2 = end_point

    if x1 == x2:
        for y in range(min(y1, y2), max(y1, y2)):
            r += pixels[x1, y][0]
            g += pixels[x1, y][1]
            b += pixels[x1, y][2]
            count += 1
    elif y1 == y2:
        for x in range(min(x1, x2), max(x1, x2)):
            r += pixels[x, y1][0]
            g += pixels[x, y1][1]
            b += pixels[x, y1][2]
            count += 1
    else:  # 斜线
        for x in range(min(x1, x2), max(x1, x2)):
            y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
            r += pixels[x, y][0]
            g += pixels[x, y][1]
            b += pixels[x, y][2]
            count += 1

    return r // count, g // count, b // count


def get_inverse_color(avg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """获得反色"""
    inverse_color = (255 - avg_color[0], 255 - avg_color[1], 255 - avg_color[2])
    return inverse_color


def get_bounds_color(image_path: str, bounds: List) -> Tuple[int, int, int]:
    try:
        bounds_average_color = get_bounds_average_color(image_path, bounds)
        bounds_color = get_inverse_color(bounds_average_color)
    except ZeroDivisionError:
        bounds_color = (0, 0, 255)
    return bounds_color


def get_line_color(image_path: str, start_point: List, end_point: List) -> Tuple[int, int, int]:
    try:
        line_average_color = get_line_average_color(image_path, start_point, end_point)
        line_color = get_inverse_color(line_average_color)
    except ZeroDivisionError:
        line_color = (0, 0, 255)
    return line_color


def draw_rects(src_image_path, dst_image_path, rects):
    """
    绘制矩形和ID, 并写入 rects，rects 是字典列表，元素 item 形式:
    {'class': 'LinearLayout', 'resource_id': xxx, 'text': ['上海'], 'bounds': [26, 132, 193, 189]}
    resource_id, text, 由于元素聚合原因，可能是一个列表，不过这个函数也只需要一个 bounds 属性就够了。
    draw 本身不修改rects
    """

    if rects is None or len(rects) == 0:
        print("Invalid rects in function 'draw_rects'.")
        return -1
    
    img = Image.open(src_image_path)
    draw = ImageDraw.Draw(img)
    current_os = platform.system()
    if current_os != "Windows":
        font = ImageFont.truetype("Arial", size=35)
    else:
        font = ImageFont.truetype("arial.ttf", size=35)

    for rect in rects:
        if rect['bounds'][2] >= rect['bounds'][0] and rect['bounds'][3] >= rect['bounds'][1]:
            bound_color = get_bounds_color(src_image_path, rect['bounds'])
            draw.rectangle(rect['bounds'], outline=bound_color, width=3)
            som_tag = rect.get(DictKey.SOM_TAG, "") 
            text_width = text_width_unit * len(str(som_tag))
            width_start = rect['bounds'][0]
            height_start = rect['bounds'][1]

            if (rect['bounds'][2] - rect['bounds'][0] >= text_width and 
                rect['bounds'][3] - rect['bounds'][1] >= text_height):
                id_bounds = [width_start, height_start, 
                             width_start + text_width, height_start + text_height]
            else:
                width_start = max(rect['bounds'][0], 0)
                height_start = max(rect['bounds'][1] - text_height, 0)
                id_bounds = [width_start, height_start, 
                             width_start + text_width, height_start + text_height]

            draw.rectangle(id_bounds, fill=(0, 0, 255), width=3)
            draw.text((id_bounds[0], id_bounds[1]), str(som_tag), fill=(0, 255, 0), font=font)

    img.save(dst_image_path)

    return 0


# 剪切交互元素的图片，并且resize到224*224
def crop_image(image_path, output_dir, bounds):
    image = Image.open(image_path)
    action_image = image.crop((bounds[0], bounds[1], bounds[2], bounds[3]))
    # action_image = action_image.resize((224, 224))
    
    i = len(glob.glob(os.path.join(output_dir, '*_action.png')))
    output_path = os.path.join(output_dir, f'{i}_action.png')
    action_image.save(output_path)
    
    return output_path
    

# =============== 一些便捷绘制 =================


# 绘制矩形框
def draw_rect(src_image_path, dst_image_path, bounds,
              width=5, outline=None):
    image = Image.open(src_image_path)
    draw = ImageDraw.Draw(image)
    outline = outline if outline else get_bounds_color(src_image_path, bounds)
    draw.rectangle(bounds, outline=outline, width=width)
    image.save(dst_image_path)


# 绘制 X 形状标记
def draw_X(src_image_path, dst_image_path, point,
           size=30, width=10, fill=None):
    # size: X形状的边长, width: 线宽
    image = Image.open(src_image_path)
    draw = ImageDraw.Draw(image)
    x, y = point
    points = [
        (x - size, y - size),  # 左上
        (x + size, y + size),  # 右下
        (x - size, y + size),  # 左下
        (x + size, y - size)   # 右上
    ]
    bound = [points[0][0], points[0][1], points[1][0], points[1][1]]
    fill = fill if fill else get_bounds_color(src_image_path, bound)
    draw.line([points[0], points[1]], fill=fill, width=width)
    draw.line([points[2], points[3]], fill=fill, width=width)
    image.save(dst_image_path)


# 绘制箭头状标记，
def draw_arrow(src_image_path, dst_image_path, start_point, end_point,
               size=30, width=10, fill=None):
    # size: 箭头的大小, width: 线宽
    image = Image.open(src_image_path)
    draw = ImageDraw.Draw(image)
    fill = fill if fill else get_line_color(src_image_path, start_point, end_point)
    # 直线部分
    draw.line([start_point, end_point], fill=fill, width=width)
    image.save(dst_image_path)
    # 箭头部分用 X 代替
    draw_X(dst_image_path, dst_image_path, end_point, size=size, width=width, fill=fill)
