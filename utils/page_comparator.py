import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
from PIL import Image
from scient.algorithm import similar
from scient.image import hash
from core.cog.page_cognition import get_cog_rects
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import transforms
from torchvision.models import vit_b_16

def get_interactive_nodes_bounds(xml_file:str):
    # 布局集合
    view_groups = ["android.support.v7.widget.LinearLayoutCompat", "android.widget.HorizontalScrollView",
                   "android.widget.GridView", "androidx.drawerlayout.widget.DrawerLayout",
                   "android.widget.RelativeLayout", "androidx.recyclerview.widget.RecyclerView",
                   "com.google.android.material.card.MaterialCardView", "android.view.ViewGroup",
                   "android.widget.FrameLayout", "android.widget.LinearLayout",
                   "android.support.v7.widget.RecyclerView"]

    # 判断有无子节点可点击
    def no_children_clickable(node: ET.Element):
        for child in node:
            if child.attrib["clickable"] == "true":
                return False
            if not no_children_clickable(child):
                return False
        return True

    def traverse(node: ET.Element):
        if 'bounds' in node.attrib:
            is_container = node.attrib['class'] in view_groups
            if node.attrib['clickable'] == 'true' and (not is_container or (is_container and no_children_clickable(node))):
                bounds = node.attrib["bounds"][1:-1].split("][")
                x1, y1 = map(int, bounds[0].split(","))
                x2, y2 = map(int, bounds[1].split(","))
                bounds = [[x1, y1], [x2, y2]]
                if bounds not in bounds_list:
                    bounds_list.append(bounds)
        for child in node:
            traverse(child)

    root = ET.parse(xml_file).getroot()
    bounds_list = []
    traverse(root)
    return bounds_list


def calculate_pixies_similarity_by_ahash(img_path1:str, img_path2:str):
    # 根据页面可交互元素的位置将页面转化为一个二维数组，二维数组代表一个灰度图，采用ahash比较灰度图的相似度
    with Image.open(img_path1) as img:
        width, height = img.size
    
    # 默认img命名和xml命名相同
    xml_path1 = os.path.splitext(img_path1)[0]+'.xml'
    xml_path2 = os.path.splitext(img_path2)[0]+'.xml'
    # 灰度图路径
    grey_img_path1 = os.path.splitext(img_path1)[0]+'_grey'+'.png'
    grey_img_path2 = os.path.splitext(img_path2)[0]+'_grey'+'.png'
    bounds_list1 = get_interactive_nodes_bounds(xml_path1)
    bounds_list2 = get_interactive_nodes_bounds(xml_path2)

    grey_img1 = rects2grey_img(bounds_list1, width, height)
    grey_img2 = rects2grey_img(bounds_list2, width, height)
    #保存生成的灰度图
    cv2.imwrite(grey_img_path1, grey_img1)
    cv2.imwrite(grey_img_path2, grey_img2)
    score = similar.hamming(hash.mean(grey_img1), hash.mean(grey_img2))

    return score


def rects2grey_img(bounds_list:list, device_width:int, device_height:int)->np.array:
    # 二维数组初始值为255，对应一个空白灰度图，遍历所有元素的bounds，对于每个bounds覆盖的区域，二维数组的对应位置就-50(黑色加深)，并且边框位置额外-50，强调边框的区分作用。

    val_list = np.array([[255]*device_width for i in range(device_height)])
    for bounds in bounds_list:
        val_list[bounds[0][1]:bounds[1][1],bounds[0][0]:bounds[1][0]] -= 50
        val_list[bounds[0][1]:bounds[1][1],bounds[0][0]] -= 50
        val_list[bounds[0][1]:bounds[1][1],bounds[1][0]-1] -= 50
        val_list[bounds[0][1],bounds[0][0]:bounds[1][0]] -= 50
        val_list[bounds[1][1]-1,bounds[0][0]:bounds[1][0]] -= 50
    val_list[val_list<0] = 0
    return val_list

# 画一张原图的布局图，纯文本组件为红色块，不含文本的图像为蓝色块， 含文本的图像为绿色块
def draw_layout_image(rects:list, device_width, device_height, output_path):
    if os.path.exists(output_path):
        return
    RED = np.array([255, 0, 0], dtype=np.uint32)
    GREEN = np.array([0, 255, 0], dtype=np.uint32)
    BLUE = np.array([0, 0, 255], dtype=np.uint32)
    image_array = np.zeros((device_height, device_width, 3), dtype=np.uint32)
    for rect in rects:
        _class = rect.get('class', None)
        if not _class:
            continue
        bounds = rect['bounds']
        text = rect.get('text', None)
        content_desc = rect.get('content_desc', None)
        if 'Text' in _class:  # 纯文本组件
            image_array[bounds[1]:bounds[3], bounds[0]:bounds[2]] += RED
            image_array[bounds[1]:bounds[3], bounds[0]:bounds[2]] //= 2
        
        elif (text and text != '') or (content_desc and content_desc != ''): #  含文本图像
            image_array[bounds[1]:bounds[3], bounds[0]:bounds[2]] += GREEN
            image_array[bounds[1]:bounds[3], bounds[0]:bounds[2]] //= 2
        else:
            image_array[bounds[1]:bounds[3], bounds[0]:bounds[2]] += BLUE
            image_array[bounds[1]:bounds[3], bounds[0]:bounds[2]] //= 2   
    image_array = np.array(image_array, dtype=np.uint8)
    image = Image.fromarray(image_array)
    image.save(output_path)


def cal_layout_similarity(image_path1, image_path2, rects1, rects2):
    # 加载预训练的 ViT 模型
    model = vit_b_16(pretrained=True)
    # 移除分类头以获取嵌入
    model.heads = torch.nn.Identity()
    model.eval()

    # 定义图像预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with Image.open(image_path1) as img:
        width, height = img.size
    layout_path1 = os.path.splitext(image_path1)[0] + '_layout.png'
    layout_path2 = os.path.splitext(image_path2)[0] + '_layout.png'
    draw_layout_image(rects1, width, height, layout_path1)
    draw_layout_image(rects2, width, height, layout_path2)
    layout1 = Image.open(layout_path1)
    layout2 = Image.open(layout_path2)
    
    # layout1 = np.array(layout1)
    # layout2 = np.array(layout2)
    # layout1 = np.mean(layout1, axis=2).astype(np.uint8)
    # layout2 = np.mean(layout2, axis=2).astype(np.uint8)
    input1 = preprocess(layout1).unsqueeze(0)
    input2 = preprocess(layout2).unsqueeze(0)
    with torch.no_grad():
        embedding1 = model(input1).numpy().flatten()
        embedding2 = model(input2).numpy().flatten()
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]

    # similarity = similar.hamming(hash.mean(layout1), hash.mean(layout2))
    return similarity 

# 计算两个页面组件的交并比
def cal_page_iou(rects1, rects2, device_width, device_height):
    val_list = np.array([[0]*device_width for i in range(device_height)])
    for rect in rects1:
        if rect["class"] == "OCRText":
            continue
        bounds = rect["bounds"]
        val_list[bounds[1]:bounds[3],bounds[0]:bounds[2]] = 1
    
    for rect in rects2:
        if rect["class"] == "OCRText":
            continue
        bounds = rect["bounds"]
        sub_array = val_list[bounds[1]:bounds[3],bounds[0]:bounds[2]]
        for i in range(sub_array.shape[0]):
            for j in range(sub_array.shape[1]):
                if sub_array[i,j] in [0, 1]:
                    sub_array[i, j] += 1
        val_list[bounds[1]:bounds[3],bounds[0]:bounds[2]] = sub_array
    
    intersection = 0
    union = 0
    for i in range(val_list.shape[0]):
        for j in range(val_list.shape[1]):
            if val_list[i, j] in [1, 2]:
                union += 1
            if val_list[i, j] == 2:
                intersection += 1
                
    return intersection/union

    

if __name__ == '__main__':
    pass