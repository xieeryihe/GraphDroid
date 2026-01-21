import queue
import torch
import os
import json
import shutil
import copy
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16
from utils.macro import *
# from chat import get_answer_from_llm
from utils.gpt_chat import get_answer_from_llm

def build_adjacent_list(edges):
    # 构建图的邻接表
    graph = {}
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        if from_node not in graph:
            graph[from_node] = []
        graph[from_node].append({"to":to_node, "edge":edge, })
    
    return graph

def observe_widgets(image_path):
    image_name = os.path.splitext(image_path)[0]
    SoM_image_path = image_name +'_SoM.png'
    widgets_path = image_name + '.json'
    with open(widgets_path, 'rb') as f:
        widgets = json.load(f)
        for widget in widgets:
            widget.pop('isVisited')
            widget.pop('shouldVisit')
            widget.pop('id_bounds')
    observe_prompt = PROMPT.OBSERVOR.format(widgets=widgets)
    functions = get_answer_from_llm(observe_prompt, [SoM_image_path])
    return widgets, functions
    

def save_widget_image(widget_image_path, widget_id, output_dir, embeddings, model, preprocess, threshold):
    image = Image.open(widget_image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        widget_embedding = model(image).numpy().flatten()
    for key, embedding in embeddings.items():
        if cosine_similarity([widget_embedding], [embedding])[0][0]>threshold:
            return key, None
    embeddings[widget_id] = widget_embedding
    output_path = os.path.join(output_dir, f'{widget_id}.png')
    shutil.copy(widget_image_path, output_path)
    return widget_id, output_path


def crop_widget_image(page_image_path, widget, widget_id, output_dir, embeddings, model, preprocess, threshold):
    image = Image.open(page_image_path).crop(widget["bounds"]).convert('RGB')
    processed_image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        widget_embedding = model(processed_image).numpy().flatten()
    for key, embedding in embeddings.items():
        if cosine_similarity([widget_embedding], [embedding])[0][0]>threshold:
            return key, None
    
    embeddings[widget_id] = widget_embedding
    output_path = os.path.join(output_dir, f'{widget_id}.png')
    image.save(output_path)
    return widget_id, output_path


def build_component_library(graph, utg, components_dir, threshold=0.98):
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
    images_dir = os.path.join(components_dir, 'widgets')
    os.makedirs(images_dir, exist_ok=True)
    
    pages = utg.get('nodes')
    q = q.q()
    embeddings = []
    widgets_table = {}
    visited_pages = {}
    widget_id = 0
    q.put({'page_id':0, 'edge_path':[], 'page_path':[0]})
    while not q.empty():
        size = len(q)
        for i in range(size):
            current_node = q.get()
            current_page_id = current_node['page_id']
            if visited_pages.get(current_page_id):
                continue
            visited_pages[current_page_id] = True
            page_detail = pages[current_page_id]
            to_pages = graph[current_page_id]
            for to_page in to_pages:
                edge = to_page["edge"]
                saved_id, widget_path = save_widget_image(edge["image"], widget_id, images_dir, embeddings, model, preprocess, threshold)
                if widget_path:
                    widgets_table[widget_id] = {"image":widget_path, "edge_path":copy.deepcopy(current_node['edge_path']), "page_path":copy.deepcopy(current_node['page_path']), "function":edge["summary"]["function"]}
                    widget_id += 1
                q.put({"page_id":to_page['to'], "edge_path":copy.deepcopy(current_node['edge_path']).append(saved_id), "page_path":copy.deepcopy(current_node['page_path']).append(to_page['to'])})
            widgets, functions = observe_widgets(page_detail["image"])
            for widget in widgets:
                _, widget_path = crop_widget_image(page_detail["image"], widget, images_dir, embeddings, model, preprocess, threshold)
                if widget_path:
                    widgets_table[widget_id] = {"image":widget_path, "edge_path":copy.deepcopy(current_node['edge_path']), "page_path":copy.deepcopy(current_node['page_path']), "function":functions.get(str(widget["node_id"]))}
                    widget_id += 1
    
    widgets_table_path = os.path.join(components_dir, 'widgets_table.json')
    with open(widgets_table_path, 'w') as f:
        json.dump(widgets_table, f, indent=4, ensure_ascii=False)

            
            
    # for edge in edges:
    #     component_image = Image.open(edge["image"]).convert('RGB')
    #     component_image = preprocess(component_image).unsqueeze(0)
    #     with torch.no_grad():
    #         component_embedding = model(component_image).numpy().flatten()
    #     edge["embedding"] = component_embedding
    #     edge["is_add"] = False
    # count = 0
    # for i, edge1 in enumerate(edges):
    #     if edge1["is_add"]:
    #         continue
    #     edge1["is_add"] = True
    #     component_path = os.path.join(images_dir,f'{count}.png')
    #     shutil.copy(edge1["image"], component_path)
    #     edge1["image"] = component_path
    #     component_embedding1 = edge1["embedding"]
    #     functions = [edge1["summary"]["function"]]
    #     for j in range(i+1, len(edges)):
    #         if edge2["is_add"]:
    #             continue
    #         edge2 = edges[j]
    #         component_embedding2 = edge2["embedding"]
    #         if cosine_similarity([component_embedding1], [component_embedding2])[0][0] > threshold:
    #             edge2["is_add"] = True
    #             edge2["image"] = component_path
    #             functions.append(edge2["summary"]["function"])
        
    #     components.append({"component_id":count, "image":component_path,"functions":functions})
    #     count += 1
    




def build_components_table(work_dir, components_dir, threshold=0.98):
    utg_json_file = os.path.join(work_dir, 'utg.json')
    with open(utg_json_file, 'rb') as f:
        utg = json.load(f)
    
    graph = build_adjacent_list(utg["edges"])
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
    images_dir = os.path.join(components_dir, 'widgets')
    os.makedirs(images_dir, exist_ok=True)
    
    pages = utg.get('nodes')
    q = queue.Queue()
    embeddings = {}
    widgets_table = {}
    visited_pages = {}
    widget_id = 0
    q.put({'page_id':0, 'edge_path':[], 'page_path':[0]})
    while not q.empty():
        size = q.qsize()
        for i in range(size):
            current_node = q.get()
            current_page_id = current_node['page_id']
            if visited_pages.get(current_page_id):
                continue
            visited_pages[current_page_id] = True
            page_detail = pages[current_page_id]
            to_pages = graph.get(current_page_id, [])
            for to_page in to_pages:
                edge = to_page["edge"]
                saved_id, widget_path = save_widget_image(edge["image"], widget_id, images_dir, embeddings, model, preprocess, threshold)
                if widget_path:
                    widgets_table[widget_id] = {"image":widget_path, "edge_path":copy.deepcopy(current_node['edge_path']), "page_path":copy.deepcopy(current_node['page_path']), "function":edge["summary"]["function"]}
                    widget_id += 1
                new_edge_path = copy.deepcopy(current_node['edge_path'])
                new_page_path = copy.deepcopy(current_node['page_path'])
                new_edge_path.append(saved_id)
                new_page_path.append(to_page['to'])
                q.put({"page_id":to_page['to'], "edge_path":new_edge_path, "page_path":new_page_path})
            widgets, functions = observe_widgets(page_detail["image"])
            for widget in widgets:
                _, widget_path = crop_widget_image(page_detail["image"], widget, widget_id, images_dir, embeddings, model, preprocess, threshold)
                if widget_path:
                    widgets_table[widget_id] = {"image":widget_path, "edge_path":copy.deepcopy(current_node['edge_path']), "page_path":copy.deepcopy(current_node['page_path']), "function":functions.get(str(widget["node_id"]))}
                    widget_id += 1
        widgets_table_path = os.path.join(components_dir, 'widgets_table.json')
        with open(widgets_table_path, 'w') as f:
            json.dump(widgets_table, f, indent=4, ensure_ascii=False)
    