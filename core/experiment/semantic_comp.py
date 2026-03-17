# 页面语义建模对比
import os
from utils.utils import read_json_file, write_json_file
from utils.macro import DictKey
from utils.chat import chat_with_llm, parse_response
from utils.logger import logger, task_dir


def get_summary(image_path):
    prompt = """
        You are a mobile app testing engineer, you are presented an mobile app on a smartphone.
        Your task is to summarize the main functions of the current page in a short sentence.
    """
    response = chat_with_llm(prompt, image_urls=[image_path])
    return response
    

def get_summaries(image_dir, image_num=30):
    summaries = []
    for i in range(image_num):
        image_path = os.path.join(image_dir, f"{i}.png")
        summary = get_summary(image_path)
        logger.log(f"Processed image {i+1}/{image_num}: {image_path}")
        logger.log(f"Summary: {summary}")
        summaries.append({
            DictKey.IMAGE_PATH: image_path,
            DictKey.SUMMARY: summary["choices"][0]["text"]
        })
    return summaries


# ---


def judge_hallucination(image_path, description):
    prompt = f"""
        You are a mobile app testing engineer, you are presented an mobile app on a smartphone.
        Your task is to check whether the description of the current page of the app is accurate based on the image.
        <Description>: {description}

        if there is any hallucination, respond 0, else respond 1.
        Do not output other information.
    """
    response = chat_with_llm(prompt, image_urls=[image_path])
    return response


def judge_accuracy(image_path, description):
    prompt = f"""
        You are a mobile app testing engineer, you are presented an mobile app on a smartphone.
        Your task is to judge whether the description of the page can describe the main functions and features of the page based on the image.
        <Description>: {description}
        The score is from 0.0 to 10.0, where 0.0 means completely inaccurate and 10.0 means completely accurate. You can use one decimal place for the score.
        Respond with only the score, do not output other information.
    """
    response = chat_with_llm(prompt, image_urls=[image_path])
    return response


def judge_completeness(image_path, description):
    prompt = f"""
        You are a mobile app testing engineer, you are presented an mobile app on a smartphone.
        Your task is to judge whether the description of the page can fully cover the information and function in the image.
        <Description>: {description}
        The score is from 0.0 to 10.0, where 0.0 means completely incomplete and 10.0 means completely complete. You can use one decimal place for the score.
        Respond with only the score, do not output other information.
    """
    response = chat_with_llm(prompt, image_urls=[image_path])
    return response


# 总结
def summary_target_images(base_dir, image_num=30):
    image_dir = os.path.join(base_dir, "full_data")
    summaries = get_summaries(image_dir, image_num=image_num)
    summaries_file_path = os.path.join(task_dir, "summaries.json")
    write_json_file(summaries, summaries_file_path)
    print("summaries saved to ", summaries_file_path)


# 分析
def analyze_summaries(work_dir, task_dir, image_num=30):
    summary_file_path = os.path.join(work_dir, "summaries.json")
    summaries = read_json_file(summary_file_path)
    results = []
    for i in range(image_num):
        logger.log(f"Analyzing summary {i+1}/{image_num}")
        image_path = summaries[i][DictKey.IMAGE_PATH]
        summary = summaries[i][DictKey.SUMMARY]
        hallucination = judge_hallucination(image_path, summary)
        logger.log(f"Hallucination: {hallucination}")
        accuracy = judge_accuracy(image_path, summary)
        logger.log(f"Accuracy: {accuracy}")
        completeness = judge_completeness(image_path, summary)
        logger.log(f"Completeness: {completeness}")

        data = {
            DictKey.IMAGE_PATH: image_path,
            DictKey.SUMMARY: summary,
            "hallucination": hallucination["choices"][0]["text"],
            "accuracy": accuracy["choices"][0]["text"],
            "completeness": completeness["choices"][0]["text"]
        }
        results.append(data)

    results_file_path = os.path.join(task_dir, "analysis_results.json")
    write_json_file(results, results_file_path)


def cal_results(work_dir):
    # 计算结果的平均分等
    data = read_json_file(os.path.join(work_dir, "analysis_results.json"))
    total_hallucination = 0
    total_accuracy = 0
    total_completeness = 0
    total_summary_len = 0
    count = len(data)
    for item in data:
        total_hallucination += float(item["hallucination"]) if item["hallucination"].isdigit() else 0
        total_accuracy += float(item["accuracy"]) if item["accuracy"].replace('.', '').isdigit() else 0
        total_completeness += float(item["completeness"]) if item["completeness"].replace('.', '').isdigit() else 0
        total_summary_len += len(item[DictKey.SUMMARY])

    avg_hallucination = total_hallucination / count if count > 0 else 0
    avg_accuracy = total_accuracy / count if count > 0 else 0
    avg_completeness = total_completeness / count if count > 0 else 0

    results_summary = {
        "average_hallucination": avg_hallucination,
        "average_accuracy": avg_accuracy,
        "average_completeness": avg_completeness,
        "average_summary_length": total_summary_len / count if count > 0 else 0,
        "count": count
    }

    results_summary_file_path = os.path.join(work_dir, "results_summary.json")
    write_json_file(results_summary, results_summary_file_path)
