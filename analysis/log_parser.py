import re
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# 文件路径
LOG_FILE = "./wandb/latest-run/files/output.log"
OUTPUT_FILE = "parsed_logs.json"

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """从模型响应中提取最终答案。
    
    Args:
        solution_str: 模型的原始响应字符串。
        
    Returns:
        包含 (提取的答案, 处理后的字符串) 的元组。
    """
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None, solution_str

    # 提取 <answer> 部分
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def validate_response_structure(processed_str: str) -> bool:
    """验证模型响应的结构是否正确。"""
    validation_passed = True
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }
    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        if count != expected_count:
            validation_passed = False
    if positions['think_start'] > positions['think_end'] or \
       positions['think_end'] > positions['answer_start'] or \
       positions['answer_start'] > positions['answer_end']:
        validation_passed = False
    return validation_passed

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """解析 Ground Truth 部分，返回角色及其状态的字典。"""
    status_dict = {}
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
    return status_dict

def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """解析模型答案，返回角色及其状态的字典。"""
    status_dict = {}
    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
        else:
            return None
    return status_dict

def parse_epoch_and_step(log_content: str) -> Tuple[Optional[int], Optional[int]]:
    """解析日志内容中的 epoch 和 step 信息。
    
    Args:
        log_content: 日志内容。
    
    Returns:
        包含 (epoch, step) 的元组。
    """
    epoch_pattern = re.compile(r"epoch\s+(\d+)", re.IGNORECASE)
    step_pattern = re.compile(r"step\s+(\d+)", re.IGNORECASE)
    
    epoch_match = epoch_pattern.search(log_content)
    step_match = step_pattern.search(log_content)
    
    epoch = int(epoch_match.group(1)) if epoch_match else None
    step = int(step_match.group(1)) if step_match else None
    
    return epoch, step

def parse_log(log_content: str):
    """解析日志内容并提取结构化数据。"""
    sample_pattern = re.compile(r"=+\n=+ Processing New Sample =+\n")
    ground_truth_pattern = re.compile(r"\[Ground Truth Parsing\](.*?)\[Ground Truth\] Final identities: (.*?)\n", re.S)
    model_response_pattern = re.compile(r"\[Model Response\]\n\n<think>(.*?)</think>\n<answer>(.*?)</answer>", re.S)
    final_score_pattern = re.compile(r"Final Score.*?Format: (.*?)\n.*?Answer: (.*?)\n.*?Total: (.*?)\n", re.S)

    samples = sample_pattern.split(log_content)
    parsed_data = []

    # Track the last seen epoch and step
    current_epoch = None
    current_step = None

    for sample in samples[1:]:
        parsed_sample = {}

        # Update epoch and step if new values are found
        epoch, step = parse_epoch_and_step(sample)
        if epoch is not None:
            current_epoch = epoch
        if step is not None:
            current_step = step

        # 如果 current_epoch 或 current_step 为 None，初始化为 0
        if current_epoch is None:
            current_epoch = 0
        if current_step is None:
            current_step = 0

        # Assign the current epoch and step to the sample
        parsed_sample["epoch"] = current_epoch
        parsed_sample["step"] = current_step

        # 提取 Ground Truth
        ground_truth_match = ground_truth_pattern.search(sample)
        if ground_truth_match:
            raw_gt = ground_truth_match.group(1).strip()
            parsed_sample["ground_truth"] = parse_solution_text_format(raw_gt)

        # 提取模型响应
        model_response_match = model_response_pattern.search(sample)
        if model_response_match:
            parsed_sample["model_think"] = model_response_match.group(1).strip()
            parsed_sample["model_answer_raw"] = model_response_match.group(2).strip()

            # 验证响应结构
            parsed_sample["structure_valid"] = validate_response_structure(model_response_match.group(1))

        # 提取最终分数
        final_score_match = final_score_pattern.search(sample)
        if final_score_match:
            parsed_sample["final_score"] = {
                "format": float(final_score_match.group(1).strip()),
                "answer": float(final_score_match.group(2).strip()),
                "total": float(final_score_match.group(3).strip())
            }

        parsed_data.append(parsed_sample)

    return parsed_data

def save_to_json(data, output_file):
    """保存解析结果为 JSON 文件。"""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

def main():
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        print(f"日志文件 {LOG_FILE} 不存在！")
        return

    with open(log_path, "r") as f:
        log_content = f.read()

    parsed_data = parse_log(log_content)
    save_to_json(parsed_data, OUTPUT_FILE)
    print(f"解析完成！结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()