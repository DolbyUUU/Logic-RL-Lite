import argparse
import re
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Default output file
OUTPUT_FILE = "parsed_logs.json"

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extract the final answer from the model's response."""
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None, solution_str

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parse the Ground Truth section and return a dictionary of roles and their statuses."""
    status_dict = {}
    pattern = re.compile(r'Found:\s*([\w\s]+)\s*→\s*(knight|knave)', re.IGNORECASE)
    for match in pattern.finditer(solution_text):
        name, role = match.groups()
        status_dict[name.strip()] = role.strip().lower()
    return status_dict

def parse_model_answer(answer_text: str, expected_names: List[str]) -> Optional[Dict[str, str]]:
    """Parse the model's answer and return a dictionary of roles and their statuses."""
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
    """Parse epoch and step information from the log content."""
    epoch_pattern = re.compile(r"epoch\s+(\d+)", re.IGNORECASE)
    step_pattern = re.compile(r"step\s+(\d+)", re.IGNORECASE)
    
    epoch_match = epoch_pattern.search(log_content)
    step_match = step_pattern.search(log_content)

    epoch = int(epoch_match.group(1)) if epoch_match else None
    step = int(step_match.group(1)) if step_match else None
    
    return epoch, step

def parse_log(log_content: str) -> Tuple[List[Dict], int, int]:
    """
    Parse the log content and extract structured data.
    
    Returns:
        parsed_data: A list of parsed log entries.
        total_instances: Total number of instances in the log.
        invalid_instances: Number of invalid instances in the log.
    """
    sample_pattern = re.compile(r"=+\n=+ Processing New Sample =+\n")
    ground_truth_pattern = re.compile(r"\[Ground Truth Parsing\](.*?)\[Ground Truth\] Final identities: (.*?)\n", re.S)
    model_response_pattern = re.compile(r"\[Model Response\]\n\n<think>(.*?)</think>\n<answer>(.*?)</answer>", re.S)
    final_score_pattern = re.compile(r"Final Score.*?Format: (.*?)\n.*?Answer: (.*?)\n.*?Total: (.*?)\n", re.S)

    samples = sample_pattern.split(log_content)
    parsed_data = []

    # Track invalid instances
    invalid_instances = 0

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

        # Initialize epoch and step to 0 if they are still None
        if current_epoch is None:
            current_epoch = 0
        if current_step is None:
            current_step = 0

        # Assign the current epoch and step to the sample
        parsed_sample["epoch"] = current_epoch
        parsed_sample["step"] = current_step

        # Extract Ground Truth
        ground_truth_match = ground_truth_pattern.search(sample)
        if ground_truth_match:
            raw_gt = ground_truth_match.group(1).strip()
            parsed_sample["ground_truth"] = parse_solution_text_format(raw_gt)
        else:
            parsed_sample["ground_truth"] = None  # Default when ground truth is missing

        # Extract Model Response
        model_response_match = model_response_pattern.search(sample)
        if model_response_match:
            parsed_sample["model_think"] = model_response_match.group(1).strip()
            parsed_sample["model_answer"] = model_response_match.group(2).strip()
        else:
            parsed_sample["model_think"] = None  # Default for missing <think> tag
            parsed_sample["model_answer"] = None  # Default for missing <answer> tag

        # Check if the instance is invalid
        if parsed_sample["model_think"] is None or parsed_sample["model_answer"] is None:
            invalid_instances += 1

        # Extract Final Score
        final_score_match = final_score_pattern.search(sample)
        if final_score_match:
            parsed_sample["final_score"] = {
                "format": float(final_score_match.group(1).strip()),
                "answer": float(final_score_match.group(2).strip()),
                "total": float(final_score_match.group(3).strip())
            }
        else:
            parsed_sample["final_score"] = None  # Default when final score is missing

        parsed_data.append(parsed_sample)

    total_instances = len(parsed_data)
    return parsed_data, total_instances, invalid_instances

def save_to_json(data, output_file):
    """Save the parsed results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

def main():
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Parse log files and generate a JSON file.")
    parser.add_argument("log_file", help="Path to the log file.")
    parser.add_argument("-o", "--output", default=OUTPUT_FILE, help="Path to the output JSON file.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Log file {args.log_file} does not exist!")
        return

    with open(log_path, "r") as f:
        log_content = f.read()

    # Parse the log content
    parsed_data, total_instances, invalid_instances = parse_log(log_content)
    valid_instances = total_instances - invalid_instances

    # Save to JSON
    save_to_json(parsed_data, args.output)
    print(f"Parsing completed! Results saved to {args.output}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"- Total instances: {total_instances}")
    print(f"- Invalid instances: {invalid_instances}")
    print(f"- Valid instances: {valid_instances}")

if __name__ == "__main__":
    main()