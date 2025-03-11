import json
from typing import List, Dict, Tuple
from datetime import datetime

# File path
PARSED_LOG_FILE = "parsed_logs.json"

def detect_english_and_chinese(text: str) -> Tuple[bool, bool]:
    """
    Detect if the text contains English and Chinese characters.
    
    Args:
        text: The text to be checked. If None, it will be treated as an empty string.
        
    Returns:
        A tuple (contains_english, contains_chinese), indicating whether the text contains English and Chinese.
    """
    if text is None:
        text = ""  # Treat None as an empty string
    contains_english = any('a' <= char.lower() <= 'z' for char in text)
    contains_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    return contains_english, contains_chinese


def analyze_language_in_logs(file_path: str):
    """
    Analyze the distribution and mixing of English and Chinese in `model_think` and `model_answer` fields from the log file,
    and save the results to a timestamped .txt file in Markdown table format.
    
    Args:
        file_path: Path to the parsed log file.
    """
    try:
        with open(file_path, "r") as f:
            parsed_logs = json.load(f)
    except FileNotFoundError:
        print(f"File {file_path} does not exist!")
        return
    except json.JSONDecodeError:
        print(f"File {file_path} is not a valid JSON file!")
        return

    # Extract all `model_think` and `model_answer` texts
    model_think_texts = [entry.get("model_think", "") for entry in parsed_logs]
    model_answer_texts = [entry.get("model_answer", "") for entry in parsed_logs]

    # Initialize statistics
    think_stats = {"english_only": 0, "chinese_only": 0, "mixed": 0}
    answer_stats = {"english_only": 0, "chinese_only": 0, "mixed": 0}

    think_mixed_entries = []
    answer_mixed_entries = []

    # Analyze the language distribution in `model_think`
    for idx, text in enumerate(model_think_texts):
        contains_english, contains_chinese = detect_english_and_chinese(text)
        if contains_english and contains_chinese:
            think_stats["mixed"] += 1
            think_mixed_entries.append({"index": idx, "text": text})
        elif contains_english:
            think_stats["english_only"] += 1
        elif contains_chinese:
            think_stats["chinese_only"] += 1

    # Analyze the language distribution in `model_answer`
    for idx, text in enumerate(model_answer_texts):
        contains_english, contains_chinese = detect_english_and_chinese(text)
        if contains_english and contains_chinese:
            answer_stats["mixed"] += 1
            answer_mixed_entries.append({"index": idx, "text": text})
        elif contains_english:
            answer_stats["english_only"] += 1
        elif contains_chinese:
            answer_stats["chinese_only"] += 1

    # Calculate totals
    total_think = len(model_think_texts)
    total_answer = len(model_answer_texts)

    # Get the current timestamp and generate the output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"language_mixing_{timestamp}.txt"

    # Save the results to the file in Markdown table format
    with open(output_file, "w", encoding="utf-8") as f:
        # Write summary statistics for `model_think`
        f.write("## Language Distribution in `model_think`\n\n")
        f.write("| Category               | Count | Percentage |\n")
        f.write("|------------------------|-------|------------|\n")
        f.write(f"| Only English           | {think_stats['english_only']} | {think_stats['english_only'] / total_think * 100:.2f}% |\n")
        f.write(f"| Only Chinese           | {think_stats['chinese_only']} | {think_stats['chinese_only'] / total_think * 100:.2f}% |\n")
        f.write(f"| Mixed (English & Chinese) | {think_stats['mixed']} | {think_stats['mixed'] / total_think * 100:.2f}% |\n\n")

        # Write summary statistics for `model_answer`
        f.write("## Language Distribution in `model_answer`\n\n")
        f.write("| Category               | Count | Percentage |\n")
        f.write("|------------------------|-------|------------|\n")
        f.write(f"| Only English           | {answer_stats['english_only']} | {answer_stats['english_only'] / total_answer * 100:.2f}% |\n")
        f.write(f"| Only Chinese           | {answer_stats['chinese_only']} | {answer_stats['chinese_only'] / total_answer * 100:.2f}% |\n")
        f.write(f"| Mixed (English & Chinese) | {answer_stats['mixed']} | {answer_stats['mixed'] / total_answer * 100:.2f}% |\n\n")

        # Write mixed text details for `model_think`
        f.write("## Mixed Texts in `model_think`\n\n")
        f.write("| Index | Text |\n")
        f.write("|-------|------|\n")
        for entry in think_mixed_entries:
            f.write(f"| {entry['index']} | {entry['text']} |\n")

        # Write mixed text details for `model_answer`
        f.write("\n## Mixed Texts in `model_answer`\n\n")
        f.write("| Index | Text |\n")
        f.write("|-------|------|\n")
        for entry in answer_mixed_entries:
            f.write(f"| {entry['index']} | {entry['text']} |\n")

    print(f"Analysis results have been saved to the file: {output_file}")


if __name__ == "__main__":
    analyze_language_in_logs(PARSED_LOG_FILE)