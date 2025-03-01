import json
from typing import List, Dict, Tuple
from datetime import datetime

# File path
PARSED_LOG_FILE = "parsed_logs.json"

def detect_english_and_chinese(text: str) -> Tuple[bool, bool]:
    """
    Detect if the text contains English and Chinese characters.
    
    Args:
        text: The text to be checked.
        
    Returns:
        A tuple (contains_english, contains_chinese), indicating whether the text contains English and Chinese.
    """
    contains_english = any('a' <= char.lower() <= 'z' for char in text)
    contains_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    return contains_english, contains_chinese


def analyze_language_in_logs(file_path: str):
    """
    Analyze the distribution and mixing of English and Chinese in `model_think` and `model_answer_raw` fields from the log file,
    and save the results to a timestamped file.
    
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

    # Extract all `model_think` and `model_answer_raw` texts
    model_think_texts = [entry.get("model_think", "") for entry in parsed_logs]
    model_answer_raw_texts = [entry.get("model_answer_raw", "") for entry in parsed_logs]

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

    # Analyze the language distribution in `model_answer_raw`
    for idx, text in enumerate(model_answer_raw_texts):
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
    total_answer = len(model_answer_raw_texts)

    # Get the current timestamp and generate the output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"language_mixing_{timestamp}.txt"

    # Save the results to the file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Language distribution in `model_think`:\n")
        f.write(f"Texts containing only English: {think_stats['english_only']} ({think_stats['english_only'] / total_think * 100:.2f}%)\n")
        f.write(f"Texts containing only Chinese: {think_stats['chinese_only']} ({think_stats['chinese_only'] / total_think * 100:.2f}%)\n")
        f.write(f"Texts containing both English and Chinese: {think_stats['mixed']} ({think_stats['mixed'] / total_think * 100:.2f}%)\n\n")

        f.write("Language distribution in `model_answer_raw`:\n")
        f.write(f"Texts containing only English: {answer_stats['english_only']} ({answer_stats['english_only'] / total_answer * 100:.2f}%)\n")
        f.write(f"Texts containing only Chinese: {answer_stats['chinese_only']} ({answer_stats['chinese_only'] / total_answer * 100:.2f}%)\n")
        f.write(f"Texts containing both English and Chinese: {answer_stats['mixed']} ({answer_stats['mixed'] / total_answer * 100:.2f}%)\n\n")

        f.write("Detailed structure of mixed texts in `model_think`:\n")
        for entry in think_mixed_entries:
            f.write(f"Index: {entry['index']}, Text: {entry['text']}\n")

        f.write("\nDetailed structure of mixed texts in `model_answer_raw`:\n")
        for entry in answer_mixed_entries:
            f.write(f"Index: {entry['index']}, Text: {entry['text']}\n")

    print(f"Analysis results have been saved to the file: {output_file}")


if __name__ == "__main__":
    analyze_language_in_logs(PARSED_LOG_FILE)