import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# File path
PARSED_LOG_FILE = "parsed_logs.json"

def find_word_occurrences(parsed_data: List[Dict], direct_words: List[str], regex_words: List[str]) -> Tuple[Dict[str, Optional[Dict[str, int]]], Dict[str, int]]:
    """
    Check the first occurrence of specified words in "model_think" and record the epoch and step.
    Also calculate statistical data for each word.

    Args:
        parsed_data: A list of parsed log data.
        direct_words: A list of words for direct matching.
        regex_words: A list of words for regex matching.

    Returns:
        A tuple with:
            - Word occurrences (first epoch and step).
            - Overall statistics of how often each word appears.
    """
    word_occurrences = {word: None for word in direct_words + regex_words}  # Initialize the result dictionary
    word_stats = {word: 0 for word in direct_words + regex_words}  # Count occurrences for each word

    for entry in parsed_data:
        # Safely retrieve and normalize the "model_think" field
        model_think = entry.get("model_think", "")
        if not isinstance(model_think, str):
            model_think = ""  # Default to empty string if not a valid string

        model_think = model_think.lower()  # Convert to lowercase for case-insensitive matching

        # Handle direct matching words
        for word in direct_words:
            if word.lower() in model_think:
                word_stats[word] += 1  # Increment count
                if word_occurrences[word] is None:  # Record first occurrence
                    word_occurrences[word] = {"epoch": entry.get("epoch"), "step": entry.get("step")}

        # Handle regex matching words
        for word in regex_words:
            if re.search(rf'\b{word}\b', model_think, flags=re.IGNORECASE):
                word_stats[word] += 1  # Increment count
                if word_occurrences[word] is None:  # Record first occurrence
                    word_occurrences[word] = {"epoch": entry.get("epoch"), "step": entry.get("step")}

    return word_occurrences, word_stats

def load_parsed_logs(file_path: str) -> List[Dict]:
    """
    Load the parsed log file.

    Args:
        file_path: Path to the parsed log file.

    Returns:
        A list containing the log data.
    """
    try:
        with open(file_path, "r") as f:
            parsed_data = json.load(f)
        return parsed_data
    except FileNotFoundError:
        print(f"File {file_path} does not exist!")
        return []
    except json.JSONDecodeError:
        print(f"File {file_path} is not a valid JSON file!")
        return []

def main():
    # Set of directly matched words
    direct_words = [
        "rethink", "re-think", "think again", 
        "retry", "re-try", "try again", 
        "recheck", "re-check", "check again", 
        "reevaluate", "re-evaluate", 
        "double check", "double-check", 
        "verify", 
        "summarize", "summary"
    ]

    # Set of regex-matched words (case-insensitive)
    regex_words = [
        "aha", "wait"
    ]

    # Load parsed log data
    parsed_logs = load_parsed_logs(PARSED_LOG_FILE)

    if not parsed_logs:
        print("Failed to load parsed log data.")
        return

    # Find the first occurrence of words and their statistics
    occurrences, stats = find_word_occurrences(parsed_logs, direct_words, regex_words)

    # Total number of valid instances
    total_instances = len(parsed_logs)
    invalid_instances = sum(1 for entry in parsed_logs if entry.get("model_think") is None or entry.get("model_answer") is None)
    valid_instances = total_instances - invalid_instances

    # Get the current timestamp and generate a file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"aha_moment_{timestamp}.txt"

    # Write the results to the file
    with open(output_file, "w", encoding="utf-8") as f:
        # Write summary
        f.write(f"# Analysis Summary\n\n")
        f.write(f"- Total instances: {total_instances}\n")
        f.write(f"- Total valid instances: {valid_instances}\n")
        f.write(f"- Total invalid instances: {invalid_instances}\n\n")
        f.write(f"| Word           | First Occurrence (epoch, step) | Instances Found | Percentage (%) |\n")
        f.write(f"|----------------|--------------------------------|----------------|----------------|\n")
        
        for word, location in occurrences.items():
            instances_found = stats[word]
            percentage = (instances_found / valid_instances) * 100 if valid_instances > 0 else 0
            first_occurrence = f"({location['epoch']}, {location['step']})" if location else "N/A"
            f.write(f"| {word:<14} | {first_occurrence:<30} | {instances_found:<14} | {percentage:>14.2f} |\n")

    print(f"Analysis results have been saved to the file: {output_file}")

if __name__ == "__main__":
    main()