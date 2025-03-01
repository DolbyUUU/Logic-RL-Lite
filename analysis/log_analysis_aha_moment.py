import json
import re
from typing import List, Dict, Optional
from datetime import datetime

# File path
PARSED_LOG_FILE = "parsed_logs.json"

def find_word_occurrences(parsed_data: List[Dict], direct_words: List[str], regex_words: List[str]) -> Dict[str, Optional[Dict[str, int]]]:
    """
    Check the first occurrence of specified words in "model_think" and record the epoch and step.

    Args:
        parsed_data: A list of parsed log data.
        direct_words: A list of words for direct matching.
        regex_words: A list of words for regex matching.

    Returns:
        A dictionary where the keys are the words and the values are dictionaries containing the first occurrence's epoch and step.
        If the word does not occur, the value is None.
    """
    word_occurrences = {word: None for word in direct_words + regex_words}  # Initialize the result dictionary

    for entry in parsed_data:
        epoch = entry.get("epoch")
        step = entry.get("step")
        model_think = entry.get("model_think", "").lower()  # Convert to lowercase for case-insensitive matching

        # Handle direct matching words
        for word in direct_words:
            if word_occurrences[word] is None and word.lower() in model_think:
                word_occurrences[word] = {"epoch": epoch, "step": step}

        # Handle regex matching words
        for word in regex_words:
            if word_occurrences[word] is None and re.search(rf'\b{word}\b', model_think, flags=re.IGNORECASE):
                word_occurrences[word] = {"epoch": epoch, "step": step}

    return word_occurrences

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

    # Find the first occurrence of words in terms of epoch and step
    occurrences = find_word_occurrences(parsed_logs, direct_words, regex_words)

    # Get the current timestamp and generate a file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"aha_moment_{timestamp}.txt"

    # Write the results to the file
    with open(output_file, "w", encoding="utf-8") as f:
        for word, location in occurrences.items():
            if location:
                result = f"The word '{word}' first appeared at epoch {location['epoch']}, step {location['step']}.\n"
            else:
                result = f"The word '{word}' did not appear in the logs.\n"
            f.write(result)

    print(f"Analysis results have been saved to the file: {output_file}")

if __name__ == "__main__":
    main()