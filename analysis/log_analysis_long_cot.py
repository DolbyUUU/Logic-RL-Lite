import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer
from datetime import datetime

# File paths
json_file_path = "parsed_logs.json"  # Path to the JSON file
output_dir = "plots/"  # Output folder path

# Ensure the output folder exists
os.makedirs(output_dir, exist_ok=True)

# Load Tokenizer (using a generic pretrained model tokenizer, e.g., BERT or GPT-2)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with other models if needed

# Load JSON data
with open(json_file_path, "r") as file:
    data = json.load(file)

# Define a function to calculate token counts using tokenizer
def count_tokens_with_tokenizer(text):
    if not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text, truncation=True, add_special_tokens=False))

# Extract necessary data
answer_scores = []  # Store the `answer` values from `final_score`
think_token_counts = []  # Store token counts for `model_think`
answer_raw_token_counts = []  # Store token counts for `model_answer_raw`

for entry in data:
    # Extract the answer score
    answer_score = entry.get("final_score", {}).get("answer", None)
    if answer_score is not None:  # Ensure `answer` exists
        answer_scores.append(answer_score)
        
        # Calculate token counts for `model_think` and `model_answer_raw`
        think_token_counts.append(count_tokens_with_tokenizer(entry.get("model_think", "")))
        answer_raw_token_counts.append(count_tokens_with_tokenizer(entry.get("model_answer_raw", "")))

# Create a DataFrame for Seaborn visualization
df = pd.DataFrame({
    "answer": answer_scores,
    "think_tokens": think_token_counts,
    "answer_raw_tokens": answer_raw_token_counts
})

# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Plot regression plots
plt.figure(figsize=(12, 6))

# Plot 1: Regression plot of `answer` vs `model_think` token count
plt.subplot(1, 2, 1)
sns.regplot(x="think_tokens", y="answer", data=df, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.title("Regression Plot of 'answer' vs 'model_think' tokens")
plt.xlabel("Number of tokens in 'model_think'")
plt.ylabel("Value of 'answer'")
plt.grid(True)

# Plot 2: Regression plot of `answer` vs `model_answer_raw` token count
# Uncomment the following block if you want to include this plot
# plt.subplot(1, 2, 2)
# sns.regplot(x="answer_raw_tokens", y="answer", data=df, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
# plt.title("Regression Plot of 'answer' vs 'model_answer_raw' tokens")
# plt.xlabel("Number of tokens in 'model_answer_raw'")
# plt.ylabel("Value of 'answer'")
# plt.grid(True)

# Save regression plots
output_regression_path = os.path.join(output_dir, f"regression_answer_vs_tokens_{timestamp}.png")
plt.tight_layout()
plt.savefig(output_regression_path, dpi=300)
print(f"Regression plot saved to: {output_regression_path}")

# Plot grouped bar plots
# Group token counts into bins
think_bins = pd.cut(df["think_tokens"], bins=10)
answer_raw_bins = pd.cut(df["answer_raw_tokens"], bins=10)

# Calculate the mean `answer` value for each group
think_group_mean = df.groupby(think_bins)["answer"].mean()
answer_raw_group_mean = df.groupby(answer_raw_bins)["answer"].mean()

# Plot bar plots
plt.figure(figsize=(12, 6))

# Plot 1: Bar plot for `model_think` token count ranges
plt.subplot(1, 2, 1)
think_group_mean.plot(kind="bar", color="blue", alpha=0.7)
plt.title("Mean 'answer' vs 'model_think' token ranges")
plt.xlabel("Token count ranges in 'model_think'")
plt.ylabel("Mean reward value of 'answer'")
plt.xticks(rotation=45)

# Plot 2: Bar plot for `model_answer_raw` token count ranges
# Uncomment the following block if you want to include this plot
# plt.subplot(1, 2, 2)
# answer_raw_group_mean.plot(kind="bar", color="green", alpha=0.7)
# plt.title("Mean 'answer' vs 'model_answer_raw' token ranges")
# plt.xlabel("Token count ranges in 'model_answer_raw'")
# plt.ylabel("Mean reward value of 'answer'")
# plt.xticks(rotation=45)

# Save bar plots
output_barplot_path = os.path.join(output_dir, f"barplot_answer_vs_tokens_{timestamp}.png")
plt.tight_layout()
plt.savefig(output_barplot_path, dpi=300)
print(f"Bar plot saved to: {output_barplot_path}")

# Show plots (optional)
plt.show()