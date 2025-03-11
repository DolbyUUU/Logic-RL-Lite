# Logic-RL-Lite: Lightweight Replication of DeepSeek-R1-Zero and Result Analysis

**Logic-RL-Lite** is a lightweight replication study of the [DeepSeek-R1-Zero](https://github.com/deepseek-ai/DeepSeek-R1) framework. This project investigates the use of **pure reinforcement learning (RL)** without supervised fine-tuning (SFT) to post-train base models for reasoning capabilities. It is a follow-up of the Logic-RL project.

It leverages the following key components:

1. RL Framework: **[veRL](https://github.com/volcengine/verl)**
2. RL Algorithms: [**REINFORCE++**](https://arxiv.org/html/2501.03262v1) and [**GRPO**](https://arxiv.org/abs/2402.03300)
3. RL Dataset: **[Knights and Knaves (K&K) Logic Puzzle Dataset](https://github.com/AlphaPav/mem-kk-logic)**
4. Base Models: Qwen2.5 (1.5B, 3B), Llama3.2 (3B)

---

## Dataset

Knights and Knaves (K&K) Logic Puzzle: Imagine there are two types of people: **Knights** and **Knaves**. Knights always tell the truth. Knaves always lie.  

The K&K dataset is designed to test logical reasoning capabilities by presenting puzzles involving statements made by multiple "people," where the goal is to determine who is a knight and who is a knave based on the given clues.

---

## RL Reward Design
1. Format Reward: Yes
2. Answer Reward: Yes
3. Language Consistency Reward or Others: No

---

## Training
After configuring your WandB, GPUs, and other settings, execute the training:  
```bash
bash run_rl_trainer_xxx.sh
```

---

## Key Findings

For more visualized details, refer to my WandB report:  
**[Logic-RL-Lite Training Report](https://wandb.ai/yuwang91-hk/Logic-RL-Lite/reports/Logic-RL-Lite-Lightweight-Replication-of-DeepSeek-R1-Zero--VmlldzoxMTU5ODkzNQ)**

Note: The findings may be specific to this experimentation setup.

### 1. **Smallest Model Capable of Learning Reasoning**
- **1.5B Models and Smaller**:
  - Instruction-tuned or pretrained models cannot learn reasoning.
- **3B Models**:
  - **Instruction-tuned models** (e.g., Qwen2.5-3B) can learn reasoning.
  - **Pretrained models** (e.g., Llama3.2-3B) struggle to learn reasoning.
- **7B Models and Larger**:
  - Consistently learn reasoning.

---

### 2. **Base Model Selection Matters**
- Cognitive differences between **Qwen2.5-3B** and **Llama3.2-3B** are discussed in [this paper](https://arxiv.org/abs/2503.01307).  
- Qwen2.5-3B demonstrates stronger instruction-following behavior compared to Llama3.2-3B.  
- Hypothesis: Qwen2.5-3B may have undergone partial instruction tuning.

---

### 3. **No "Aha Moment" During Pure RL**
- Self-reflection and rethinking behaviors **appear at epoch 0** (or even step 0) in **instruction-tuned base models**.
- These behaviors likely stem from **instruction tuning**, rather than emergent properties of pure RL.
- See findings from [OAT-ZERO](https://github.com/sail-sg/oat-zero) and [Logic-RL](https://github.com/Unakar/Logic-RL).

#### Table: Appearance of Self-Reflection and Verification Keywords During Training (Base Model = Qwen2.5-3B-Instruct)

| Keyword         | Epoch | Step |
|------------------|-------|------|
| rethink          | 0     | 4    |
| re-think         | N/A   | N/A  |
| think again      | N/A   | N/A  |
| retry            | N/A   | N/A  |
| re-try           | N/A   | N/A  |
| try again        | N/A   | N/A  |
| recheck          | 0     | 0    |
| re-check         | 0     | 14   |
| check again      | 0     | 52   |
| reevaluate       | 0     | 121  |
| re-evaluate      | 0     | 0    |
| double check     | 0     | 1    |
| double-check     | 0     | 7    |
| verify           | 0     | 1    |
| aha              | N/A   | N/A  |
| wait             | 0     | 63   |

#### Table: Appearance of Summarization Keywords During Training (Base Model = Qwen2.5-3B-Instruct)

| Keyword         | Epoch | Step |
|------------------|-------|------|
| summarize        | 0     | 1    |
| summary          | 0     | 0    |

---

### 4. **Longer Chain-of-Thought (CoT) ≠ Higher Accuracy**
- While CoT becomes longer and the mean rewards increase, longer CoT does not correlate with higher accuracy.
- This aligns with **superficial self-reflection** findings from [OAT-ZERO](https://github.com/sail-sg/oat-zero).

#### Figures (Base Model = Qwen2.5-3B-Instruct):
- **Left Figure**: Answer accuracy versus token count distribution.  
- **Right Figure**: Regression analysis of accuracy against token count.  

<div style="display: flex; justify-content: space-between; gap: 1px;">

<img src="analysis/QWEN3B-INSTRUCT-KKLOGIC-3/plots/barplot_answer_vs_tokens_20250302_180806.png" alt="Barplot: Answer Accuracy vs Token Count" style="width: 48%;">

<img src="analysis/QWEN3B-INSTRUCT-KKLOGIC-3/plots/regression_answer_vs_tokens_20250302_180806.png" alt="Regression: Accuracy vs Token Count" style="width: 48%;">

</div>

---

### 5. **Language Mixing**
- **Instruction-Tuned Model as Base Model**:
  - Rare occurrences of language mixing.
- **Pretrained Model as Base Model**:
  - Language mixing is more prevalent.

#### Table: Language Distribution in Model Outputs (Base Model = Qwen2.5-3B-Instruct)
| Output Type         | Only English | Only Chinese | Mixed (English & Chinese) |
|----------------------|--------------|--------------|---------------------------|
| `model_think`        | 98.71%       | 0.00%        | 0.82%                     |
| `model_answer_raw`   | 99.44%       | 0.00%        | 0.00%                     |

---

### 6. **RL Algorithm Stability**
- **REINFORCE++** demonstrates greater stability compared to **GRPO** during training.  
- Further experiments are required to validate this observation.  
- For a technical comparison of **REINFORCE++**, **GRPO**, and **PPO**, see [this report](https://hijkzzz.notion.site/reinforce-plus-plus).

---

## Acknowledgements

This project builds upon and references several open-source works:

- **[veRL Framework](https://github.com/volcengine/verl)**: Reinforcement learning framework.
- **[Logic-RL](https://github.com/Unakar/Logic-RL)**: Reproduction of R1-Zero on logic puzzles.
- **[OAT-ZERO](https://github.com/sail-sg/oat-zero)**: Insights on reasoning with pure RL.
- **[TinyZero](https://github.com/Jiayi-Pan/TinyZero)**: Implementation of reward models and Countdown task.
- **[DeepScaler](https://github.com/agentica-project/deepscaler)**: Iterative context scaling with GRPO.
- **[Knights and Knaves (K&K) Puzzle Dataset](https://github.com/AlphaPav/mem-kk-logic)**: Logical reasoning tasks for LLMs.
