# Logic-RL-Lite: Lightweight Replication of DeepSeek-R1-Zero

**Logic-RL-Lite** is a lightweight replication study of the [DeepSeek-R1-Zero](https://github.com/deepseek-ai/DeepSeek-R1) framework. This project investigates the use of **pure reinforcement learning (RL)** without supervised fine-tuning (SFT) to post-train base models for reasoning capabilities. It follows up on the Logic-RL project.

It leverages the following key components:

1. RL Framework: **veRL**
2. RL Dataset: **Knights and Knaves (K&K) Logic Puzzle Dataset**
3. Base Models: **Qwen2.5** (1.5B, 3B), **Llama3.2** (3B)

---

## Key Findings

### 1. **Smallest Model Capable of Learning Reasoning**
- **1.5B Models and Smaller**:
  - Instruction-tuned or pretrained models cannot learn reasoning.
- **3B Models**:
  - **Instruction-tuned models** (e.g., Qwen2.5-3B) can learn reasoning.
  - **Pretrained models** (e.g., Llama3.2-3B) struggle to learn reasoning.
  - Hypothesis: **Qwen2.5-3B-Pretrain** is likely somewhat instruction-tuned, making it significantly more capable than Llama3.2-3B-Pretrain.
- **7B Models and Larger**:
  - Consistently learn reasoning.

---

### 2. **No "Aha Moment" During Pure RL**
- Self-reflection and rethinking behaviors **appear at epoch 0** (or even step 0) in **instruction-tuned base models**.
- These behaviors likely stem from **instruction tuning**, rather than emergent properties of pure RL.
- See findings from [OAT-ZERO](https://github.com/sail-sg/oat-zero) and [Logic-RL](https://github.com/Unakar/Logic-RL).

---

### 3. **Longer Chain-of-Thought (CoT) ≠ Higher Accuracy**
- While CoT becomes longer and the mean rewards increase, longer CoT does not correlate with higher accuracy.
- See **superficial self-reflection** findings from [OAT-ZERO](https://github.com/sail-sg/oat-zero).

---

### 4. **Language Mixing**
- **Instruction-Tuned Models**:
  - Rare occurrences of language mixing.
- **Pretrained Models**:
  - Language mixing is prevalent.

---

### 5. **Stability of RL Algorithms**
- **Reinforce++** appears more stable than **GRPO**.
- Further experiments are expected.

---

## Acknowledgements

This project builds upon and references several open-source works:

- **[veRL Framework](https://github.com/volcengine/verl)**: Reinforcement learning framework.
- **[Logic-RL](https://github.com/Unakar/Logic-RL)**: Reproduction of R1-Zero on logic puzzles.
- **[OAT-ZERO](https://github.com/sail-sg/oat-zero)**: Insights on reasoning with pure RL.
- **[TinyZero](https://github.com/Jiayi-Pan/TinyZero)**: Implementation of reward models and Countdown task.
- **[DeepScaler](https://github.com/agentica-project/deepscaler)**: Iterative context scaling with GRPO.
- **[Knights and Knaves (K&K) Puzzle Dataset](https://github.com/AlphaPav/mem-kk-logic)**: Memory-based logic tasks.
