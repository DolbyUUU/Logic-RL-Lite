# Logic-RL-Lite: Lightweight Replication of DeepSeek-R1-Zero

**Logic-RL-Lite** is a lightweight replication study of the [DeepSeek-R1-Zero](https://github.com/deepseek-ai/DeepSeek-R1) framework. This project investigates the use of **pure reinforcement learning (RL)** without supervised fine-tuning (SFT) to post-train base models for reasoning capabilities. It follows up Logic-RL project.

It leverages the following key components:

1. **veRL Framework** (developed by ByteDance)  
2. **Knights and Knaves (K&K) Logic Puzzle Dataset** (provided by the Logic-RL project)  
3. **Small-Scale Base Models**, including:
   - **Qwen2.5** (1.5B, 3B)
   - **Llama3.2** (3B)

---

## Key Findings

### 1. **Smallest Model Capable of Learning Reasoning via Pure RL**
- **1.5B Models**:
  - Instruction-tuned or pretrained models cannot learn reasoning.  
- **3B Models**:
  - **Instruction-tuned models** (e.g., Qwen2.5-3B) can learn reasoning.  
  - **Pretrained-only models** (e.g., Llama3.2-3B) struggle to learn reasoning.  
  - Hypothesis: **Qwen2.5-3B-Pretrain** is probably somewhat instruction tuned, making it significantly more capable than Llama3.2-3B-Pretrain.  
- **7B Models and Larger**:
  - Models of this scale consistently and easily learn reasoning.

---

### 2. **No "Aha Moment" During Pure RL**
- Self-reflection and rethinking behaviors **appear at epoch 0** (or even step 0) in **instruction-tuned base models**.  
- These behaviors likely stem from **instruction tuning**, rather than emergent properties of pure RL.  
- This aligns with findings from [oat-zero](https://github.com/sail-sg/oat-zero) and .  

---

### 3. **Longer Chain-of-Thought (CoT) ≠ Higher Accuracy**
- While long CoT responses often lead to higher **rewards**, they do not necessarily translate to higher accuracy.  
- This phenomenon highlights the presence of **superficial self-reflection** in the model's reasoning processes.  

---

### 4. **Language Mixing and Nonsense Outputs**
- **Instruction-Tuned Models**:
  - Rare instances of language mixing during reasoning tasks.  
- **Pretrained Models**:
  - Prevalent language mixing and nonsensical outputs during reasoning tasks.  

---

### 5. **Stability of Reinforcement Learning Algorithms**
- **Reinforce++** appears to be more stable than **GRPO** for fine-tuning reasoning capabilities with pure RL.  
- Further experiments are required to confirm this observation.  

---

## Acknowledgements

This project builds upon and references several open-source frameworks and datasets:

- **[veRL Framework](https://github.com/ByteDance/veRL)**: Reinforcement learning framework.  
- **[Knights and Knaves Dataset](https://github.com/Logic-RL/knights-knaves)**: Logic puzzle dataset.  
- **[OAT-ZERO](https://github.com/sail-sg/oat-zero)**: Key insights and findings on reasoning with pure RL.  
- **[TinyZero](https://github.com/some-repo/tinyzero)**: Implementation of reward models and Countdown task.  
- **[vLLM](https://github.com/vllm/vllm)**: Accelerated inference for large language models.  
https://github.com/sail-sg/oat-zero
https://github.com/volcengine/verl
https://github.com/deepseek-ai/DeepSeek-R1
https://github.com/sail-sg/oat-zero
https://github.com/Unakar/Logic-RL
https://github.com/Jiayi-Pan/TinyZero
https://github.com/agentica-project/deepscaler
https://github.com/AlphaPav/mem-kk-logic
