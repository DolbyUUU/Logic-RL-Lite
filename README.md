# Logic-RL-Small
Replication study of DeepSeek-R1. 

Explores pure RL without SFT for post-training for reasoning capability, leveraging 

(1) veRL framework, 
(2) Knights and Knaves (K&K) logic puzzle dataset, and 
(3) small-scale base model. 

Findings
(1) The smallest model that can learn reasoning with pure RL
1.5B-instruct/pretrained cannot learn (citation)
3B-instruct model can learn
3B-pretrain model is difficult to learn (Qwen-2.5-3B sure can, but Llama-3.2-3B looks nope) 
[hypothesis: Qwen-2.5-3B pretrain version is instruction tuned, because it is way much smarter than Llama-3.2-3B-pretrain]
7B and larger model can easily learn

(2) No "aha moment"  (citation 1 and 2)
reflections and rethiking actions appear at epoch 0 and even at step 0 when base model is instruct model
probably reflections and rethiking capability come from instruciton tuning

(3) long cot does not necessarily lead to higher accuracy (citation)

(4) language mixing appears (citation)
when base model is instruct model, language mixing appears but rare
when base model is pretrain model, language mixing is prevalent and so does nonsenses

(5) reinforce++ is more stable than grpo 
need more experiments to confirm
