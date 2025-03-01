set -x

# Environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_THREADING_LAYER=GNU
export TORCH_CUDA_ALLOW_TF32=1
export HYDRA_FULL_ERROR=1
# export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3

# wandb configuration
export WANDB_MODE=online
export WANDB_API_KEY=c2fe654fd0527d4fad92e03cdc1e3f59b9a20595
export WANDB_BASE_URL=https://api.wandb.ai

# Default parameters
BASE_MODEL="Qwen/Qwen2.5-3B"
DATA_DIR="data/kklogic/3ppl/"
EXPERIMENT_NAME="QWEN3BPT-REINFORCE++-KKLOGIC-4RTX4090-$(date +%Y%m%d%H%M)"
PROJECT_NAME="Logic-RL-Lite"
N_GPUS=4
ROLLOUT_TP_SIZE=2

# Clear GPU memory cache
python -c "import torch; torch.cuda.empty_cache()"

# Start training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_hdfs_dir=./saved_models \
    trainer.total_epochs=5 \
    +trainer.val_before_train=True 2>&1 | tee log.log