#!/bin/bash
NEMO_DIR=/NeMo
EXP_DIR=$NEMO_DIR/workspace/nemo_experiments
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH
export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export HF_DATASETS_CACHE="/NeMo/.cache"


WANDB_API_KEY=''
wandb login $WANDB_API_KEY


# Exp
CUDA_VISIBLE_DEVICES=$1
# CUDA_VISIBLE_DEVICES=2,3
# devices=1
devices=2

config_name="whisper_llama"


project_name="MISTA"
name=$3

dataset_config=$2


restore_from_path=null

train_batch_size=8

# trainer
max_epochs=10
# optim
lr=1e-4
warmup_steps=2000

# Model
# lora=null


# connector
connector_mode="qformer_1"
connector_prompt_size=64

# generation config
max_new_tokens=128
do_sample=False

# exp_manager
create_wandb_logger=true


wandb_project_name=${project_name}
wandb_exp_name=$(date +%y%m%d-%H)@$(echo ${dataset_config} | sed 's|/|+|g')@${name}

exp_name=$wandb_project_name/$wandb_exp_name # dir for this exp

save_dir=$EXP_DIR/$exp_name
mkdir -p $save_dir/backup
cat $0 > $save_dir/backup/run.sh
echo $restore_from_path > $save_dir/from_pretrained

read -r -d '' COMMAND << EOF
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "$NEMO_DIR/examples/desta/run_speech_llama.py" \\
    --config-name "$config_name" \\
    save_dir="$save_dir/" \\
    +dataset="$dataset_config" \\
    name="$exp_name" \\
    trainer.max_epochs=$max_epochs \\
    trainer.devices=$devices \\
    model.optim.lr=$lr \\
    model.optim.sched.warmup_steps=$warmup_steps \\
    ++model.restore_from_path="$restore_from_path" \\
    ++exp_manager.exp_dir="$EXP_DIR" \\
    ++exp_manager.create_wandb_logger=$create_wandb_logger \\
    ++exp_manager.wandb_logger_kwargs.project="$wandb_project_name" \\
    ++exp_manager.wandb_logger_kwargs.name="$wandb_exp_name" \\
    ++exp_manager.wandb_logger_kwargs.save_dir="$save_dir/" \\
    ++dataset.train_ds.batch_size=$train_batch_size \\
    ++model.generation_config.max_new_tokens=$max_new_tokens \\
    ++model.generation_config.do_sample=$do_sample \\
    ++dataset.num_workers=0
EOF

echo "$COMMAND"
echo "$COMMAND" > $save_dir/cmd.sh

sleep 1
NCCL_DEBUG=WARN eval "$COMMAND"

exit 0
