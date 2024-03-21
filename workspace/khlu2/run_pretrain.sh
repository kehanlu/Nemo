NEMO_DIR=/NeMo
EXP_DIR=/NeMo/workspace/nemo_experiments

export PYTHONPATH=$NEMO_DIR:$PYTHONPATH
export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0

export WANDB_API_KEY='69fe60f3d102a5898bea902f94d302a595fd1dbe'
wandb login $WANDB_API_KEY



LLM_NAME='llama7b-base'
if [ "$LLM_NAME" = "llama7b-base" ]; then
    MEGATRON_CKPT=/NeMo/data/llama2-7b.nemo
elif [ "$LLM_NAME" = "llama7b-chat" ]; then
    MEGATRON_CKPT=/NeMo/data/TinyLlama-chat.nemo
elif [ "$LLM_NAME" = "debug" ]; then
    MEGATRON_CKPT=/NeMo/data/TinyLlama-v1.test.nemo
fi

ASR_NAME='whisper-medium'
if [ "$ASR_NAME" = "whisper-medium" ]; then
    ASR_MODEL="openai/whisper-medium" # huggingface id
elif [ "$ASR_NAME" = "whisper-large-v3" ]; then
    ASR_MODEL="openai/whisper-large-v3"
else
    # error
    echo "ASR model not found"
    exit 1;
fi


GLOBAL_BATCH=12
MICRO_BATCH=6

perception_mode="cnn_1"
prompt_size=60


TRAIN_MANIFESTS=[/NeMo/data/pretraining_data/0306_pretrain.jsonl]
VAL_MANIFESTS=[/NeMo/data/pretraining_data/val.caption.1000.jsonl,/NeMo/data/pretraining_data/val.100.jsonl]
train_questions=[/NeMo/data/pretraining_data/pretrain_questions_base.txt]
valid_questions=[/NeMo/data/pretraining_data/pretrain_questions_base.txt,/NeMo/data/pretraining_data/pretrain_questions_base.txt]

devices=4
random_transcription_prob=0.0

WANDB_EXP_NAME=$(date +%y%m%d)@${perception_mode}@${LLM_NAME}_${ASR_NAME}@${GLOBAL_BATCH}_${MICRO_BATCH}
WANDB_PROJ_NAME=pretrain-base-llama

exp_name=${WANDB_PROJ_NAME}/${WANDB_EXP_NAME}

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 python \
run_sft_whisper_llama.py --config-path="../examples/multimodel/conf/khlu/" --config-name "pretrain_whisper_llama.yaml" \
    name=$exp_name \
    ++exp_manager.create_wandb_logger=true \
    model.pretrained_audio_model=$ASR_MODEL \
    ++exp_manager.wandb_logger_kwargs.name=${WANDB_EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${WANDB_PROJ_NAME} \
    ++exp_manager.wandb_logger_kwargs.resume=true \
    model.restore_from_path=$MEGATRON_CKPT \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    ++model.data.train_ds.random_context_prob=0 \
    ++model.data.train_ds.random_context_num=0 \
    ++model.data.validation_ds.random_context_prob=0 \
    ++model.data.validation_ds.random_context_num=0 \
    ++model.data.validation_ds.output_file_path_prefix=$EXP_DIR/$exp_name/ \
    ++model.data.validation_ds.write_predictions_to_file=True \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.tensor_model_parallel_size=2 \
    ++model.pipeline_model_parallel_size=1 \
    ++trainer.num_nodes=1 \
    ++trainer.devices=$devices \
    ++model.perception.mode=$perception_mode \
    ++trainer.max_epochs=10 \
    ++model.data.train_ds.question_file=$train_questions \
    ++model.data.validation_ds.question_file=$valid_questions \
    ++model.perception.prompt_size=$prompt_size \
    ++model.data.train_ds.random_transcription_prob=$random_transcription_prob \
    model.optim.sched.min_lr=1e-6 \
    model.optim.sched.warmup_steps=2500 \
    model.data.validation_ds.tokens_to_generate=100 \





