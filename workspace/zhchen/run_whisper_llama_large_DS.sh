NEMO_DIR=/NeMo
EXP_DIR=/NeMo/workspace/nemo_experiments
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0


MEGATRON_CKPT=/NeMo/data/llama2-7b-chat.nemo

TRAIN_MANIFESTS=/NeMo/data/dynamic-superb/0228_train.jsonl
VAL_MANIFESTS=/NeMo/data/dynamic-superb/0228_val.50.jsonl

exp_name="llama7B-whisperL-DSFT/0228_train.jsonl"

# Configs
GLOBAL_BATCH=12
MICRO_BATCH=12

ASR_MODEL="openai/whisper-large-v3" # huggingface id
perception_mode="cnn_1"
prompt_size=60
random_transcription_prob=0.0

devices=4

# Message
message="
train: $TRAIN_MANIFESTS
val: $VAL_MANIFESTS

perception_mode: $perception_mode
prompt_size: $prompt_size
random_transcription_prob: $random_transcription_prob
"
mkdir -p $EXP_DIR/$exp_name
echo -e "=======================" >> $EXP_DIR/$exp_name/readme.txt
echo -e $(date) $0 >> $EXP_DIR/$exp_name/readme.txt
printf "%s" "$message" >> $EXP_DIR/$exp_name/readme.txt
echo -e "=======================" >> $EXP_DIR/$exp_name/readme.txt


NCCL_DEBUG=WARN HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python \
run_sft_whisper_llama.py --config-path="../examples/multimodel/conf/khlu/" --config-name "whisper_llama_config_7b.yaml" \
    name=$exp_name \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.peft.restore_from_path=$restore_from_path \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    ++model.data.train_ds.random_context_prob=0 \
    ++model.data.train_ds.random_context_num=0 \
    ++model.data.train_ds.random_transcription_prob=$random_transcription_prob \
    ++model.data.validation_ds.random_context_prob=0 \
    ++model.data.validation_ds.random_context_num=0 \
    ++model.data.validation_ds.output_file_path_prefix=$EXP_DIR/$exp_name/ \
    ++model.data.validation_ds.write_predictions_to_file=True \
    ++model.data.validation_ds.random_transcription_prob=$random_transcription_prob \
    ++model.data.validation_ds.tokens_to_generate=20 \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.tensor_model_parallel_size=2 \
    ++model.pipeline_model_parallel_size=1 \
    ++trainer.num_nodes=1 \
    ++trainer.devices=$devices \
    ++model.perception.mode=$perception_mode \
    ++trainer.max_epochs=5 \
    ++model.perception.prompt_size=$prompt_size
