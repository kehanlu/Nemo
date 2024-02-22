NEMO_DIR=/NeMo
EXP_DIR=/NeMo/workspace/nemo_experiments

export PYTHONPATH=$NEMO_DIR:$PYTHONPATH
export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0

# MEGATRON_CKPT=/NeMo/data/TinyLlama-v1.nemo
# MEGATRON_CKPT=/NeMo/data/TinyLlama-v1.test.nemo
MEGATRON_CKPT=/NeMo/data/llama2-7b-chat.nemo
# ASR_MODEL="ssl_en_conformer_large"
# ASR_MODEL="stt_en_fastconformer_transducer_large"
ASR_MODEL="openai/whisper-large-v3" # huggingface id
GLOBAL_BATCH=4
MICRO_BATCH=4
perception_mode="qformer_1"


# VAL_MANIFESTS=/NeMo/data/PromptTTS/test.jsonl

TRAIN_MANIFESTS=[/NeMo/data/pretraining_data/debug_set.jsonl]
VAL_MANIFESTS=/NeMo/data/pretraining_data/val.jsonl
train_questions=[/NeMo/data/pretraining_data/pretrain_questions.txt]
valid_questions=[/NeMo/data/pretraining_data/pretrain_questions.txt]

# exp_name="llama7B-whisperB/attr"
exp_name="debug"
devices=2

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python \
run_sft_whisper_llama.py --config-path="../examples/multimodel/conf/khlu/" --config-name "whisper_llama_config_7b.yaml" \
    name=$exp_name \
    model.pretrained_audio_model=$ASR_MODEL \
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
    ++model.data.validation_ds.question_file=$valid_questions




