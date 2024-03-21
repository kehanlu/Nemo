NEMO_DIR=/NeMo
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0

# MEGATRON_CKPT=/NeMo/data/TinyLlama-v1.nemo
MEGATRON_CKPT=/NeMo/data/TinyLlama-chat.nemo
# MEGATRON_CKPT=/NeMo/data/llama2-7b-chat.nemo
# ASR_MODEL="ssl_en_conformer_large"
ASR_MODEL="stt_en_fastconformer_transducer_large"
GLOBAL_BATCH=8
MICRO_BATCH=4


TRAIN_MANIFESTS=/NeMo/data/PromptTTS/test.jsonl
# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/manifest.attr.jsonl
# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/llama_test.jsonl

VAL_MANIFESTS=/NeMo/data/PromptTTS/test.jsonl
# VAL_MANIFESTS=/NeMo/data/PromptTTS/llama_test.jsonl

exp_name="whisper-llama"
devices=2

CUDA_VISIBLE_DEVICES=0,1 python \
run_sft_audio_lm.py --config-path="../examples/multimodel/conf/speechllm/" --config-name "modularized_speech_gpt_config.yaml" \
    name=$exp_name \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    ++model.data.train_ds.question_file_set=$train_questions \
    ++model.data.train_ds.random_context_prob=0.5 \
    ++model.data.train_ds.random_context_num=64 \
    ++model.data.validation_ds.question_file_set=$valid_questions \
    ++model.data.validation_ds.random_context_prob=0.5 \
    ++model.data.validation_ds.random_context_num=64 \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.tensor_model_parallel_size=2 \
    ++model.pipeline_model_parallel_size=1 \
    ++trainer.num_nodes=1 \
    ++trainer.devices=$devices \




