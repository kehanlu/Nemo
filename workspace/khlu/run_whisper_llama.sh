NEMO_DIR=/NeMo
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0

MEGATRON_CKPT=/NeMo/data/TinyLlama-chat.nemo
# MEGATRON_CKPT=/NeMo/data/llama2-7b-chat.nemo
# ASR_MODEL="ssl_en_conformer_large"
# ASR_MODEL="stt_en_fastconformer_transducer_large"
ASR_MODEL="openai/whisper-medium" # huggingface id
GLOBAL_BATCH=16
MICRO_BATCH=8

# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/test.jsonl
# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/manifest.attr.jsonl
# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/manifest.attr.small.jsonl
# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/llama_test.jsonl
TRAIN_MANIFESTS=["/NeMo/data/PromptTTS/manifest.attr2.jsonl"]
train_questions=["/NeMo/data/PromptTTS/manifest.attr.question.txt"]
valid_questions=["/NeMo/data/PromptTTS/manifest.attr.question.txt"]

# VAL_MANIFESTS=/NeMo/data/PromptTTS/test.jsonl
VAL_MANIFESTS=/NeMo/data/PromptTTS/manifest.attr2.small.jsonl

# exp_name="llama7B-whisperB/attr"
# exp_name="7bllama"
exp_name="promptts/0202-3"
devices=2
# perception_mode="prompt_1"
perception_mode="qformer_1"

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python \
run_sft_whisper_llama.py --config-path="../examples/multimodel/conf/khlu/" --config-name "whisper_llama_config.yaml" \
    name=$exp_name \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    ++model.data.train_ds.question_file=$train_questions \
    ++model.data.train_ds.random_context_prob=0 \
    ++model.data.train_ds.random_context_num=0 \
    ++model.data.validation_ds.question_file=$valid_questions \
    ++model.data.validation_ds.random_context_prob=0 \
    ++model.data.validation_ds.random_context_num=0 \
    model.data.validation_ds.output_file_path_prefix=$exp_name/ \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.tensor_model_parallel_size=1 \
    ++model.pipeline_model_parallel_size=1 \
    ++trainer.num_nodes=1 \
    ++trainer.devices=$devices \
    ++model.perception.mode=$perception_mode





