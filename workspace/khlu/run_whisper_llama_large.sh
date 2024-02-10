NEMO_DIR=/NeMo
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
GLOBAL_BATCH=2
MICRO_BATCH=2

# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/test.jsonl
# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/manifest.nq.caption.jsonl
# TRAIN_MANIFESTS=/NeMo/data/PromptTTS/llama_test.jsonl

# VAL_MANIFESTS=/NeMo/data/PromptTTS/test.jsonl
TRAIN_MANIFESTS=[/NeMo/data/PromptTTS/PromptTTS.nq.caption2.jsonl,/NeMo/data/LibriTTS/LibriTTS.nq.caption2.jsonl,/NeMo/data/IEMOCAP/IEMOCAP.nq.caption2.jsonl]
VAL_MANIFESTS=/NeMo/data/PromptTTS/manifest.attr.nq.val.jsonl
train_questions=[/NeMo/data/PromptTTS/manifest.caption.question.txt,/NeMo/data/PromptTTS/manifest.caption.question.txt,/NeMo/data/PromptTTS/manifest.caption.question.txt]
valid_questions=[/NeMo/data/PromptTTS/manifest.caption.question.txt,/NeMo/data/PromptTTS/manifest.caption.question.txt,/NeMo/data/PromptTTS/manifest.caption.question.txt]

# exp_name="llama7B-whisperB/attr"
exp_name="0209-llama7B-whisperL-caption"
devices=2
perception_mode="qformer_1"

restore_from_path=/NeMo/workspace/nemo_experiments/0209-llama7B-whisperL-caption/checkpoints/mp_rank_00/0209-llama7B-whisperL-caption--validation_loss\=2.249-step\=264525-epoch\=0.ckpt
restore_from_hparams_path=/NeMo/workspace/nemo_experiments/0209-llama7B-whisperL-caption/version_3/hparams.yaml

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python \
run_sft_whisper_llama.py --config-path="../examples/multimodel/conf/khlu/" --config-name "whisper_llama_config_7b.yaml" \
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
    ++model.tensor_model_parallel_size=2 \
    ++model.pipeline_model_parallel_size=1 \
    ++trainer.num_nodes=1 \
    ++trainer.val_check_interval=0.2 \
    ++trainer.devices=$devices \
    ++model.perception.mode=$perception_mode




