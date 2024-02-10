NEMO_DIR=/NeMo
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

# 220m
# MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
# ALM_CKPT='nemo_experiments/megatron_audio_gpt_peft_tuning/checkpoints/megatron_audio_gpt_peft_tuning--validation_wer\=0.000-step\=165.ckpt'
# ALM_YAML='nemo_experiments/megatron_audio_gpt_peft_tuning/version_0/hparams.yaml'

# MEGATRON_CKPT=/NeMo/data/TinyLlama-chat.nemo
MEGATRON_CKPT=/NeMo/data/llama2-7b-chat.nemo
# ALM_CKPT='nemo_experiments/0128-Tinyllama-whisperB-caption/checkpoints/0128-Tinyllama-whisperB-caption--validation_loss\=2.174-step\=18738-epoch\=2.ckpt'
# ALM_YAML='nemo_experiments/0128-Tinyllama-whisperB-caption/version_1/hparams.yaml'
exp_name='/NeMo/workspace/nemo_experiments/0204-llama7B-whisperM-caption'
ALM_CKPT='/NeMo/workspace/nemo_experiments/0204-llama7B-whisperM-caption/checkpoints/mp_rank_01/0204-llama7B-whisperM-caption--validation_loss\=2.836-step\=737040-epoch\=4.ckpt'
ALM_YAML='/NeMo/workspace/nemo_experiments/0204-llama7B-whisperM-caption/version_1/hparams.yaml'

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/debug_1.json,/media/data/datasets/LibriSpeech/debug_1.json]"
# VAL_NAMES=[debug-1,debug-1]

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean_cleaned.json,/media/data/datasets/LibriSpeech/dev_other.json,/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json]"
# VAL_NAMES="[dev-clean,dev-other,train-clean-100]"

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean_10_q.json]"
# VAL_NAMES="[dev_clean_10]"
# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/test_clean_32.json]"
# VAL_NAMES="[test_clean_32]"

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/test_clean_q.json,/media/data/datasets/LibriSpeech/test_clean_q.json]"
# VAL_NAMES="[test_clean_q,test_clean_q]"

# VAL_MANIFESTS="[/NeMo/data/PromptTTS/manifest.attr.val.jsonl]"
# VAL_MANIFESTS="[/NeMo/data/LTU_iemocap/manifest.attr3.small.jsonl]"
# VAL_MANIFESTS="[/NeMo/data/LTU_iemocap/manifest.attr3.small.jsonl]"
VAL_MANIFESTS="[/NeMo/data/PromptTTS/manifest.attr.nq.val.jsonl]"
# VAL_NAMES="[val10]"
VAL_NAMES="[val_5_0.3Lora_v2]"
valid_questions=/NeMo/data/PromptTTS/manifest.test_question.txt

HYDRA_FULL_ERROR=1 
#python \
python \
eval_whisper_llama.py \
    model.restore_from_path=$MEGATRON_CKPT \
    model.peft.restore_from_path=$ALM_CKPT \
    model.peft.restore_from_hparams_path=$ALM_YAML \
    model.data.test_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.data.test_ds.question_file=$valid_questions \
    model.data.test_ds.names=$VAL_NAMES \
    model.data.test_ds.global_batch_size=8 \
	model.data.test_ds.micro_batch_size=4 \
    model.data.test_ds.output_file_path_prefix=$exp_name/ \
    ++trainer.devices=2 \
    ++inference.greedy=True \
    ++inference.temperature=0.8 \
    model.data.test_ds.tokens_to_generate=256
