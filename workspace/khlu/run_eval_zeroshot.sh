NEMO_DIR=/NeMo
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

# 220m

MEGATRON_CKPT=/NeMo/data/llama2-7b-chat.nemo


epoch=7
exp_name='/NeMo/workspace/nemo_experiments/pretrain-base-llama/240318@cnn_1@llama7b-base_whisper-medium@12_6'
ALM_CKPT='/NeMo/workspace/nemo_experiments/pretrain-base-llama/240318@cnn_1@llama7b-base_whisper-medium@12_6/checkpoints/pretrain-base-llama/mp_rank_00/240318@cnn_1@llama7b-base_whisper-medium@12_6--validation_loss\=1.980-step\=97504-epoch\=7.ckpt'
ALM_YAML='/NeMo/workspace/nemo_experiments/pretrain-base-llama/240318@cnn_1@llama7b-base_whisper-medium@12_6/version_1/hparams_eval.yaml'


TEST_SET="test"
if [ "$TEST_SET" = "test" ]; then
    VAL_MANIFESTS="[/NeMo/data/promptTTS_test/0310_manifest_sysprompt2.jsonl]"
elif [ "$TEST_SET" = "hf200@SV" ]; then
    VAL_MANIFESTS="[/NeMo/data/dynamic-superb-test-new-hf/audio/0227-DS-HF.test.jsonl]"
elif [ "$TEST_SET" = "DS200@woAudio" ]; then
    VAL_MANIFESTS="[/NeMo/data/dynamic-superb-test-subset/DS-test.200.noAudio.jsonl]"
elif [ "$TEST_SET" = "hf200@noAudio" ]; then
    VAL_MANIFESTS=[/NeMo/data/dynamic-superb-test-new-hf/0305-DS-HF.test.NoAudio.jsonl]
fi


VAL_NAMES=ep$epoch@$TEST_SET@chat@lora05
valid_questions=null

mkdir -p $exp_name/zeroshot_results

#python \
HYDRA_FULL_ERROR=1 python \
eval_whisper_llama.py \
    ++model.restore_from_path=$MEGATRON_CKPT \
    model.peft.restore_from_path=$ALM_CKPT \
    model.peft.restore_from_hparams_path=$ALM_YAML \
    model.data.test_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.data.test_ds.random_transcription_prob=0.0 \
    model.data.test_ds.names=$VAL_NAMES \
    model.data.test_ds.global_batch_size=16 \
	model.data.test_ds.micro_batch_size=8 \
    model.data.test_ds.output_file_path_prefix=$exp_name/zeroshot_results/ \
    ++model.tensor_model_parallel_size=2 \
    ++trainer.devices=4 \
    ++inference.greedy=False \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    ++model.pretrained_audio_model=openai/whisper-medium \