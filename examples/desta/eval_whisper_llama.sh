#!/bin/bash
NEMO_DIR=/NeMo
EXP_DIR=$NEMO_DIR/workspace/nemo_experiments
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH
export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export HF_DATASETS_CACHE="/NeMo/.cache"

exp_dir="/NeMo/workspace/nemo_experiments/MISTA/240810-17@project_kira+0811_whatyouhear.yaml@10e@"
config_file="/NeMo/workspace/nemo_experiments/MISTA/240810-17@project_kira+0811_whatyouhear_MixedConcat.yaml@10e@/config_eval_zh.yaml"

epoch=9

###################
###################

dataset_name="dynamic-superb-test"

if [ "$dataset_name" = "dynamic-superb-test" ]; then
    manifest_filepaths="/NeMo/data/dynamic-superb/240811_dynamic-superb-test.jsonl"
    data_root="/NeMo/data/audios/dynamic-superb-test"
fi




echo exp_dir: $exp_dir
echo epoch: $epoch
echo manifest_filepaths: $manifest_filepaths
echo config_file: $config_file


CUDA_VISIBLE_DEVICES=0 python $NEMO_DIR/examples/desta/eval_speech_llama.py \
    --exp_dir $exp_dir \
    --epoch $epoch \
    --dataset_name $dataset_name \
    --manifest_filepaths $manifest_filepaths \
    --data_root $data_root \
    --config_file $config_file

exit 0;
