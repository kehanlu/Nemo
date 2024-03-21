NEMO_DIR=/NeMo
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH


PATH_TO_TRAINED_MODEL=/NeMo/data/llama2-7b-chat.nemo
TEST_DS="/NeMo/data/test.jsonl"

python /NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
    model.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=1 \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.test_ds.names=['test'] \
    model.data.test_ds.global_batch_size=2 \
    model.data.test_ds.micro_batch_size=2 \
    model.data.test_ds.tokens_to_generate=20 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    inference.greedy=True \
    model.data.test_ds.output_file_path_prefix=/results/sft_results \
    model.data.test_ds.write_predictions_to_file=True