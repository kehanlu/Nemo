import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from datasets import load_dataset, Dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
)
model.to(device)
processor = WhisperProcessor.from_pretrained(model_id)



pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")



BATCH_SIZE = 8

audio_paths = [{"path":str(p)} for p in sorted((Path("/NeMo/data/IEMOCAP_LTU").glob("**/*.wav")))]

dataset = Dataset.from_list(audio_paths)

def iterate_data(dataset):
    for i, item in enumerate(dataset):
        yield item["path"]


predictions = []
fo = open("result.txt", "w")
for path, out in tqdm(zip(iterate_data(dataset), pipe(iterate_data(dataset), batch_size=BATCH_SIZE))):
    fo.write(json.dumps({"path": str(path), "text": out["text"]}) + "\n")
fo.close()