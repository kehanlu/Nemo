# DeSTA2 Training


## Installation

We highly recommend to use docker image from Nemo to run the code.

In the container, run the following command to install the dependencies:
```
./install_hf.sh
```

## Training

### Data preparation

- manifest.jsonl
- audios/
    - dataset1/
        - audio1.wav
        - audio2.wav
        - ...
    - dataset2/
        - audio1.wav
        - audio2.wav
        - ...

In the manifest, each line is a json object with the following fields:
- `audio_filepath`: the path to the audio file, relative to the "data root" path
- `transcription`: the transcription of the audio file
- `input`: the input prompt for the audio file
- `target`: the target prompt for the audio file
- `duration`: the duration of the audio file

```
{"audio_filepath": "audios/dataset1/audio1.wav", "transcription": "...", "input": "...", "target": "...", duration: 4.0}
{"audio_filepath": "audios/dataset2/audio1.wav", "transcription": "...", "input": "...", "target": "...", duration: 4.0}
```


### Training command

```bash
./run_whisper_llama.sh <dataset_config> <exp_name>

# example
./run_whisper_llama.sh debug.yaml first-run
```

### Evaluation

```bash
./eval_whisper_llama.sh
```