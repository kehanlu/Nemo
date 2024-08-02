git clone https://github.com/huggingface/transformers
cd transformers
git checkout v4.41.1
pip install -e .
cd ..

git clone https://github.com/huggingface/peft
cd peft
pip install -e .
cd ..

pip install lhotse whisper_normalizer

pip install -U huggingface_hub==0.23.5