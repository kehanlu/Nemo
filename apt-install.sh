#!/bin/bash
apt update
apt install -y tmux htop zsh

# Install transformers 4.36.1
git clone https://github.com/huggingface/transformers.git transformers
cd transformers
git checkout v4.36.1
pip install -e .
