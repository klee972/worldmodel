#!/bin/bash

# uv run python jasmine/baselines/diffusion/train_tokenizer_mae.py \
#     --data_dir data/data/coinrun_episodes/train \
#     --val_data_dir data/data/coinrun_episodes/val

uv run python jasmine/dreamer4/train_tokenizer_coinrun.py

