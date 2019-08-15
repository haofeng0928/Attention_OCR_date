#!/usr/bin/env bash

python src/launcher.py \
	--phase=train \
	--data-path=data/coco-text/new_train.txt \
	--data-root-dir=data/coco-text/train_words \
    --lexicon-file=data/lexicon.txt \
	--log-path=log.txt \
	--attn-num-hidden=256 \
	--batch-size=256 \
    --channel=1 \
	--mean=[128] \
	--model-dir=model \
	--initial-learning-rate=1.0 \
	--load-model \
	--num-epoch=30 \
	--gpu-id=1 \
	--use-gru \
	--steps-per-checkpoint=200 \
    --target-embedding-size=10  \
    --target-vocab-size=80
