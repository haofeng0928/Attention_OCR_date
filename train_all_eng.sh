#!/usr/bin/env bash

python src/launcher.py \
	--phase=train \
	--data-path=/home/xiaguomiao/data/ai-hack/crop_words/all/train.txt \
	--data-root-dir=/home/xiaguomiao/data/ai-hack/crop_words/all/images \
    --lexicon-file=/home/xiaguomiao/data/ai-hack/eng_lexicon.txt \
	--log-path=log.txt \
	--attn-num-hidden=256 \
	--batch-size=256 \
    --channel=1 \
	--mean=[128] \
	--model-dir=models \
	--initial-learning-rate=1.0 \
	--load-model \
	--num-epoch=20 \
	--gpu-id=1 \
	--use-gru \
	--steps-per-checkpoint=400 \
    --target-embedding-size=10  \
    --target-vocab-size=48