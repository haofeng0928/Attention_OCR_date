#!/usr/bin/env bash

python src/launcher.py \
	--phase=train \
	--data-path=D:/myData/COCO-Text-words-trainval/train.txt \
	--data-root-dir=D:/myData/COCO-Text-words-trainval/train_words \
    --lexicon-file=D:/myData/COCO-Text-words-trainval/lexicon.txt \
	--log-path=log.txt \
	--attn-num-hidden=256 \
	--batch-size=256 \
    --channel=1 \
	--mean=[128] \
	--model-dir=models \
	--initial-learning-rate=1.0 \
	--load-model \
	--num-epoch=300 \
	--gpu-id=1 \
	--use-gru \
	--steps-per-checkpoint=200 \
    --target-embedding-size=10  \
    --target-vocab-size=101 # default 48
