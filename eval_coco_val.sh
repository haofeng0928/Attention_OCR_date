#!/usr/bin/env bash

python src/launcher.py \
    --phase=test \
    --data-path=/home/xiaguomiao/data/ai-hack/crop_words/coco-text/val.txt \
    --data-root-dir=/home/xiaguomiao/data/ai-hack/crop_words/coco-text/val_words \
    --log-path=log.txt \
    --load-model \
    --mean=[128] \
    --channel=1 \
    --attn-num-hidden=256 \
    --attn-num-layers=2 \
    --batch-size=256 \
    --gpu-id=1 \
    --use-gru \
    --model-dir=models \
    --output-dir=results \
    --lexicon-file=/home/xiaguomiao/data/ai-hack/eng_lexicon.txt \
    --target-embedding-size=10 \
    --target-vocab-size=49
