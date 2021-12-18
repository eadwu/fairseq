#! /usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SRCS=(
    "hy"
    "lv"
    "ne"
)
TGT=en

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/../../scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

BPESIZE=16384
DATA=$ROOT/data.hy_lv_ne.en.bpe16k

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

URLS=(
  # UNUSED
)
ARCHIVES=(
  # UNUSED
)
VALID_SETS=(
    "hye.dev"
    "lav.dev"
    "npi.dev"
)

# learn BPE with sentencepiece
TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $DATA/train.${SRC}-${TGT}.${SRC}; echo $DATA/train.${SRC}-${TGT}.${TGT}; done | tr "\n" ",")
echo "learning joint BPE over ${TRAIN_FILES}..."
python3 "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$DATA/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# encode train/valid
echo "encoding train with learned BPE..."
for SRC in "${SRCS[@]}"; do
    python3 "$SPM_ENCODE" \
        --model "$DATA/sentencepiece.bpe.model" \
        --output_format=piece \
        --inputs $DATA/train.${SRC}-${TGT}.${SRC} $DATA/train.${SRC}-${TGT}.${TGT} \
        --outputs $DATA/train.bpe.${SRC}-${TGT}.${SRC} $DATA/train.bpe.${SRC}-${TGT}.${TGT} \
        --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
done

echo "encoding valid with learned BPE..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    VALID_SET=(${VALID_SETS[i]})
    for ((j=0;j<${#VALID_SET[@]};++j)); do
        python3 "$SPM_ENCODE" \
            --model "$DATA/sentencepiece.bpe.model" \
            --output_format=piece \
            --inputs $DATA/valid${j}.${SRC}-${TGT}.${SRC} $DATA/valid${j}.${SRC}-${TGT}.${TGT} \
            --outputs $DATA/valid${j}.bpe.${SRC}-${TGT}.${SRC} $DATA/valid${j}.bpe.${SRC}-${TGT}.${TGT}
    done
done
