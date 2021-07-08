#! /usr/bin/env bash

# CAVIA-specific arguments
#   --batch-ensemble-root 0
#   --cavia-inner-updates 4
#   --cavia-lr-inner 8.0
#   --cavia-first-order

# Example usage
# train.sh \
#   checkpoints/baseline \
#   "0,1" \
#   --task multilingual_translation \
#   --arch multilingual_transformer_iwslt_de_en \
#   --share-decoders --share-decoder-input-output-embed \
#   --memory-efficient-fp16 &

./train.sh \
  checkpoints/cavia_LL_4_8 \
  "0,1" \
  --user-dir examples/cavia/src \
  --task multilingual_translation_cavia \
  --batch-ensemble-root 0 --cavia-inner-updates 4 --cavia-lr-inner 8.0 \
  --arch cavia_multilingual_transformer \
  --share-decoders --share-decoder-input-output-embed \
  --memory-efficient-fp16 &

CHECKPOINT="$1"
shift

GPUS="$1"
shift

rm -rf "${CHECKPOINT}"
mkdir -p "${CHECKPOINT}"

lang_pairs="de-en,it-en,nl-en,ro-en"
databin_dir="$(readlink -f data-bin/iwslt17.de_it_nl_ro.en.bpe16k)"

CUDA_VISIBLE_DEVICES="${GPUS}" fairseq-train "${databin_dir}" \
  --max-epoch 50 \
  --ddp-backend=legacy_ddp \
  --lang-pairs "${lang_pairs}" \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.0005 --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
  --dropout 0.3 --weight-decay 0.0001 \
  --save-dir "${CHECKPOINT}" \
  --max-tokens 4000 \
  --update-freq 8 \
  --combine-val \
  --no-progress-bar --no-epoch-checkpoints --keep-interval-updates 0 \
  --log-interval 1 --log-file "${CHECKPOINT}/debug.log" \
  "$@" &> "${CHECKPOINT}/stdout.log" &
