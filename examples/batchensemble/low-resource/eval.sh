#! /usr/bin/env bash

FAIRSEQ="$1"
shift

EVAL_DIR="$1"
shift

CHECKPOINT="$1"
shift

GPUS="$1"
shift

lang_pairs="hy-en,lv-en,ne-en"
databin_dir="$(readlink -f data-bin/data.hy_lv_ne.bpe16k/)"

mkdir -p "${CHECKPOINT}/bleu"
rm -f "${CHECKPOINT}/bleu.score"


BIBLE_TGT=( "hy" "lv" "ne" )
FLORES_TGT=( "hye" "lav" "npi" )

for ((i=0;i<${#BIBLE_TGT[@]};++i)); do
  SRC="${BIBLE_TGT[i]}"
  F_SRC="${FLORES_TGT[i]}"

  LANG_PAIR="${SRC}-en"
  DEST_LANG_PAIR="en-${SRC}"

  cat "${EVAL_DIR}/${F_SRC}.devtest" | \
    python3 "${FAIRSEQ}/scripts/spm_encode.py" \
      --model="${FAIRSEQ}/examples/batchensemble/data.hy_lv_ne.en.bpe16k/sentencepiece.bpe.model" | \
    CUDA_VISIBLE_DEVICES="${GPUS}" fairseq-interactive "${databin_dir}" \
      --lang-pairs "${lang_pairs}" \
      --source-lang "${SRC}" --target-lang "en" \
      --path "${CHECKPOINT}/checkpoint_best.pt" \
      --remove-bpe=sentencepiece --beam 5 \
      --buffer-size 2000 \
      "$@" > "${CHECKPOINT}/bleu/flores101.${LANG_PAIR}.en"

  echo "${LANG_PAIR}" >> "${CHECKPOINT}/bleu.score"
  grep ^H "${CHECKPOINT}/bleu/flores101.${LANG_PAIR}.en" | cut -f3 | \
    sacrebleu -tok 'none' -s 'none' \
      "${EVAL_DIR}/eng.devtest" \
    >> "${CHECKPOINT}/bleu.score"
done
