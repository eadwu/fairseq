#! /usr/bin/env bash

FAIRSEQ="$1"
shift

EVAL_DIR="$1"
shift

CHECKPOINT="$1"
shift

GPUS="$1"
shift

lang_pairs="de-en,it-en,nl-en,ro-en"
databin_dir="$(readlink -f data-bin/iwslt17.de_it_nl_ro.en.bpe16k)"

mkdir -p "${CHECKPOINT}/bleu"
rm -f "${CHECKPOINT}/bleu.score"

for SRC in "de" "it" "nl" "ro"; do
  LANG_PAIR="${SRC}-en"
  DEST_LANG_PAIR="en-${SRC}"

  cat "${EVAL_DIR}/IWSLT17.TED.tst2017.mltlng.${LANG_PAIR}.${SRC}.xml" | \
    python3 "${FAIRSEQ}/scripts/spm_encode.py" \
      --model="${FAIRSEQ}/examples/cavia/iwslt17.de_it_nl_ro.en.bpe16k/sentencepiece.bpe.model" | \
    CUDA_VISIBLE_DEVICES="${GPUS}" fairseq-interactive "${databin_dir}" \
      --lang-pairs "${lang_pairs}" \
      --source-lang "${SRC}" --target-lang "en" \
      --path "${CHECKPOINT}/checkpoint_best.pt" \
      --remove-bpe=sentencepiece --beam 5 \
      --buffer-size 2000 \
      "$@" > "${CHECKPOINT}/bleu/iwslt17.${SRC}-en.en"

  echo "${LANG_PAIR}" >> "${CHECKPOINT}/bleu.score"
  grep ^H "${CHECKPOINT}/bleu/iwslt17.${SRC}-en.en" | cut -f3 | \
    sacrebleu -tok 'none' -s 'none' \
      "${EVAL_DIR}/IWSLT17.TED.tst2017.mltlng.${DEST_LANG_PAIR}.en.xml" \
    >> "${CHECKPOINT}/bleu.score"
done
