#! /usr/bin/env bash

# Armenian - hye | hy
# Latvian - lav | lv
# Nepali - npi | ne

ROOT=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
cd $ROOT

BIBLE_TGT=( "hy" "lv" "ne" )
FLORES_TGT=( "hye" "lav" "npi" )

# Data directory
DATA="${ROOT}/data.hy_lv_ne.bpe16k"
rm -rf "${DATA}"
mkdir "${DATA}"
cd "${DATA}"

## Training Data (bible-uedin)
for TGT in "${BIBLE_TGT[@]}"; do
  rm -rf "en-${TGT}.txt.zip"

  # https://opus.nlpl.eu/download.php?f=bible-uedin/v1/moses/en-${TGT}.txt.zip
  wget "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/en-${TGT}.txt.zip" \
    -O "en-${TGT}.txt.zip"

  mkdir -p "${TGT}/en"
  unzip -d "${TGT}/en" "en-${TGT}.txt.zip"
  rm "en-${TGT}.txt.zip"
done

## Validation / Testing Data (FLORES-101)
rm -rf flores101_dataset.tar.gz flores101_dataset
wget "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz" \
  flores101_dataset.tar.gz
tar xzvf flores101_dataset.tar.gz
rm flores101_dataset.tar.gz

## Preprocess training / validation data
mv "flores101_dataset/devtest/eng.devtest" ./.
for ((i=0;i<${#BIBLE_TGT[@]};++i)); do
  TGT="${BIBLE_TGT[i]}"
  F_TGT="${FLORES_TGT[i]}"

  # echo "$TGT" "$F_TGT" "$i"

  mv "${TGT}/en/bible-uedin.en-${TGT}.en" "train.${TGT}-en.en"
  mv "${TGT}/en/bible-uedin.en-${TGT}.${TGT}" "train.${TGT}-en.${TGT}"

  rm -r "${TGT}"

  cp "flores101_dataset/dev/eng.dev" "valid0.${TGT}-en.en"
  mv "flores101_dataset/dev/${F_TGT}.dev" "valid0.${TGT}-en.${TGT}"
  mv "flores101_dataset/devtest/${F_TGT}.devtest" ./.
done
rm -r flores101_dataset

cd "${ROOT}"
rm -rf fairseq/examples/batchensemble/data.hy_lv_ne.en.bpe16k
mv "${DATA}" fairseq/examples/batchensemble/data.hy_lv_ne.en.bpe16k

cd fairseq/examples/batchensemble && bash ./prepare-data.sh

cd "${ROOT}"

# Binarize training/validation data
CORES=$(nproc)

# Special joined dictionary hack
## https://github.com/pytorch/fairseq/issues/859
# Strip the first three special tokens and append fake counts for each vocabulary
TEXT=fairseq/examples/batchensemble/data.hy_lv_ne.en.bpe16k

rm -rf joined.vocab
tail -n +4 "${TEXT}/sentencepiece.bpe.vocab" | cut -f1 | \
  sed 's/$/ 100/g' > joined.vocab

rm -rf data-bin

# Binarize the hy-en dataset
fairseq-preprocess --source-lang hy --target-lang en \
    --trainpref "${TEXT}/train.bpe.hy-en" \
    --validpref "${TEXT}/valid0.bpe.hy-en" \
    --srcdict joined.vocab \
    --tgtdict joined.vocab \
    --destdir data-bin/data.hy_lv_ne.bpe16k/ \
    --workers "${CORES}"

# Binarize the lv-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang lv --target-lang en \
    --trainpref "${TEXT}/train.bpe.lv-en" \
    --validpref "${TEXT}/valid0.bpe.lv-en" \
    --srcdict joined.vocab \
    --tgtdict data-bin/data.hy_lv_ne.bpe16k/dict.en.txt \
    --destdir data-bin/data.hy_lv_ne.bpe16k/ \
    --workers "${CORES}"

# Binarize the ne-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang ne --target-lang en \
    --trainpref "${TEXT}/train.bpe.ne-en" \
    --validpref "${TEXT}/valid0.bpe.ne-en" \
    --srcdict joined.vocab \
    --tgtdict data-bin/data.hy_lv_ne.bpe16k/dict.en.txt \
    --destdir data-bin/data.hy_lv_ne.bpe16k/ \
    --workers "${CORES}"
