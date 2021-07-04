#! /usr/bin/env bash

ROOT=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

python3 -m pip install --user gdown

cd $ROOT

# Training/Validation Set
# https://drive.google.com/file/d/12ycYSzLIG253AFN35Y6qoyf9wtkOjakp/view
rm -rf 2017-01-trnmted iwslt17-2017-01-trnmted.tgz
gdown "https://drive.google.com/uc?id=12ycYSzLIG253AFN35Y6qoyf9wtkOjakp" -O iwslt17-2017-01-trnmted.tgz
tar xzvf iwslt17-2017-01-trnmted.tgz
rm iwslt17-2017-01-trnmted.tgz
## Extract and preprocess training/validation data
rm -rf fairseq/examples/cavia/iwslt17-trnmted
tar xzvf 2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz -C ./.
mv DeEnItNlRo-DeEnItNlRo/ fairseq/examples/cavia/iwslt17-trnmted
cd fairseq/examples/cavia && bash ./prepare-iwslt17.sh

# Binarize training/validation data
CORES=$(nproc)

# Special joined dictionary hack
## https://github.com/pytorch/fairseq/issues/859
# Strip the first three special tokens and append fake counts for each vocabulary
TEXT=fairseq/examples/cavia/iwslt17.de_it_nl_ro.en.bpe16k
tail -n +4 $TEXT/sentencepiece.bpe.vocab | cut -f1 | sed 's/$/ 100/g' > joined.vocab

# Binarize the de-en dataset
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en \
    --srcdict joined.vocab \
    --tgtdict joined.vocab \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "${CORES}"

# Binarize the it-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang it --target-lang en \
    --trainpref $TEXT/train.bpe.it-en \
    --validpref $TEXT/valid0.bpe.it-en,$TEXT/valid1.bpe.it-en \
    --srcdict joined.vocab \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "${CORES}"

# Binarize the nl-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang nl --target-lang en \
    --trainpref $TEXT/train.bpe.nl-en \
    --validpref $TEXT/valid0.bpe.nl-en,$TEXT/valid1.bpe.nl-en \
    --srcdict joined.vocab \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "${CORES}"

# Binarize the ro-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang ro --target-lang en \
    --trainpref $TEXT/train.bpe.ro-en \
    --validpref $TEXT/valid0.bpe.ro-en,$TEXT/valid1.bpe.ro-en \
    --srcdict joined.vocab \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "${CORES}"

# Testing Set
# https://drive.google.com/file/d/1hIryKXS4iR9Wv_yiZv9AlgLGnOnKx2_Z/view
# gdown "https://drive.google.com/uc?id=1hIryKXS4iR9Wv_yiZv9AlgLGnOnKx2_Z" -O 2017-01-mted-test.tgz
# tar xzvf 2017-01-mted-test.tgz
# rm 2017-01-mted-test.tgz
