#! /usr/bin/env bash

ROOT=$(dirname $(readlink -f ${BASE_SOURCE[0]}))

alias python=$(command -v python3)
python -m pip install gdown

cd $ROOT

# Training/Validation Set
# https://drive.google.com/file/d/12ycYSzLIG253AFN35Y6qoyf9wtkOjakp/view
gdown "https://drive.google.com/uc?id=12ycYSzLIG253AFN35Y6qoyf9wtkOjakp" -O iwslt17-2017-01-trnmted.tgz
tar xzvf iwslt17-2017-01-trnmted.tgz
rm iwslt17-2017-01-trnmted.tgz
## Extract and preprocess training/validation data
tar xzvf 2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz -C ./.
mv DeEnItNlRo-DeEnItNlRo/ fairseq/examples/cavia/iwslt17-trnmted
cd fairseq/examples/cavia && bash ./prepare-iwslt17.sh

CORES=$(nproc)

# Binarize the de-en dataset
TEXT=fairseq/examples/cavia/iwslt17.de_it_nl_ro.en.bpe16k/
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "${CORES}"

# Binarize the it-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang it --target-lang en \
    --trainpref $TEXT/train.bpe.it-en \
    --validpref $TEXT/valid0.bpe.it-en,$TEXT/valid1.bpe.it-en
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "${CORES}"

# Binarize the nl-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang nl --target-lang en \
    --trainpref $TEXT/train.bpe.nl-en \
    --validpref $TEXT/valid0.bpe.nl-en,$TEXT/valid1.bpe.nl-en
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "${CORES}"

# Binarize the ro-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang ro --target-lang en \
    --trainpref $TEXT/train.bpe.ro-en \
    --validpref $TEXT/valid0.bpe.ro-en,$TEXT/valid1.bpe.ro-en
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "${CORES}"

# Testing Set
# https://drive.google.com/file/d/1hIryKXS4iR9Wv_yiZv9AlgLGnOnKx2_Z/view
# gdown "https://drive.google.com/uc?id=1hIryKXS4iR9Wv_yiZv9AlgLGnOnKx2_Z" -O 2017-01-mted-test.tgz
# tar xzvf 2017-01-mted-test.tgz
# rm 2017-01-mted-test.tgz
