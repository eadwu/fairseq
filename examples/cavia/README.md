# Transformers with CAVIA + BatchEnsemble

## Preparation

Training and validation uses the IWSLT 2017 training and development sets for the multilingual TED task over the translations concerning German-English (de-en), Italian-English (it-en), Dutch-English (nl-en), and Romanian-English (ro-en).

### [Extra] Dependencies
#### Pypi
1. gdown

### Command Line

```bash
# Download the dataset
rm -rf 2017-01-trnmted iwslt17-2017-01-trnmted.tgz
# https://drive.google.com/file/d/12ycYSzLIG253AFN35Y6qoyf9wtkOjakp/view
gdown "https://drive.google.com/uc?id=12ycYSzLIG253AFN35Y6qoyf9wtkOjakp" -O iwslt17-2017-01-trnmted.tgz
tar xzvf iwslt17-2017-01-trnmted.tgz
rm iwslt17-2017-01-trnmted.tgz

FAIRSEQ="$(readlink -f ./.)"

# Extract and prepare the relevant data
rm -rf "${FAIRSEQ}/examples/cavia/iwslt17-trnmted"
tar xzvf 2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz -C ./.
mv DeEnItNlRo-DeEnItNlRo/ "${FAIRSEQ}/examples/cavia/iwslt17-trnmted"
cd "${FAIRSEQ}/examples/cavia" && bash ./prepare-iwslt17.sh
rm -r 2017-01-trnmted
```

If only `--share-decoders` will be used then the following commands are fine.

```bash
# Special joined dictionary hack
## https://github.com/pytorch/fairseq/issues/859
# Strip the first three special tokens and append fake counts for each vocabulary
TEXT="${FAIRSEQ}/examples/cavia/iwslt17.de_it_nl_ro.en.bpe16k"

# Binarize the de-en dataset
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "$(nproc)"

# Binarize the it-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang it --target-lang en \
    --trainpref $TEXT/train.bpe.it-en \
    --validpref $TEXT/valid0.bpe.it-en,$TEXT/valid1.bpe.it-en \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "$(nproc)"

# Binarize the nl-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang nl --target-lang en \
    --trainpref $TEXT/train.bpe.nl-en \
    --validpref $TEXT/valid0.bpe.nl-en,$TEXT/valid1.bpe.nl-en \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "$(nproc)"

# Binarize the ro-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang ro --target-lang en \
    --trainpref $TEXT/train.bpe.ro-en \
    --validpref $TEXT/valid0.bpe.ro-en,$TEXT/valid1.bpe.ro-en \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "$(nproc)"
```

If you intend to use `--share-encoders` then the following commands should be used instead.

```bash
# Special joined dictionary hack
## https://github.com/pytorch/fairseq/issues/859
# Strip the first three special tokens and append fake counts for each vocabulary
TEXT="${FAIRSEQ}/examples/cavia/iwslt17.de_it_nl_ro.en.bpe16k"
tail -n +4 "${TEXT}/sentencepiece.bpe.vocab" | cut -f1 | sed 's/$/ 100/g' > joined.vocab

# Binarize the de-en dataset
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en \
    --srcdict joined.vocab \
    --tgtdict joined.vocab \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "$(nproc)"

# Binarize the it-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang it --target-lang en \
    --trainpref $TEXT/train.bpe.it-en \
    --validpref $TEXT/valid0.bpe.it-en,$TEXT/valid1.bpe.it-en \
    --srcdict joined.vocab \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "$(nproc)"

# Binarize the nl-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang nl --target-lang en \
    --trainpref $TEXT/train.bpe.nl-en \
    --validpref $TEXT/valid0.bpe.nl-en,$TEXT/valid1.bpe.nl-en \
    --srcdict joined.vocab \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "$(nproc)"

# Binarize the ro-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang ro --target-lang en \
    --trainpref $TEXT/train.bpe.ro-en \
    --validpref $TEXT/valid0.bpe.ro-en,$TEXT/valid1.bpe.ro-en \
    --srcdict joined.vocab \
    --tgtdict data-bin/iwslt17.de_it_nl_ro.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_it_nl_ro.en.bpe16k/ \
    --workers "$(nproc)"
```

## Training a multilingual model with CAVIA + BatchEnsemble

<!-- Adaptive training with -->

### Task-specific arguments
1. batch-ensemble-root
    - implements Lifelong Learning outlined within the BatchEnsemble paper
        - In other words, restricts which task (language pair) that would update the shared weight
2. cavia-inner-updates
    - number of times to update context parameters within the meta-learning loop
3. cavia-lr-inner
    - fixed learning rate for the inner training loop
4. cavia-first-order
    - restrict gradients to be first order
5. cavia-relative-inner-lr
    - `cavia-lr-inner` becomes a multiplier to the current epoch's learning rate (relative to `optimizer.get_lr()`)

### Command Line

Uses the included `scripts/train.sh`

```bash
./train.sh \
  checkpoints/cavia_LL_4_8 \
  "2,3" \
  --user-dir examples/cavia/src \
  --task multilingual_translation_cavia \
  --batch-ensemble-root 0 --cavia-inner-updates 4 --cavia-lr-inner 8.0 \
  --arch cavia_multilingual_transformer \
  --share-decoders --share-decoder-input-output-embed \
  --memory-efficient-fp16 --tensorboard-logdir checkpoints/cavia_LL_4_8 &
```

## Evaluation

Evaluation uses the IWSLT 2017 Evaluation Campaign for the multilingual TED Talks MT task concerning German-English (de-en), Italian-English (it-en), Dutch-English (nl-en), and Romanian-English (ro-en).

### Preparation

```bash
# Download the data
rm -rf 2017-01-mted-test 2017-01-mted-test_orig 2017-01-mted-test.tgz
# https://drive.google.com/file/d/1hIryKXS4iR9Wv_yiZv9AlgLGnOnKx2_Z/view
gdown "https://drive.google.com/uc?id=1hIryKXS4iR9Wv_yiZv9AlgLGnOnKx2_Z" -O 2017-01-mted-test.tgz
tar xzvf 2017-01-mted-test.tgz
rm 2017-01-mted-test.tgz

mv 2017-01-mted-test 2017-01-mted-test_orig
ROOT="$(readlink -f 2017-01-mted-test_orig)"
DEST="$(readlink -f 2017-01-mted-test)"

cd "${ROOT}"
mkdir -p "${DEST}"

find "${ROOT}/texts" -name '.*.tgz' -exec rm "{}" \;
find "${ROOT}/texts" -name '*.tgz' -exec tar xzvf "{}" \;

find "${ROOT}" -name '*.xml' -exec mv "{}" "${DEST}" \;

cd "${DEST}"
rm -r "${ROOT}"
```

### Command Line

Uses the included `scripts/eval.sh`

```bash
./eval.sh \
  ./fairseq \
  ./2017-01-mted-test \
  checkpoints/cavia_LL_4_8 \
  "1" \
  --user-dir fairseq/examples/cavia/src --task multilingual_translation_cavia \
  --batch-size 64 --memory-efficient-fp16
```

## Comparison

### Baseline

#### Training

Uses the included `scripts/train.sh`

```bash
./train.sh \
  checkpoints/baseline \
  "0,1" \
  --task multilingual_translation \
  --arch multilingual_transformer_iwslt_de_en \
  --share-decoders --share-decoder-input-output-embed \
  --memory-efficient-fp16 --tensorboard-logdir checkpoints/baseline
```

#### Evaluation

Uses the included `scripts/eval.sh`

```bash
./eval.sh \
  ./fairseq \
  ./2017-01-mted-test \
  checkpoints/baseline \
  "0" \
  --task multilingual_translation \
  --batch-size 64 --memory-efficient-fp16
```
