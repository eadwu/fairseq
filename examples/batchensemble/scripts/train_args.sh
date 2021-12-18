#! /usr/bin/env bash

GPUS="$1"
shift

args=(
  "checkpoints-randn/batchensemble_language_specific-1.6_LL;--user-dir;examples/batchensemble/src;--task;multilingual_translation_be;--arch;batch_ensemble_multilingual_transformer;--batchensemble-lr-multiplier;1.6;--batchensemble-lifelong-learning"
  "checkpoints-randn/batchensemble_language_specific-1.8_LL;--user-dir;examples/batchensemble/src;--task;multilingual_translation_be;--arch;batch_ensemble_multilingual_transformer;--batchensemble-lr-multiplier;1.8;--batchensemble-lifelong-learning"
  "checkpoints-randn/batchensemble_language_specific-2.0_LL;--user-dir;examples/batchensemble/src;--task;multilingual_translation_be;--arch;batch_ensemble_multilingual_transformer;--batchensemble-lr-multiplier;2.0;--batchensemble-lifelong-learning"
  "checkpoints-randn/batchensemble_language_specific-3.0_LL;--user-dir;examples/batchensemble/src;--task;multilingual_translation_be;--arch;batch_ensemble_multilingual_transformer;--batchensemble-lr-multiplier;3.0;--batchensemble-lifelong-learning"
  "checkpoints-randn/batchensemble_language_specific-4.0_LL;--user-dir;examples/batchensemble/src;--task;multilingual_translation_be;--arch;batch_ensemble_multilingual_transformer;--batchensemble-lr-multiplier;4.0;--batchensemble-lifelong-learning"
)

for config in "${args[@]}"
do
  # BASH-specific to array V
  IFS=";" read -r -a arr <<< "${config}"

  ./train.sh "${GPUS}" "${arr[@]}" "${@}"
done
