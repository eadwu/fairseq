from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.trainer import (
    _catalog_shared_params,
    _get_module_by_path,
    _set_module_by_path,
)

import torch
import torch.nn as nn

from .models.cavia_transformer import CAVIATransformerDecoder


@register_task("multilingual_translation_cavia")
class MultilingualTranslationCAVIATask(MultilingualTranslationTask):
    """A task for multilingual translation with CAVIA and BatchEnsemble.
    """
    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        # fmt: off
        # args for Training with BatchEnsemble
        parser.add_argument('--batch-ensemble-root', type=int, default=-1,
                            help='Batch Ensemble root task (0-based) for lifelong learning')
        # args for Meta-Training with CAVIA
        parser.add_argument('--cavia-inner-updates', type=int, default=1,
                            help='Number of inner-loop updates (during training)')
        parser.add_argument('--cavia-lr-inner', type=float, default=1.0,
                            help='Inner-loop learning rate (task-specific)')
        parser.add_argument('--cavia-first-order', default=False, action='store_true',
                            help='Run first-order approximation version')
        parser.add_argument('--cavia-relative-inner-lr', default=False, action='store_true',
                            help='Inner-loop learning rate is relative to optimizer learning rate, with --cavia-lr-inner now being a multiplier')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)

        self.lang_pairs = args.lang_pairs
        if isinstance(self.lang_pairs, str):
            self.lang_pairs = self.lang_pairs.split(",")
        self.eval_lang_pairs = self.lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.n_tasks = len(self.lang_pairs)
        assert self.n_tasks > 0

        if training:
            # Needed for shared TransformerDecoders (r_i, s_i, and b_i)
            assert self.args.share_decoders

            # Validate argument batch_ensemble_root
            assert args.batch_ensemble_root == -1 or (
                args.batch_ensemble_root >= 0 and
                args.batch_ensemble_root < len(self.lang_pairs)
            )

            # Learning rate for context parameters
            self.context_lr = self.args.cavia_lr_inner
            self.relative_lr = getattr(self.args, "cavia_relative_inner_lr", False)

        # Hack for tracking in between functions without copying a bunch more code
        self._base_model = None
        self.context_parameters = None
        self.shared_context_parameters = None

    def _verify_support(self, models):
        # Only one type of supported decoder ...
        assert isinstance(
            self._base_model.decoder, CAVIATransformerDecoder
        )

        for model in models:
            # Memory references should be the same
            assert model.decoder == self._base_model.decoder

    def _get_lang_pair_idx(self, lang_pair):
        # Set language pair index
        lang_pair_idx = [
            i
            for i, lp in enumerate(self.model_lang_pairs)
            if lp == lang_pair
        ]

        assert len(lang_pair_idx) == 1
        lang_pair_idx = lang_pair_idx[0]
        return lang_pair_idx

    # Populate with the initial data needed for context parameters
    def _fetch_context_parameters(self, root_model):
        # Similar to fairseq/trainer.py, the collection of shared parameters
        # but filtered to only contain context parameters
        shared_params = [
            param_subset
            for param_subset in _catalog_shared_params(root_model)
            if "context_param" in param_subset[0]
        ]

        # Similar to fairseq/trainer.py and multilingual_transformer.py, define
        # the reference parameter as the first in the list
        self.context_parameters = [
            param_subset[0] for param_subset in shared_params
        ]

        # The names of the parameters that link to the reference
        self.shared_context_parameters = [
            param_subset[1:] for param_subset in shared_params
        ]

    # Fetches the relevant Tensor references for the context parameters
    def _get_context_parameters(self, root_model, lang_pair_idx):
        filtered_context_parameters = [
            path
            for path in self.context_parameters
            if (
                f"r_{lang_pair_idx}" in path or
                f"s_{lang_pair_idx}" in path or
                f"b_{lang_pair_idx}" in path
            )
        ]

        return filtered_context_parameters, [
            _get_module_by_path(root_model, path)
            for path in filtered_context_parameters
        ]

    # Resets the shared context parameters, which thankfully is just zeroing
    # out the Tensors, except that it needs to be detached from the
    # computational graph ... which means a new Tensor ...
    def _reset_context_parameters(self, root_model, lang_pair_idx):
        filtered_context_parameters, _ = self._get_context_parameters(
            root_model, lang_pair_idx
        )

        for path in filtered_context_parameters:
            ref = _get_module_by_path(root_model, path)
            _set_module_by_path(
                root_model, path,
                nn.Parameter(torch.zeros_like(ref)),
            )

    def _per_lang_pair_train_loss(
        self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        # Reset context parameters on every new task, or in this case
        # language pair
        self._reset_context_parameters(model, lang_pair_idx)

        if self.relative_lr:
            self.context_lr = optimizer.get_lr() * self.args.cavia_lr_inner

        for _ in range(self.args.cavia_inner_updates):
            # Fetch the current references to the Tensors used
            context_names, context_params = self._get_context_parameters(
                model, lang_pair_idx
            )

            # Calculate loss with current parameters
            loss, sample_size, logging_output = criterion(
                model.models[lang_pair], sample[lang_pair]
            )

            if ignore_grad:
                loss *= 0

            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss, context_params,
                create_graph=not self.args.cavia_first_order,
            )

            # Gradient Descent on context parameters
            merged = zip(context_names, context_params, task_gradients)
            for param_name, param, gradient in merged:
                _set_module_by_path(
                    model, param_name,
                    nn.Parameter(param - self.context_lr * gradient),
                )

        # Recompute loss after context parameter update
        loss, sample_size, logging_output = criterion(
            model.models[lang_pair], sample[lang_pair]
        )

        if ignore_grad:
            loss *= 0

        # Default training scheme
        optimizer.backward(loss)

        # Flush context parameters just in case
        self._reset_context_parameters(model, lang_pair_idx)
        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        # Doesn't matter which model, just need to make sure decoders
        # are shared.
        self._base_model = model.models[self.lang_pairs[0]]
        self._verify_support([model.models[m] for m in model.models])

        # Populate Task with context parameter information
        self._fetch_context_parameters(model)

        agg_loss, agg_sample_size, agg_logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

        for param in model.parameters():
            if param.grad is not None:
                param.grad = param.grad / self.n_tasks

        return agg_loss, agg_sample_size, agg_logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        # Reset context parameters on every new task, or in this case
        # language pair
        self._reset_context_parameters(model, lang_pair_idx)

        for _ in range(self.args.cavia_inner_updates):
            # Fetch the current references to the Tensors used
            context_names, context_params = self._get_context_parameters(
                model, lang_pair_idx
            )

            # Gradients aren't enabled during validation ...
            with torch.enable_grad():
                # Calculate loss with current parameters
                loss, _, __ = criterion(
                    model.models[lang_pair], sample[lang_pair]
                )

            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss,
                context_params,
                create_graph=not self.args.cavia_first_order,
            )

            # Gradient Descent on context parameters
            merged = zip(context_names, context_params, task_gradients)
            for param_name, param, gradient in merged:
                # Break from computational graph
                if self.args.cavia_first_order:
                    gradient = gradient.detach()

                _set_module_by_path(
                    model, param_name,
                    nn.Parameter(param - self.context_lr * gradient)
                )

        # Recompute loss after context parameter update
        return criterion(model.models[lang_pair], sample[lang_pair])

    def valid_step(self, sample, model, criterion):
        # Populate Task with context parameter information
        self._fetch_context_parameters(model)

        return super().valid_step(sample, model, criterion)

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        lang_pair = f"{self.args.source_lang}-{self.args.target_lang}"

        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        for model in models:
            model.decoder.set_lang_pair_idx(lang_pair_idx)

        # No context parameter changes, used those saved within the model
        # to perform evaluation
        return super().inference_step(
            generator, models, sample, prefix_tokens, constraints
        )
