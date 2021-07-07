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
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)

        self.lang_pairs = args.lang_pairs
        self.eval_lang_pairs = self.lang_pairs
        self.model_lang_pairs = self.lang_pairs
        assert len(self.lang_pairs) > 0
        self.n_tasks = float(len(self.lang_pairs))

        if training:
            # Needed to satisfy an assumption for shared gradient accumulation
            assert self.args.share_encoders
            # Needed for shared r_i, s_i, and b_i, through shared TransformerDecoder(s)
            assert self.args.share_decoders

            # Validate argument batch_ensemble_root
            assert args.batch_ensemble_root == -1 or (
                args.batch_ensemble_root >= 0 and
                args.batch_ensemble_root < len(self.lang_pairs)
            )

            # Learning rate for context parameters
            self.context_lr = self.args.cavia_lr_inner

        # Hack for tracking in between functions without copying a bunch more code
        self.meta_gradient = None
        self.shared_parameters = None
        self.context_parameters = None
        self.shared_context_parameters = None

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

    # This is confusing, anyway the main rationale for reimplementing this here
    # is because I'm not entirely sure on how shared parameters work.  From
    # what I gathered, Tensors are shared by reference, but individual
    # references are by ???.  Shared decoders are indexed by the same reference
    # when they are initialized but Fairseq explicitly links the shared
    # parameters again using the module path?  So due to the confusing nature
    # of the implemention, since for multi-order gradients the previous leaf
    # Tensors need to be detached but also the next run need to be able to
    # reference the updated context parameters, the implementation sticks with
    # the module path route for all context parameters accesses and updates,
    # thus the need for manually synchronizing shared context parameters across
    # updates.

    # This isn't a problem with the shared model parameters, since it uses an
    # optimizer with single-order gradients, such that it is a supported
    # "in-place" operation with PyTorch's optimizers.

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
            if f"r_{lang_pair_idx}" in path or
                f"s_{lang_pair_idx}" in path or
                f"b_{lang_pair_idx}" in path
        ]

        return filtered_context_parameters, [
            _get_module_by_path(root_model, path)
            for path in filtered_context_parameters
        ]

    # Resyncs the shared context parameters, or otherwise said, the Tensor
    # references
    def _sync_shared_context_references(self, root_model):
        for i, linked_params in enumerate(self.shared_context_parameters):
            root_param = self.context_parameters[i]
            ref = _get_module_by_path(root_model, root_param)
            for param in linked_params:
                _set_module_by_path(root_model, param, ref)

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

        self._sync_shared_context_references(root_model)

    def _per_lang_pair_train_loss(
        self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        # Only one type of supported model
        assert isinstance(
            model.models[lang_pair].decoder, CAVIATransformerDecoder
        )

        def run_model():
            loss, sample_size, logging_output = criterion(
                model.models[lang_pair], sample[lang_pair]
            )

            if ignore_grad:
                loss *= 0

            return loss, sample_size, logging_output

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

            # Calculate loss with current parameters
            loss, _, __ = run_model()

            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss,
                context_params,
                create_graph=not self.args.cavia_first_order,
            )

            # Gradient Descent on context parameters
            for i, _ in enumerate(task_gradients):
                context_parameter_name = context_names[i]
                _set_module_by_path(
                    model, context_parameter_name,
                    nn.Parameter(
                        context_params[i] - self.context_lr * task_gradients[i]
                    ),
                )

            # Synchronize updates across shared context parameters
            self._sync_shared_context_references(model)

        # Recompute loss after context parameter update
        loss, sample_size, logging_output = run_model()

        # Filter out Tensors that don't need gradients for lifelong learning
        shared_parameters = {
            name: param
            for name, param in self.shared_parameters.items()
            if param.requires_grad
        }
        shared_parameters_n = [name for name in shared_parameters.keys()]
        shared_parameters_p = [shared_parameters[n] for n in shared_parameters_n]

        # Compute task_gradients with respect to the shared [model] parameters
        task_gradients = torch.autograd.grad(loss, shared_parameters_p)

        # Update meta-gradient
        for i in range(len(task_gradients)):
            param_n = shared_parameters_n[i]
            self.meta_gradient[param_n] += task_gradients[i].detach()

        # Flush context parameters just in case
        self._reset_context_parameters(model, lang_pair_idx)
        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        base_model = model.models[self.model_lang_pairs[0]]
        self.shared_parameters = {
            name: parameters
            for name, parameters in base_model.named_parameters()
            if "context_param" not in name
        }

        self.meta_gradient = {name: 0 for name in self.shared_parameters}

        # Populate Task with context parameter information
        self._fetch_context_parameters(model)

        agg_loss, agg_sample_size, agg_logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

        # Apply meta-gradient to shared parameters
        optimizer.zero_grad()
        for name, param in self.shared_parameters.items():
            param.grad = self.meta_gradient[name] / self.n_tasks
            param.grad.data.clamp_(-10, 10)
        optimizer.step()

        return agg_loss, agg_sample_size, agg_logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        # Only one type of supported model
        assert isinstance(
            model.models[lang_pair].decoder, CAVIATransformerDecoder
        )

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
            for i, _ in enumerate(task_gradients):
                context_parameter_name = context_names[i]

                gradient = task_gradients[i]
                if self.args.cavia_first_order: # break from computational graph
                    gradient = gradient.detach()

                _set_module_by_path(
                    model, context_parameter_name,
                    nn.Parameter(
                        context_params[i] - self.context_lr * gradient
                    )
                )

            # Synchronize updates across shared context parameters
            self._sync_shared_context_references(model)

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
            # Only one type of supported model
            assert isinstance(
                model.models[lang_pair].decoder, CAVIATransformerDecoder
            )

            model.decoder.set_lang_pair_idx(lang_pair_idx)

        # No context parameter changes, used those saved within the model
        # to perform evaluation
        return super().inference_step(
            generator, models, sample, prefix_tokens, constraints
        )
