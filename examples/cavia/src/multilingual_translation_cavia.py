from fairseq.logging.metrics import state_dict
import torch
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

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
                            help='Number of inner-loop updates')
        parser.add_argument('--cavia-lr-inner-multiplier', type=float, default=1.0,
                            help='Inner-loop learning rate multiplier (relative to global)')
        parser.add_argument('--cavia-first-order', default=False, action='store_true',
                            help='Run first-order version of CAVIA')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)

        # Needed to satisfy an assumption for shared gradient accumulation
        assert self.args.share_encoders
        # Needed for shared r_i, s_i, and b_i, through shared TransformerDecoder(s)
        assert self.args.share_decoders

        self.lang_pairs = args.lang_pairs
        self.eval_lang_pairs = self.lang_pairs
        self.model_lang_pairs = self.lang_pairs
        assert len(self.lang_pairs) > 0
        self.n_tasks = float(len(self.lang_pairs))

        # Validate argument batch_ensemble_root
        assert args.batch_ensemble_root == -1 or (
            args.batch_ensemble_root >= 0 and
            args.batch_ensemble_root < len(self.lang_pairs)
        )

        # Learning rate for context parameters
        self.context_lr = self.args.lr[0] * self.args.cavia_lr_inner_multiplier

        # Hack for tracking in between functions without copying a bunch more code
        self.meta_gradient = None
        self.shared_parameters = None

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

    def _get_context_parameters(self, lang_pair_idx, model):
        context_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if "context_param" in name and (
                f"r_{lang_pair_idx}" in name or
                f"s_{lang_pair_idx}" in name or
                f"b_{lang_pair_idx}" in name
            )
        ]

        return [
            [n for n, _ in context_parameters],
            [p for _, p in context_parameters],
        ]

    def _parse_context_module(self, path, lang_pair_idx):
        # decoder.layers.$DECODER_LAYER.context_param-$TYPE_$LANG_PAIR_IDX
        module_path = path.split(".")
        module_path = module_path[2:] # discard unecessary stuff

        decoder_layer = module_path[0]
        context_p_name = module_path[-1].split("-")

        context_p_name = context_p_name[-1] # discard unnecessary stuff
        context_type, context_lang_pair = context_p_name.split("_")

        # Proper type conversion
        assert decoder_layer.isdigit()
        decoder_layer = int(decoder_layer)

        # Just a check...
        assert context_lang_pair.isdigit()
        context_lang_pair = int(context_lang_pair)
        assert context_lang_pair == lang_pair_idx

        return decoder_layer, context_type, context_lang_pair

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

        # Reset context parameters on every new task
        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)

        context_n, context_p = self._get_context_parameters(
            lang_pair_idx, model.models[lang_pair]
        )

        for _ in range(self.args.cavia_inner_updates):
            # Calculate loss with current parameters
            loss, _, __ = run_model()

            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss,
                context_p,
                create_graph=not self.args.cavia_first_order,
            )

            # Gradient Descent on context parameters
            for i, _ in enumerate(task_gradients):
                decoder_layer, context_type, _ = self._parse_context_module(
                    context_n[i], lang_pair_idx
                )

                model.models[lang_pair].decoder._update_context_param(
                    decoder_layer, context_type, lang_pair_idx,
                    context_p[i] - self.context_lr * task_gradients[i]
                )

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
        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)
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

        # Since context parameters are only reset in the beginning and the
        # model was implemented in such a way that context parameters are
        # independent from one another, this allows for the context parameters
        # to be saved as part of the model for independent evaluation.
        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)

        context_n, context_p = self._get_context_parameters(
            lang_pair_idx, model.models[lang_pair]
        )

        for _ in range(self.args.cavia_inner_updates):
            # Calculate loss with current parameters
            loss, _, __ = criterion(
                model.models[lang_pair], sample[lang_pair]
            )

            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss,
                context_p,
                create_graph=not self.args.cavia_first_order,
            )

            # Gradient Descent on context parameters
            for i, _ in enumerate(task_gradients):
                decoder_layer, context_type, _ = self._parse_context_module(
                    context_n[i], lang_pair_idx
                )

                gradient = task_gradients[i]
                if self.args.cavia_first_order:
                    gradient = gradient.detach()
                model.models[lang_pair].decoder._update_context_param(
                    decoder_layer, context_type, lang_pair_idx,
                    context_p[i] - self.context_lr * gradient
                )

        # Recompute loss after context parameter update
        return criterion(model.models[lang_pair], sample[lang_pair])

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        lang_pair = f"{self.args.source_lang}-${self.args.target_lang}"

        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        for model in models:
            # Only one type of supported model
            assert isinstance(
                model.models[lang_pair].decoder, CAVIATransformerDecoder
            )

            model.decoder.set_lang_pair_idx(lang_pair_idx)

        super().inference_step(
            generator, models, sample, prefix_tokens, constraints
        )
