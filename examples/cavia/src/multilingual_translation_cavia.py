from fairseq.logging.metrics import state_dict
import torch
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask


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
        return [
            parameters
            for name, parameters in model.named_parameters()
            if "context_param" in name and (
                f"r_{lang_pair_idx}" in name or
                f"s_{lang_pair_idx}" in name or
                f"b_{lang_pair_idx}" in name
            )
        ]

    def _per_lang_pair_train_loss(
        self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        # Reset context parameters on every new task
        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)

        # Calculate loss with current parameters
        loss, sample_size, logging_output = criterion(
            model.models[lang_pair], sample[lang_pair]
        )

        if ignore_grad:
            loss *= 0

        context_parameters = self._get_context_parameters(
            lang_pair_idx, model.models[lang_pair]
        )

        for _ in range(self.args.cavia_inner_updates):
            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss,
                context_parameters,
                create_graph=not self.args.cavia_first_order
            )

            # Gradient Descent on context parameters
            for i, _ in enumerate(task_gradients):
                gradient = task_gradients[i]
                if self.args.cavia_first_order:
                    gradient = gradient.detach()
                gradient = self.context_lr * gradient
                context_parameters[i] = context_parameters[i] - gradient

            # Recompute loss after context parameter update
            loss, sample_size, logging_output = criterion(
                model.models[lang_pair], sample[lang_pair]
            )

            if ignore_grad:
                loss *= 0

        # Compute task_gradients with respect to the shared [model] parameters
        task_gradients = torch.autograd.grad(loss, self.shared_parameters)

        # Update meta-gradient
        for i in range(len(task_gradients)):
            self.meta_gradient[i] += task_gradients[i].detach()

        # Flush context parameters just in case
        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)
        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        base_model = model.models[self.model_lang_pairs[0]]
        self.shared_parameters = [
            parameters
            for name, parameters in base_model.named_parameters()
            if "context_param" not in name
        ]

        self.meta_gradient = [0 for _ in range(len(self.shared_parameters))]

        agg_loss, agg_sample_size, agg_logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

        # Apply meta-gradient to shared parameters
        optimizer.zero_grad()
        for i, param in enumerate(self.shared_parameters()):
            param.grad = self.meta_gradient[i] / self.n_tasks
            param.grad.data.clamp_(-10, 10)
        optimizer.step()

        return agg_loss, agg_sample_size, agg_logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        # Since context parameters are only reset in the beginning and the
        # model was implemented in such a way that context parameters are
        # independent from one another, this allows for the context parameters
        # to be saved as part of the model for independent evaluation.
        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)

        # Calculate loss with current parameters
        loss, sample_size, logging_output = criterion(
            model.models[lang_pair], sample[lang_pair]
        )

        context_parameters = self._get_context_parameters(
            lang_pair_idx, model.models[lang_pair]
        )

        for _ in range(self.args.cavia_inner_updates):
            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss,
                context_parameters,
                create_graph=not self.args.cavia_first_order
            )

            # Gradient Descent on context parameters
            for i, _ in enumerate(task_gradients):
                gradient = task_gradients[i]
                if self.args.cavia_first_order:
                    gradient = gradient.detach()
                gradient = self.context_lr * gradient
                context_parameters[i] = context_parameters[i] - gradient

            # Recompute loss after context parameter update
            loss, sample_size, logging_output = criterion(
                model.models[lang_pair], sample[lang_pair]
            )

        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        lang_pair = f"{self.args.source_lang}-${self.args.target_lang}"

        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        for model in models:
            model.decoder.set_lang_pair_idx(lang_pair_idx)

        super().inference_step(
            generator, models, sample, prefix_tokens, constraints
        )
