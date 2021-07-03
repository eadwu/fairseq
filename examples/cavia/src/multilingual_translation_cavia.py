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
        self.src_langs, self.tgt_langs = zip(
            *[
              (lang.split("-")[0], lang.split("-")[1])
              for lang in args.lang_pairs
            ]
        )

        # Needed to satisfy an assumption for shared gradient accumulation
        assert self.args.share_encoders
        # Needed for shared r_i, s_i, and b_i, through shared TransformerDecoder(s)
        assert self.args.share_decoders

        self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        self.eval_lang_pairs = self.lang_pairs
        self.model_lang_pairs = self.lang_pairs

        # Learning rate for context parameters
        self.context_lr = self.args.lr * self.args.cavia_lr_inner_multiplier

        # Hack for tracking in between functions without copying a bunch more code
        self.meta_gradient = None

    def _per_lang_pair_train_loss(
        self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        # Set language pair index
        lang_pair_idx = [
            i
            for i, lp in enumerate(self.model_lang_pairs)
            if lp == lang_pair
        ]

        assert len(lang_pair_idx) == 1
        lang_pair_idx = lang_pair_idx[0]
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        # Calculate loss with shared parameters
        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)
        loss, sample_size, logging_output = criterion(
            model.models[lang_pair], sample[lang_pair]
        )

        if ignore_grad:
            loss *= 0

        for _ in range(self.args.cavia_inner_updates):
            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss,
                model.models[lang_pair].decoder.context_parameters(lang_pair_idx),
                create_graph=not self.args.cavia_first_order
            )[0]

            # Needs to be detached in the case of multi-order task_gradients
            # using gradient descent
            model.context_params -= self.context_lr * task_gradients.detach()

            # Recompute loss after context parameter update
            loss, sample_size, logging_output = criterion(
                model.models[lang_pair], sample[lang_pair]
            )

            if ignore_grad:
                loss *= 0

        # Compute task_gradients with respect to the shared [model] parameters
        task_gradients = torch.autograd.grad(
            loss,
            model.models[lang_pair].parameters()
        )

        # Clamp and assign meta-gradient
        for i in range(len(task_gradients)):
            self.meta_gradient[i] += task_gradients[i].detach().clamp_(-10, 10)

        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)

        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        base_model = model.models[self.model_lang_pairs[0]]
        self.meta_gradient = [0 for _ in range(len(base_model.state_dict()))]

        agg_loss, agg_sample_size, agg_logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

        # Apply meta-gradient to shared parameters
        for i, param in enumerate(base_model.parameters()):
            param.grad = self.meta_gradient[i] / len(self.model_lang_pairs)
        optimizer.step()

        return agg_loss, agg_sample_size, agg_logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        lang_pair_idx = [
            i
            for i, lp in enumerate(self.model_lang_pairs)
            if lp == lang_pair
        ]

        assert len(lang_pair_idx) == 1
        lang_pair_idx = lang_pair_idx[0]

        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        model.models[lang_pair].decoder.reset_context_parameters(lang_pair_idx)
        loss, sample_size, logging_output = criterion(
            model.models[lang_pair], sample[lang_pair]
        )

        for _ in range(self.args.cavia_inner_updates):
            # Compute task_gradients with respect to context parameters
            task_gradients = torch.autograd.grad(
                loss,
                model.models[lang_pair].decoder.context_parameters(lang_pair_idx),
                create_graph=not self.args.cavia_first_order
            )[0]

            # Needs to be detached in the case of multi-order task_gradients
            # using gradient descent
            model.context_params -= self.context_lr * task_gradients.detach()

            # Recompute loss after context parameter update
            loss, sample_size, logging_output = criterion(
                model.models[lang_pair], sample[lang_pair]
            )

        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        raise NotImplementedError(
            "Inference is not supported with meta-learning"
        )

    @property
    def src_lang_idx_dict(self):
        return {lang: lang_idx for lang_idx, lang in enumerate(self.src_langs)}

    @property
    def tgt_lang_idx_dict(self):
        return {lang: lang_idx for lang_idx, lang in enumerate(self.tgt_langs)}
