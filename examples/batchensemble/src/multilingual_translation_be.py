from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

import torch


@register_task("multilingual_translation_be")
class MultilingualTranslationBatchEnsembleTask(MultilingualTranslationTask):
    """A task for multilingual translation with BatchEnsemble.
    """
    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        # fmt: off
        # args for Training with BatchEnsemble
        parser.add_argument('--batchensemble-vanilla', default=False, action='store_true',
                            help='Adjusts the behavior of BatchEnsemble to be like that of the paper, an ensemble')
        parser.add_argument('--batchensemble-lifelong-learning', default=False, action='store_true',
                            help='BatchEnsemble root task (0-based) for lifelong learning')
        parser.add_argument('--batchensemble-lr-multiplier', type=float, default=1.0,
                            help='Learning rate multiplier for BatchEnsemble parameters')
        parser.add_argument('--batchensemble-lr-relative', default=False, action='store_true',
                            help='Learning rate relative to optimizer.get_lr()')
        parser.add_argument('--batchensemble-normal-init', default=False, action='store_true',
                            help='Initialize r_i and s_i with random normal initialization')
        parser.add_argument('--batchensemble-kaiming-init', default=False, action='store_true',
                            help='Initialize r_i and s_i with kaiming initialization (he_normal)')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)

        self.args = args
        self.lang_pairs = args.lang_pairs
        if isinstance(self.lang_pairs, str):
            self.lang_pairs = self.lang_pairs.split(",")
        self.eval_lang_pairs = self.lang_pairs
        self.model_lang_pairs = self.lang_pairs

        self.n_tasks = len(self.lang_pairs)
        assert self.n_tasks > 0

        self.batchensemble_vanilla = getattr(
            args, "batchensemble_vanilla", False
        )

        if training:
            self.lr_multiplier = getattr(
                args, "batchensemble_lr_multiplier", 1.0
            )
            self.relative_lr = getattr(
                args, "batchensemble_lr_relative", False
            )

            self.context_lr = self.args.lr[0] * self.lr_multiplier

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

    def _per_lang_pair_train_loss(
        self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        # Fixup target vector based off ensembling size
        if self.batchensemble_vanilla:
            sample[lang_pair]["target"] = torch.tile(
                sample[lang_pair]["target"],
                [self.n_tasks, 1]
            )
            sample[lang_pair]["ntokens"] *= self.n_tasks

            model.models[lang_pair].with_state(self.n_tasks, False)
        else:
            # Or `1, False`
            model.models[lang_pair].with_state(None, False)

        return super()._per_lang_pair_train_loss(
            lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        # Calculate the decaying learning rate if requested
        if self.relative_lr:
            self.context_lr = optimizer.get_lr() * self.lr_multiplier

        agg_loss, agg_sample_size, agg_logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

        # Fixup gradients ... this operates under a shared decoder assumption
        assert getattr(self.args, "share_decoders", False)
        first_lang_pair = self.lang_pairs[0]
        # Calculation may not be equivalent depending on how gradients are
        # accumulated
        with torch.no_grad():
            for name, param in model.named_parameters():
                if first_lang_pair in name and (
                    "context_param" in name or
                    "alpha" in name or
                    "gamma" in name or
                    "ensemble_bias" in name
                ):
                    param.grad *= self.context_lr

        return agg_loss, agg_sample_size, agg_logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        # Results are averaged during validation / testing, independent errors
        # are only needed during the optimization step (training)
        if self.batchensemble_vanilla:
            model.models[lang_pair].with_state(self.n_tasks, True)
        else:
            # Or `1, False`
            model.models[lang_pair].with_state(None, False)

        return super()._per_lang_pair_valid_loss(
            lang_pair, model, criterion, sample
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        lang_pair = f"{self.args.source_lang}-{self.args.target_lang}"

        # Update model state
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        for model in models:
            model.decoder.set_lang_pair_idx(lang_pair_idx)
            # Results are averaged during validation / testing,
            # independent errors are only needed during the optimization step
            if self.batchensemble_vanilla:
                model.with_state(self.n_tasks, True)
            else:
                # Or `1, False`
                model.with_state(None, False)

        return super().inference_step(
            generator, models, sample, prefix_tokens, constraints
        )
