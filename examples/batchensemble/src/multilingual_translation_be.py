from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask


@register_task("multilingual_translation_be")
class MultilingualTranslationBatchEnsembleTask(MultilingualTranslationTask):
    """A task for multilingual translation with BatchEnsemble.
    """
    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        # fmt: off
        # args for Training with BatchEnsemble
        parser.add_argument('--batch-ensemble-root', type=int, default=-1,
                            help='Batch Ensemble root task (0-based) for lifelong learning')
        parser.add_argument('--batch-ensemble-linear-init', default=False, action='store_true',
                            help='Initialize weights and biases akin to nn.Linear')
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
            # Validate argument batch_ensemble_root
            assert args.batch_ensemble_root == -1 or (
                args.batch_ensemble_root >= 0 and
                args.batch_ensemble_root < len(self.lang_pairs)
            )

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

        return super()._per_lang_pair_train_loss(
            lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
        )

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        return super()._per_lang_pair_valid_loss(
            lang_pair, model, criterion, sample
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        lang_pair = f"{self.args.source_lang}-{self.args.target_lang}"

        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        for model in models:
            model.decoder.set_lang_pair_idx(lang_pair_idx)

        return super().inference_step(
            generator, models, sample, prefix_tokens, constraints
        )
