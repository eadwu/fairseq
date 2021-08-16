from collections import OrderedDict

import torch

from fairseq import utils
from fairseq.models import (
    register_model, register_model_architecture,
    FairseqMultiModel, FairseqEncoderDecoderModel,
)
from fairseq.models.multilingual_transformer import (
    MultilingualTransformerModel, multilingual_transformer_iwslt_de_en
)
from fairseq.models.transformer import (
    Embedding,
    TransformerEncoder,
)

from .be_transformer import BatchEnsembleTransformerDecoder


class BEFairseqEncoderDecoderModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

        self.ensemble_size = None
        self.avg_ensemble = False

    def with_state(self, ensemble_size, avg_ensemble):
        self.ensemble_size = ensemble_size
        self.avg_ensemble = avg_ensemble

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if self.ensemble_size is not None:
            src_tokens_dim = [1 for _ in src_tokens.shape]
            src_tokens_dim[0] = self.ensemble_size
            src_tokens = torch.tile(src_tokens, src_tokens_dim)

            src_lengths_dim = [1 for _ in src_lengths.shape]
            src_lengths_dim[0] = self.ensemble_size
            src_lengths = torch.tile(src_lengths, src_lengths_dim)

            prev_output_tokens_dim = [1 for _ in prev_output_tokens.shape]
            prev_output_tokens_dim[0] = self.ensemble_size
            prev_output_tokens = torch.tile(
                prev_output_tokens, prev_output_tokens_dim
            )

        decoder_out, output_dict = super().forward(
            src_tokens, src_lengths, prev_output_tokens, **kwargs
        )

        if self.avg_ensemble:
            decoder_out = torch.mean(torch.stack(torch.split(
                decoder_out, decoder_out.shape[0] / self.ensemble_size
            )))

        return decoder_out, output_dict


@register_model("be_multilingual_transformer")
class BatchEnsembleMultilingualTransformer(MultilingualTransformerModel):
    """A variant of standard multilingual Transformer models whose decoder
    supports uses BatchEnsemble.
    """
    def __init__(
        self, encoders, decoders, instance=BEFairseqEncoderDecoderModel
    ):
        super().__init__(encoders, decoders, instance=instance)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        MultilingualTransformerModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        batch_ensemble_multilingual_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split("-")[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split("-")[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=src_langs,
                    embed_dim=args.encoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.encoder_embed_path,
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embed_dim=args.decoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.decoder_embed_path,
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.encoder_embed_dim,
                        args.encoder_embed_path,
                    )
                lang_encoders[lang] = cls._get_module_class(
                    True, args, task.dicts[lang], encoder_embed_tokens, src_langs
                )
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.decoder_embed_dim,
                        args.decoder_embed_path,
                    )
                lang_decoders[lang] = cls._get_module_class(
                    False, args, task.dicts[lang], decoder_embed_tokens, tgt_langs
                )
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = (
                shared_encoder if shared_encoder is not None else get_encoder(src)
            )
            decoders[lang_pair] = (
                shared_decoder if shared_decoder is not None else get_decoder(tgt)
            )

        return BatchEnsembleMultilingualTransformer(encoders, decoders)

    @classmethod
    def _get_module_class(
        cls, is_encoder, args, lang_dict, embed_tokens, langs
    ):
        module_class = (
            TransformerEncoder
            if is_encoder
            else BatchEnsembleTransformerDecoder
        )

        return module_class(args, lang_dict, embed_tokens)


@register_model_architecture(
    "be_multilingual_transformer",
    "batch_ensemble_multilingual_transformer"
)
def batch_ensemble_multilingual_architecture(args):
    multilingual_transformer_iwslt_de_en(args)


@register_model_architecture(
    "be_multilingual_transformer",
    "batch_ensemble_phat_multilingual_transformer"
)
def batch_ensemble_phat_multilingual_architecture(args):
    # Latent Depth number of layers
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 24)

    batch_ensemble_multilingual_architecture(args)
