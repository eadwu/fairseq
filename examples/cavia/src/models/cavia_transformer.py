import math

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, List, Optional

from fairseq.models.transformer import TransformerDecoder
from fairseq.modules import TransformerDecoderLayer


class CAVIATransformerDecoder(TransformerDecoder):
    """CAVIA (adapted with BatchEnsemble) implemented in TransformerDecoder
    """

    def __init__(
        self, args, dictionary, embed_tokens, no_encoder_attn=False
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )

        self.layers = nn.ModuleList(
            [
                self._build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )

    def set_lang_pair_idx(self, lang_pair_idx):
        for layer in self.layers:
            layer.set_lang_pair_idx(lang_pair_idx)

    def _build_decoder_layer(self, args, no_encoder_attn=False):
        return CAVIATransformerDecoderLayer(
            args, no_encoder_attn=no_encoder_attn
        )


class CAVIATransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer with BatchEnsemble weights
    """
    def __init__(
        self,
        args,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

        self.args = args

        # The number of tasks is the number of language pairs
        self.lang_pairs = args.lang_pairs
        if isinstance(self.lang_pairs, str):
            self.lang_pairs = self.lang_pairs.split(",")
        self.n_tasks = len(self.lang_pairs)

        # BatchEnsemble
        self.batch_ensemble_root = getattr(args, "batch_ensemble_root", -1)
        self.kaiming_init = getattr(args, "batch_ensemble_kaiming_init", False)

        # BatchEnsemble current language pair [index]
        self.lang_pair_idx = None

        for i in range(self.n_tasks):
            # Initialize context parameters (BatchEnsemble) for CAVIA
            # Due to the manual gradient and Tensor processing steps, these
            # should only _ever_ be indexed through their registered parameter
            # names.
            r_i = nn.Parameter(torch.zeros(
                args.decoder_ffn_embed_dim, 1,
                dtype=torch.float16 if args.fp16 else torch.float32,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            ), requires_grad=True)
            s_i = nn.Parameter(torch.zeros(
                self.embed_dim, 1,
                dtype=torch.float16 if args.fp16 else torch.float32,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            ), requires_grad=True)

            self.register_parameter(f"context_param-r_{i}", r_i)
            self.register_parameter(f"context_param-s_{i}", s_i)

            if self.kaiming_init:
                nn.init.kaiming_uniform_(r_i, a=math.sqrt(5))
                nn.init.kaiming_uniform_(s_i, a=math.sqrt(5))

            if hasattr(self.fc1, "bias"):
                b_i = nn.Parameter(torch.zeros(
                    args.decoder_ffn_embed_dim, 1,
                    dtype=torch.float16 if args.fp16 else torch.float32,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                ), requires_grad=True)

                self.register_parameter(f"context_param-b_{i}", b_i)
                if self.kaiming_init:
                    nn.init.kaiming_uniform_(b_i, a=math.sqrt(5))

    def set_lang_pair_idx(self, lang_pair_idx):
        self.lang_pair_idx = lang_pair_idx
        # Set gradient on shared weights based on lifelong learning
        self.fc1.requires_grad_(
            self.batch_ensemble_root == -1 or
            lang_pair_idx == self.batch_ensemble_root
        )

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        # Independent r_i, s_i, and b_i for each language pair
        r_i = getattr(self, f"context_param-r_{self.lang_pair_idx}")
        s_i = getattr(self, f"context_param-s_{self.lang_pair_idx}")

        w_i = r_i @ s_i.T
        W = getattr(self.fc1, "weight") * w_i
        x = x @ W.T
        if hasattr(self.fc1, "bias"):
            b_i = getattr(self, f"context_param-b_{self.lang_pair_idx}").squeeze(dim=1)
            b = getattr(self.fc1, "bias") * b_i
            x = x + b

        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None
