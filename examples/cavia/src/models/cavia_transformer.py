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
                self._build_decoder_layer(args, no_encoder_attn, idx)
                for idx in range(args.decoder_layers)
            ]
        )

    def set_lang_pair_idx(self, lang_pair_idx):
        for layer in self.layers:
            layer.set_lang_pair_idx(lang_pair_idx)

    def _build_decoder_layer(self, args, no_encoder_attn=False, idx=None):
        return CAVIATransformerDecoderLayer(
            args, idx, no_encoder_attn=no_encoder_attn
        )

    def context_parameters(self, lang_pair_idx):
        return [
            param
            for layer in self.layers
            for context_params in layer.context_parameters(lang_pair_idx)
            for param in context_params
        ]

    def reset_context_parameters(self, lang_pair_idx):
        for layer in self.layers:
            layer.reset_context_parameters(lang_pair_idx)


class CAVIATransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer with BatchEnsemble weights
    """
    def __init__(
        self,
        args,
        idx,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

        self.lang_pair_idx = None

        # The number of tasks is the number of language pairs
        self.lang_pairs = args.lang_pairs
        if isinstance(self.lang_pairs, str):
            self.lang_pairs = self.lang_pairs.split(",")
        self.n_tasks = len(self.lang_pairs)

        # BatchEnsemble lifelong learning
        ## Which task should update the shared parameters
        self.batch_ensemble_root = getattr(args, "batch_ensemble_root", -1)

        # Initialize context parameters, BatchEnsemble r_i, s_i, and b_i
        # r_i is the column vector
        self.r_i = [
            torch.zeros(
                args.decoder_ffn_embed_dim, 1,
                dtype=torch.float16 if args.fp16 else torch.float32,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                requires_grad=True,
            )
            for _ in range(self.n_tasks)
        ]
        # s_i is the row vector
        self.s_i = [
            torch.zeros(
                self.embed_dim, 1,
                dtype=torch.float16 if args.fp16 else torch.float32,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                requires_grad=True,
            )
            for _ in range(self.n_tasks)
        ]
        # b_i is the bias vectors
        self.b_i = [
            torch.zeros(
                args.decoder_ffn_embed_dim, 1,
                dtype=torch.float16 if args.fp16 else torch.float32,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                requires_grad=True,
            )
            for _ in range(self.n_tasks)
        ]

        for i in range(self.n_tasks):
            # Register as buffers so that parameters are saved but not included
            # into the parameters() iterator
            self.register_buffer(f"{idx}-r_{i}", self.r_i[i])
            self.register_buffer(f"{idx}-s_{i}", self.s_i[i])
            self.register_buffer(f"{idx}-b_{i}", self.b_i[i])
            # Ensure BatchEnsemble parameters can calculate their gradients
            self.r_i[i].requires_grad = True
            self.s_i[i].requires_grad = True
            self.b_i[i].requires_grad = True

    def set_lang_pair_idx(self, lang_pair_idx):
        self.lang_pair_idx = lang_pair_idx

    def context_parameters(self, lang_pair_idx):
        return [
            (
                self.r_i[lang_pair_idx].parameters(),
                self.s_i[lang_pair_idx].parameters(),
                self.b_i[lang_pair_idx].parameters(),
            )
        ]

    def reset_context_parameters(self, lang_pair_idx):
        # Zero out tensors, don't calculate gradients with this operation
        with torch.no_grad():
            self.r_i[lang_pair_idx] *= 0
            self.s_i[lang_pair_idx] *= 0
            self.b_i[lang_pair_idx] *= 0
        # Reset gradient
        self.r_i[lang_pair_idx].grad = None
        self.s_i[lang_pair_idx].grad = None
        self.b_i[lang_pair_idx].grad = None

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

        def fc1_set_grad(cond):
            for param in self.fc1.parameters():
                param.requires_grad_(cond)

        # Independent r_i and s_i for each lang_pair
        r_i = self.r_i[self.lang_pair_idx]
        s_i = self.s_i[self.lang_pair_idx]
        w_i = r_i @ s_i.T
        # Set gradient on shared weights based on lifelong learning
        fc1_set_grad(self.lang_pair_idx == self.batch_ensemble_root)
        W = self.fc1.weight * w_i
        x = x @ W.T

        b_i = self.b_i[self.lang_pair_idx]
        x = x + b_i

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
