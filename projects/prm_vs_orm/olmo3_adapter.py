"""``Olmo3ForSequenceClassification`` for the PRM-vs-ORM reward models.

The base policy is Olmo3 370M. ``open_instruct/olmo_adapter/__init__.py`` ships sequence-
classification wrappers for Olmo2 and Olmoe but **not** Olmo3, and the installed
``transformers`` (5.4.0) exposes ``Olmo3Model`` / ``Olmo3PreTrainedModel`` / ``Olmo3ForCausalLM``
but no ``Olmo3ForSequenceClassification``. This module adds one, mirroring the upstream Olmo2
adapter (a ``score`` linear head on the last hidden state, last-non-pad-token pooling,
MSE/CE/BCE by ``num_labels`` + ``problem_type``), and registers it with
``AutoModelForSequenceClassification`` so ``from_pretrained`` on the converted Olmo3 checkpoint
builds an Olmo3 backbone + a fresh classification head.

Additive and confined to the experiment overlay — it does not edit ``open_instruct``. The
reward trainer scores via ``rm_common.pooled_scores``, which calls this class's ``forward`` (the
DistributedDataParallel-correct entry point) and reads its pooled ``logits``. ``forward`` pools at
the last real token using the attention mask (robust to ``pad_token_id == eos_token_id``), falling
back to the upstream Olmo2-style pad-token scan only when no mask is supplied.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.olmo3.modeling_olmo3 import Olmo3Config, Olmo3Model, Olmo3PreTrainedModel


class Olmo3ForSequenceClassification(Olmo3PreTrainedModel):
    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Olmo3Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | SequenceClassifierOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        # Pool at the last *real* token. Prefer the explicit attention mask: a pad-token scan
        # mislocates the position whenever ``pad_token_id == eos_token_id`` (it stops at the real
        # eos), and the RM collator always supplies a mask. Fall back to the pad-token scan only
        # when no mask is given, preserving the upstream Olmo2-style behaviour for that path.
        if attention_mask is not None:
            sequence_lengths = attention_mask.long().sum(-1) - 1
            sequence_lengths = sequence_lengths.to(logits.device)
        elif self.config.pad_token_id is None:
            if batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


def register() -> None:
    """Register the Olmo3 sequence-classification head with the Auto* factories (idempotent).

    ``exist_ok=True`` on the seq-classification mapping is load-bearing: newer ``transformers``
    images ship a *native* ``Olmo3ForSequenceClassification`` and pre-register ``Olmo3Config`` in
    ``AutoModelForSequenceClassification``'s mapping at import, so a plain ``.register`` raises
    ``'Olmo3Config' is already used by a Transformers model``. ``exist_ok=True`` overrides that
    mapping with *this* module's class (the one whose forward/pooling the RM path was validated
    against) whether or not a native class is present, and is a harmless no-op on images where it
    is absent. ``AutoConfig.register`` stays wrapped in ``suppress(ValueError)`` for the same
    already-registered case.
    """
    with contextlib.suppress(ValueError):
        AutoConfig.register("olmo3", Olmo3Config)  # raises if already registered this process
    AutoModelForSequenceClassification.register(Olmo3Config, Olmo3ForSequenceClassification, exist_ok=True)
