"""EAGLE-3 target support for OLMoE, so speculative decoding can drive an RL rollout.

WHY THIS FILE EXISTS. EAGLE-3 drafts from the *target's* intermediate hidden states, so the
target has to hand three of them out alongside its final one. vLLM asks for that through two
things: the ``SupportsEagle3`` marker on the ``ForCausalLM`` class, and an inner model whose
``forward`` returns ``(hidden_states, aux_hidden_states)``. vLLM's ``OlmoeForCausalLM``
declares neither -- it is ``(nn.Module, SupportsPP, SupportsLoRA)`` and its
``OlmoeModel.forward`` returns one tensor -- so an ``eagle3`` speculative config against an
OLMoE policy cannot draft. ``Qwen3MoeModel`` is the same architecture shape with the support
already in it, and is what the ``forward`` below follows.

WHY IT SUBCLASSES RATHER THAN COPIES, AND THE ONE PLACE THAT REVERSES. ``OlmoeModel`` is
decorated with ``@support_torch_compile``, which is easy to get wrong in both directions:

- The decorator *mutates the class it is given*, appending ``TorchCompileWithNoGuardsWrapper``
  to ``cls.__bases__`` and replacing ``cls.__init__``. Its "already decorated" guard reads
  ``cls.__bases__``, which lists direct bases only. So subclassing ``OlmoeModel`` **and
  re-applying the decorator** would append that wrapper a second time and double-wrap
  ``__init__``. :class:`OlmoeModelEagle3` therefore carries no decorator of its own.
- Without re-decorating, the inherited wrapper still compiles the override, because it reads
  ``self.__class__.forward`` rather than a captured function (``compilation/wrapper.py``).
  So the subclass gets ``__init__``, ``load_weights`` and ``get_expert_mapping`` for free --
  which matters most for ``load_weights``, the ~60 lines that map 64 experts per layer and
  the stacked ``qkv_proj``. A copy of that going stale would mis-load weights in silence.

``OlmoeForCausalLM.__init__`` is the exception: it is not decorated, but it hard-codes
``self.model = OlmoeModel(...)``, so pointing it at our inner model means restating it.
``test_olmoe_eagle3.py`` compares the set of ``self.<attr>`` assignments in our copy against
upstream's, so an attribute added upstream fails a test instead of leaving a half-built model.

WHAT THIS DOES NOT DO. The draft is not trained here and is not updated during RL. vLLM keeps
target and draft weight transfer on separate calls -- ``start_weight_update()`` for the target,
``start_draft_weight_update()`` for the drafter -- and open-instruct only ever calls the first,
so the draft stays frozen at whatever it was initialised to. That is the paper's "offline" mode,
worth 1.77x -> 1.78x against an in-domain draft (arXiv:2604.26779 Table 5). None of it changes
what the rollout samples: rejection sampling makes the accepted tokens the *target's* samples,
so a stale draft costs throughput and never accuracy.

WHERE THE REGISTRATION LIVES. Not here, on purpose:
:mod:`open_instruct.spec_decode.registration` installs a lazy ``"<module>:<class>"`` pointer to
this file's target and imports nothing else from vLLM. Putting the installer beside the class
would mean importing the class in order to register it, and the point of registering lazily is
that the parent process does not import it at all.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3, supports_eagle3
from vllm.model_executor.models.olmoe import OlmoeDecoderLayer, OlmoeForCausalLM, OlmoeModel
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors

from open_instruct.spec_decode import moe_weights


class OlmoeModelEagle3(OlmoeModel, EagleModelMixin):
    """``OlmoeModel`` that also returns the hidden states EAGLE-3 drafts from.

    ``EagleModelMixin`` carries ``aux_hidden_state_layers`` and the two methods that read and
    write it; ``forward`` below is the only thing that changes, and only to call them.

    Deliberately **not** decorated with ``@support_torch_compile`` -- see the module docstring.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            # noqa rather than a ternary: this block is byte-identical to
            # OlmoeModel.forward's, so the only textual difference between that method and
            # this one is the auxiliary-state collection. Rewriting it would hide that.
            if inputs_embeds is not None:  # noqa: SIM108
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Absolute layer indices, so the requested set still means the same layers under
        # pipeline parallelism, where this rank owns [start_layer, end_layer) not [0, n).
        aux_hidden_states = self._maybe_add_hidden_state([], self.start_layer, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer), start=self.start_layer
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        # A bare tensor whenever nothing was requested, because that is what every caller
        # other than an EAGLE-3 drafter expects: ``gpu_model_runner`` unpacks the 2-tuple only
        # when ``use_aux_hidden_state_outputs`` is set, and that is set only for eagle3.
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class OlmoeForCausalLMEagle3(OlmoeForCausalLM, SupportsEagle3):
    """``OlmoeForCausalLM`` wired to :class:`OlmoeModelEagle3`.

    Everything but ``__init__`` and ``forward`` is inherited -- ``compute_logits``,
    ``load_weights``, ``get_expert_mapping``, ``embed_input_ids`` and the packed-module
    mapping. The two EAGLE-3 methods, ``set_aux_hidden_state_layers`` and
    ``get_eagle3_default_aux_hidden_state_layers``, come from ``SupportsEagle3`` and need no
    override: the default is ``(2, n // 2, n - 3)``, which for OLMoE's 16 layers is
    ``(2, 8, 13)`` -- the early/middle/late spread EAGLE-3 wants.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", layer_type: type[nn.Module] = OlmoeDecoderLayer):
        # ``nn.Module.__init__`` rather than ``super().__init__``: the parent would build a
        # plain ``OlmoeModel`` and allocate every weight in it before we could swap it, which
        # on a 7B checkpoint is a real allocation and not just a wasted reference. Kept in
        # step with upstream by the source hash in ``test_olmoe_eagle3.py``.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = OlmoeModelEagle3(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"), layer_type=layer_type
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=maybe_prefix(prefix, "lm_head")
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # Passed straight through. The inner model decides whether this is a tensor or a
        # 2-tuple, and the model runner branches on the same condition it does.
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The only functional difference from upstream's, and it has nothing to do with EAGLE-3:
        # transformers 5.x hands over each layer's experts as two fused tensors, which vLLM 0.21's
        # per-expert mapping cannot match, so a GRPO weight sync dies with
        # KeyError: 'layers.0.mlp.experts.gate_up_proj'. unfuse_olmoe_experts expands them and is a
        # no-op on an already-per-expert stream, so the initial load from a transformers-4-era Hub
        # checkpoint is unaffected. See moe_weights.py for why the split order is derived rather
        # than assumed.
        #
        # Submodule names are otherwise identical to upstream's, so an OLMoE checkpoint still loads
        # with no remapping and no missing keys, and the qkv stacking and MoE routing come from the
        # inherited OlmoeModel.load_weights that AutoWeightsLoader finds.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(moe_weights.unfuse_olmoe_experts(weights))


def verify_supports_eagle3() -> None:
    """Raise unless this module's target still satisfies vLLM's EAGLE-3 protocol.

    Separate from the registration check in :mod:`open_instruct.spec_decode.registration`,
    which answers "is the registry pointing here"; this answers "is what it points at still
    an EAGLE-3 target". Importing this module is the cost, so it is called from tests and
    from the one-off smoke check rather than from the engine path.
    """
    if not supports_eagle3(OlmoeForCausalLMEagle3):
        raise RuntimeError(
            f"{OlmoeForCausalLMEagle3.__name__} no longer satisfies vLLM's SupportsEagle3 "
            "protocol. Speculative decoding would be configured and silently inert."
        )
