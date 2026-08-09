"""Register the OLMoE model override in every Python process, including vLLM's subprocesses.

WHY THIS FILE EXISTS, AND WHY THE OBVIOUS PLACES DO NOT WORK.

vLLM resolves a model architecture through a per-process ``ModelRegistry``. open-instruct's
``LLMRayActor`` registers the OLMoE override before building the engine, which is enough only if
the model is instantiated in that process. It is not:

    (LLMRayActor pid=9085)  (EngineCore pid=9811)  (Worker_TP0 pid=9898) ... (Worker_TP3 pid=9901)

``AsyncLLM`` runs EngineCore as a *subprocess*, and its TP workers are children of that. So a
registration made in the actor is made in the one process that never builds the model. Diagnosed on
run_019fe36d: ``assert_registered()`` passed in the actor while all four workers raised
``KeyError: 'layers.0.mlp.experts.gate_up_proj'`` from upstream's loader.

The two mechanisms vLLM offers both fail here:

- ``vllm.general_plugins`` entry points are loaded in every process, which is exactly right -- but
  entry points need installed distribution metadata, and the platform image does ``COPY . /opt/...``
  plus ``PYTHONPATH`` and never ``pip install``, so ``importlib.metadata`` finds nothing.
- ``VLLM_WORKER_MULTIPROC_METHOD=fork`` would let children inherit the parent's registry, but that
  only covers the workers, not the EngineCore subprocess above them, and forking a CUDA-initialised
  parent is its own hazard.

``sitecustomize`` is imported by the ``site`` module at interpreter startup for any module of that
name on ``sys.path``, and the image puts the repository on ``PYTHONPATH``. So it runs in the actor,
in EngineCore, and in every worker, with no packaging and no image change.

GUARDED BY AN ENVIRONMENT VARIABLE, deliberately. Importing vLLM costs seconds, and this file would
otherwise run in every DeepSpeed learner process too, where it is useless. ``create_vllm_engines``
sets the variable in the vLLM actor's runtime environment, and subprocesses inherit it.

NEVER FATAL. A failure here would break interpreter startup for every process in the repository,
including ones that have nothing to do with vLLM. So it reports and continues; if the registration
did not happen, the run fails later with vLLM's own error rather than with a broken interpreter.
"""

import os
import sys

#: Set by ``create_vllm_engines`` on the vLLM actor's runtime environment. Absent everywhere else,
#: which keeps the vLLM import out of learner processes that will never build a model.
_ENABLE_VAR = "OPEN_INSTRUCT_REGISTER_OLMOE"

if os.environ.get(_ENABLE_VAR) == "1":
    try:
        from open_instruct.spec_decode import registration

        registration.register_olmoe_eagle3()
    except Exception as error:  # noqa: BLE001 - see NEVER FATAL above
        print(
            f"sitecustomize: could not register the OLMoE override ({type(error).__name__}: {error}). "
            "vLLM will use its own OlmoeForCausalLM, which cannot unfuse transformers-5 expert "
            "tensors and has no EAGLE-3 auxiliary hidden states.",
            file=sys.stderr,
        )
