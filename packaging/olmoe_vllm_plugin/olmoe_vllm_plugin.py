"""vLLM general plugin that registers open-instruct's OLMoE model override.

WHY A SEPARATE INSTALLED PACKAGE FOR TEN LINES. vLLM resolves a model architecture through a
per-process ``ModelRegistry``, and with ``AsyncLLM`` the model is built neither in the Ray actor nor
in one process: EngineCore is a subprocess and its tensor-parallel workers are children of that.
So the registration has to happen in processes we do not start.

Three mechanisms were tried, in this order, and only the third is correct:

1. **Register in the Ray actor.** Registers in the one process that never builds a model.
   ``assert_registered()`` passed there while all four workers raised
   ``KeyError: 'layers.0.mlp.experts.gate_up_proj'`` (run_019fe36d).
2. **``sitecustomize.py`` on ``PYTHONPATH``.** Reaches every process, and did register correctly --
   but ``site`` imports it at *interpreter startup*, so it pulled vLLM in during vLLM's own worker
   spawn, before vLLM had initialised. EngineCore's shared-memory broadcast to its workers then
   blocked for twelve minutes with no progress (run_019fe734, cancelled).
3. **This.** ``vllm.general_plugins`` is loaded by vLLM itself in every process -- process0, engine
   core and workers -- at the point in *its* startup where importing vLLM machinery is safe. It is
   the mechanism vLLM provides for exactly this, and the only reason it was not used first is that
   entry points require installed distribution metadata, which the platform image lacked because it
   ``COPY``s the tree and sets ``PYTHONPATH`` rather than installing it. Installing this one small
   package supplies that metadata without making the whole repository a distribution.

The implementation stays trivial on purpose: it imports the real registration module from the
repository on ``PYTHONPATH`` and calls it, so there is exactly one definition of what gets
registered and this file cannot drift from it.
"""

from __future__ import annotations


def register() -> None:
    """Entry point for ``vllm.general_plugins``. Called by vLLM once per process."""
    # Imported inside the function, not at module scope: vLLM imports the plugin module itself, and
    # keeping the import here means the cost lands when vLLM asks for it rather than at import time.
    from open_instruct.spec_decode import registration  # noqa: PLC0415

    registration.register_olmoe_eagle3()
