import pathlib

import torch

try:
    import vllm  # noqa: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

collect_ignore = []
if not VLLM_AVAILABLE:
    collect_ignore.extend([
        "open_instruct/test_vllm_utils.py",
        "open_instruct/test_data_loader.py",
        "open_instruct/test_grpo_fast.py",
        # Import vllm's OLMoE model, registry, and spec-decode stats types.
        "open_instruct/spec_decode/test_olmoe_eagle3.py",
        "open_instruct/spec_decode/test_metrics_vllm.py",
        "open_instruct/spec_decode/test_package_imports_vllm.py",
    ])
if not torch.cuda.is_available():
    collect_ignore.extend(str(p) for p in pathlib.Path("open_instruct").glob("*_gpu.py"))
