# core

Foundation utilities, independent of any specific profiling backend.

## Quick orientation

1. Timing and logging: `utils.py` + `logger.py`.
2. Temporarily patching a method: `method_patch.py`.
3. General debugging helpers (tensor info, checksums): `utils.py`.

## Minimal example

```python
from my_utils.core import setup_logging_and_timer

logger, timer = setup_logging_and_timer(
    logger_name="train",
    log_file="train.log",
    use_cuda=True,
    rank=0,
)

timer.start("step")
# ... training code ...
timer.stop("step")
```

## Key files

- `utils.py` — `MyTimer`, `NoOpMyTimer`, `setup_logging_and_timer`,
  `ChecksumUtils`, and debugging helpers (`print_tensor_info`,
  `tensor_md5`, `print_cuda_memory_gb`, ...).
- `logger.py` — `GlobalLogger`, `get_global_logger`.
- `method_patch.py` — `MethodPatcher`, `MethodPatchHandle`.
- `annotations.py` — `parametrize_shapes`.

## Common imports

```python
from my_utils.core import MyTimer, NoOpMyTimer, setup_logging_and_timer
from my_utils.core import GlobalLogger, get_global_logger
from my_utils.core import MethodPatcher
```

torch is optional here: CUDA-aware timing and tensor helpers only activate
when torch is installed.

---

Chinese original: [docs/zh/core/README.md](../../docs/zh/core/README.md)
