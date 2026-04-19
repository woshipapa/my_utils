# core

基础工具层，不依赖具体 profiling 后端。

## 30秒定位

1. 我想做计时与日志  
看 `utils.py` + `logger.py`

2. 我想临时 patch 某个方法  
看 `method_patch.py`

3. 我想做通用调试（tensor/checksum）  
看 `utils.py`

## 最小示例

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

## 关键文件

- `utils.py`: `MyTimer`、`NoOpMyTimer`、`ChecksumUtils`、调试函数
- `logger.py`: `GlobalLogger`、`get_global_logger`
- `method_patch.py`: `MethodPatcher`、`MethodPatchHandle`
- `annotations.py`: `parametrize_shapes`

## 常用导入

```python
from my_utils.core import MyTimer, NoOpMyTimer, setup_logging_and_timer
from my_utils.core import GlobalLogger, get_global_logger
from my_utils.core import MethodPatcher
```
