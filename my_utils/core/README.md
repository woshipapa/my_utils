# core

## 作用
`core` 放置不依赖具体 profiling 后端的基础能力：日志、计时、方法 patch、通用调试工具。

## 文件
- `annotations.py`: 参数化 shape 的装饰器工具。
- `logger.py`: `GlobalLogger` 单例日志与 profile 事件写出。
- `method_patch.py`: 运行时方法替换与恢复。
- `utils.py`: `MyTimer`、`NoOpMyTimer`、`ChecksumUtils`、调试辅助函数等。

## 常用导入
```python
from my_utils.core.logger import GlobalLogger, get_global_logger
from my_utils.core.utils import MyTimer, NoOpMyTimer, setup_logging_and_timer
from my_utils.core.method_patch import MethodPatcher
```

## 说明
- 旧路径 `from my_utils.utils import MyTimer` 仍兼容，但新代码建议使用 `my_utils.core.*`。
