# hooks

## 作用
`hooks` 放置基于 PyTorch hook 的观测工具，关注前向/模块粒度跟踪与局部 profiling 控制。

## 文件
- `ForwardProfileHook.py`: 训练迭代窗口触发 profiler start/stop。
- `module_hook.py`: `ForwardTraceRecorder`，记录模块级输入输出与元信息。
- `moduleProfiler.py`: `ModuleProfiler`，模块耗时统计（可选）。

## 常用导入
```python
from my_utils.hooks.ForwardProfileHook import ForwardProfilerHook
from my_utils.hooks.module_hook import ForwardTraceRecorder
```

## 说明
- 旧路径 `my_utils.ForwardProfileHook`、`my_utils.module_hook` 仍兼容。
