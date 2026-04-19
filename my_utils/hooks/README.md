# hooks

基于 PyTorch hook 的观测与局部 profiling 控制。

## 30秒定位

1. 我想按训练窗口触发 profiler start/stop  
用 `ForwardProfilerHook`

2. 我想记录模块级前向输入输出信息  
用 `ForwardTraceRecorder`

3. 我想要模块级耗时统计（历史方案）  
用 `ModuleProfiler`（可选）

## 最小示例

```python
from my_utils.hooks import ForwardTraceRecorder

recorder = ForwardTraceRecorder()
recorder.register(model)
# forward 后可读取 recorder 的记录结果
```

## 关键文件

- `ForwardProfileHook.py`: `ForwardProfilerHook`
- `module_hook.py`: `ForwardTraceRecorder`
- `moduleProfiler.py`: `ModuleProfiler`

## 兼容性

- 旧路径 `my_utils.ForwardProfileHook`、`my_utils.module_hook` 仍可用。  
