# legacy_profilers

## 作用
`legacy_profilers` 存放历史兼容的 profiler 封装，便于旧工程平滑迁移。

## 文件
- `DITProfiler.py`: 任务类型控制的 profiler context 工具。
- `profilerwrapper.py`: torch.profiler 封装与图表输出。

## 常用导入
```python
from my_utils.legacy_profilers.DITProfiler import create_profiler_context
from my_utils.legacy_profilers.profilerwrapper import ProfilerWrapper
```

## 说明
- 新项目优先使用 `my_utils.profiling` 的统一 metrics 流程。
