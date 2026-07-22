# legacy_profilers

历史 profiler 兼容层，用于旧工程平滑迁移。

## 30秒定位

1. 旧项目里还在用 DITProfiler 语义  
用 `create_profiler_context`

2. 旧项目里直接用 torch.profiler wrapper  
用 `ProfilerWrapper`

3. 新项目应该怎么做  
优先使用 `my_utils.profiling` 的统一流程

## 关键文件

- `DITProfiler.py`: `create_profiler_context`
- `profilerwrapper.py`: `ProfilerWrapper`

## 迁移建议

旧项目先保持 `legacy_profilers` 不动，先把主流程迁到 `my_utils.profiling`，再逐步移除历史调用。  
