# memory

## 作用
`memory` 聚焦内存相关诊断：快照、OOM 信号、GPU 内存/硬件利用率监控。

## 文件
- `memory_snapshot.py`: `MemorySnapshotter` / `global_snapshotter`。
- `oom_restore.py`: 分布式 OOM 标志设置与检测。
- `gpu_mem_tracker.py`: `GPU_Performance_Tracker`（可选依赖 `pynvml` / `matplotlib`）。

## 常用导入
```python
from my_utils.memory.memory_snapshot import global_snapshotter
from my_utils.memory.oom_restore import set_oom_flag, check_oom_flag
```

## 说明
- 监控与快照能力建议按需开启，避免在长跑训练里引入额外开销。
