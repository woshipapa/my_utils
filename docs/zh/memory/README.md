# memory

内存诊断层：snapshot、OOM 信号、GPU 内存/利用率跟踪。

## 30秒定位

1. 我想在关键阶段留内存快照  
用 `global_snapshotter`

2. 我想跨 rank 传播 OOM 标记  
用 `set_oom_flag` / `check_oom_flag`

3. 我想监控 GPU 显存与利用率  
用 `GPU_Performance_Tracker`（可选依赖 `pynvml`）

## 最小示例

```python
from my_utils.memory import global_snapshotter, set_oom_flag, check_oom_flag

global_snapshotter.snapshot("before_step")
# ... train step ...
if check_oom_flag():
    print("OOM detected")
```

## 关键文件

- `memory_snapshot.py`: `MemorySnapshotter`、`global_snapshotter`
- `oom_restore.py`: OOM flag set/check
- `gpu_mem_tracker.py`: `GPU_Performance_Tracker`

## 注意

- 监控与快照建议按需开启，避免长跑训练中引入额外开销。  
