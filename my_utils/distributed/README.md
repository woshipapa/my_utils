# distributed

## 作用
`distributed` 放置分布式训练相关的辅助能力：时钟对齐、跨进程屏障、序列并行 padding。

## 文件
- `clockSyncUtils.py`: `ClockSynchronizer` / `SocketClockSynchronizer`。
- `etcd_utils.py`: 基于 etcd 的 barrier（可选依赖 `etcd3`）。
- `pad.py`: sequence parallel 场景的 `pad/remove_pad`（可选依赖 `megatron.core`）。

## 常用导入
```python
from my_utils.distributed.clockSyncUtils import ClockSynchronizer
from my_utils.distributed import etcd_barrier
from my_utils.distributed import pad_for_sequence_parallel, remove_pad_by_value
```

## 说明
- `etcd` / `megatron` 相关接口是可选导入，缺依赖时不影响其它模块。
