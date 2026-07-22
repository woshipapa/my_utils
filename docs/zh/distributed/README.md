# distributed

分布式训练辅助层。

## 30秒定位

1. 需要 rank 间时间对齐  
用 `ClockSynchronizer`

2. 需要 etcd barrier  
用 `etcd_barrier`（可选依赖 `etcd3`）

3. 需要 sequence parallel padding  
用 `pad_for_sequence_parallel` / `remove_pad_by_value`（可选依赖 `megatron.core`）

## 最小示例

```python
from my_utils.distributed import ClockSynchronizer

sync = ClockSynchronizer()
offset = sync.sync_once()
print("clock offset us:", offset)
```

## 关键文件

- `clockSyncUtils.py`: `ClockSynchronizer`、`SocketClockSynchronizer`
- `etcd_utils.py`: `etcd_barrier`
- `pad.py`: sequence parallel padding helpers

## 注意

- `etcd` 与 `megatron` 能力是“可选导入”，缺依赖时不影响其它模块。  
