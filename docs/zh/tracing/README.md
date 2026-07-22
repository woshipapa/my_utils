# tracing

追踪标注层，当前核心是 NVTX labeler 抽象与自动降级。

## 30秒定位

1. 有 NVTX 环境，想加 range 标注  
用 `create_labeler(preferred="nvtx")`

2. 没有 NVTX 也希望代码不改  
用 `create_labeler(preferred="auto")`（自动 NoOp 降级）

## 最小示例

```python
from my_utils.tracing import create_labeler

labeler = create_labeler(preferred="auto")
with labeler.range("forward"):
    # ... forward ...
    pass
```

## 关键文件

- `nvtx_utils.py`: `LabelerProtocol`、`NoOpLabeler`、`NvtxLabeler`、`TorchNvtxLabeler`、`create_labeler`  
