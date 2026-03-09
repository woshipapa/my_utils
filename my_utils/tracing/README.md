# tracing

## 作用
`tracing` 负责 trace 标注能力，当前核心是 NVTX labeler 抽象和后端选择。

## 文件
- `nvtx_utils.py`: `LabelerProtocol`、`NoOpLabeler`、`NvtxLabeler`、`TorchNvtxLabeler`、`create_labeler`。

## 常用导入
```python
from my_utils.tracing.nvtx_utils import create_labeler, NoOpLabeler
```

## 说明
- 推荐统一通过 `create_labeler(...)` 获取实现，便于在无 NVTX 环境自动降级。
