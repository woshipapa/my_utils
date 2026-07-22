# artifacts

离线产物层：中间数据落盘/回读，以及 NCU CSV 分析辅助。

## 30秒定位

1. 我想把 tensor/中间结果落盘  
用 `UniversalDumper` / `DumpConfig`

2. 我想做 NCU CSV 指标分析对比  
用 `analyze_sm_throughput_from_csv` / `compare_kernel_metrics`

## 最小示例

```python
from my_utils.artifacts import DumpConfig, UniversalDumper

cfg = DumpConfig(output_dir="./dump_out")
dumper = UniversalDumper(cfg)
dumper.dump_tensor("x", x_tensor)
```

## 关键文件

- `dump_utils.py`: `DumpTensorIO`、`DumpConfig`、`UniversalDumper`、`UniversalLoader`
- `ncu_analyze_from_csv.py`: NCU CSV 指标分析与对比工具  
