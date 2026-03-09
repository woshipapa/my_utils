# artifacts

## 作用
`artifacts` 负责中间产物的落盘/回读与离线分析输入（如 CSV）。

## 文件
- `dump_utils.py`: `DumpTensorIO`、`DumpConfig`、`UniversalDumper`、`UniversalLoader`。
- `ncu_analyze_from_csv.py`: NCU CSV 指标分析与对比。

## 常用导入
```python
from my_utils.artifacts.dump_utils import DumpConfig, UniversalDumper, get_dumper
from my_utils.artifacts.ncu_analyze_from_csv import analyze_sm_throughput_from_csv
```

## 说明
- 这里偏离线能力，适合训练后批量分析与回归对比。
