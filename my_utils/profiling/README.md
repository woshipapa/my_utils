# profiling

这个目录现在可以按两条主线使用：

1. `nsys`：训练期采集 + sqlite 离线分析。
2. `ncu`：单 kernel 深度分析（metrics / section）。

## 1) 最快上手（训练采集）

### nsys（推荐先用）

```bash
# 直接包住你的训练命令
bash my_utils/profiling/templates/run_nsys_quick.sh -- python train.py --config cfg.yaml
```

或用 YAML（参数更可控）：

```bash
python my_utils/profiling/templates/run_nsys_quick_yaml.py \
  --config my_utils/profiling/templates/nsys_quick_launch.yaml
```

如果你要“全量参数模板（逐项注释）”：

```bash
python my_utils/profiling/templates/run_nsys_quick_yaml.py \
  --config my_utils/profiling/templates/nsys_2026_2_full_args.yaml
```

### ncu（新增）

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_quick_launch.yaml
```

全量参数模板（逐项注释）：

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_2026_1_1_full_args.yaml
```

## 2) 离线分析（nsys sqlite）

```bash
myutils-profile nsys-analyze --sqlite ./train_rank0.sqlite --output ./nsys_analyze.json
myutils-profile nsys-export --sqlite ./train_rank0.sqlite --format csv --output ./kernels.csv
myutils-profile nsys-diff --before-sqlite ./a.sqlite --after-sqlite ./b.sqlite --output ./diff.json
myutils-profile nsys-timeline-html --sqlite ./train_rank0.sqlite --output ./timeline.html
```

## 3) 离线分析（ncu csv）

```bash
myutils-profile ncu-csv-skill --csv ./ncu_raw.csv --list-skills --pretty
myutils-profile ncu-csv-skill --csv ./ncu_raw.csv --skill summary --pretty
myutils-profile ncu-csv-analyze --csv ./ncu_raw.csv --top-k 20 --pretty
```

## 4) 离线分析（ncu .ncu-rep 直读）

```bash
myutils-profile ncu-report-skill --report ./run.ncu-rep --list-skills --pretty
myutils-profile ncu-report-skill --report ./run.ncu-rep --skill per_metric_stats --pretty
myutils-profile ncu-report-skill --report ./run.ncu-rep --skill bottleneck_report --param top_k=10 --pretty
myutils-profile ncu-report-analyze --report ./run.ncu-rep --top-k 20 --pretty
```

## 5) 你主要会改的文件

- `my_utils/profiling/templates/nsys_quick_launch.yaml`
- `my_utils/profiling/templates/nsys_2026_2_full_args.yaml`
- `my_utils/profiling/ncu/ncu_quick_launch.yaml`
- `my_utils/profiling/ncu/ncu_2026_1_1_full_args.yaml`

## 6) 详细文档入口

- `my_utils/profiling/templates/README.md`
- `my_utils/profiling/docs/README.md`
- `my_utils/profiling/ncu/README.md`
