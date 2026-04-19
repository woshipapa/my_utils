# framework_playbook_samples

这组样例对应 `FRAMEWORK_INTEGRATION_PLAYBOOK_ZH.md`，目标是：
**只改少量变量，直接把 NSYS/NCU 套到常见训练/推理框架。**

## 1) 快速使用

NSYS（推荐先跑）：

```bash
bash my_utils/profiling/examples/framework_playbook_samples/nsys_torchtitan.sh
```

NCU（单 kernel 深挖）：

```bash
bash my_utils/profiling/examples/framework_playbook_samples/ncu_generic_wrap.sh -- \
  torchrun --nproc_per_node=8 train.py ...
```

## 2) 框架脚本映射

- TorchTitan: `nsys_torchtitan.sh`
- Megatron-LM: `nsys_megatron.sh`
- DeepSpeed: `nsys_deepspeed.sh`
- HF Trainer: `nsys_hf_trainer.sh`
- VERL: `nsys_verl.sh`
- SLIME: `nsys_slime.sh`
- ROLL: `nsys_roll.sh`
- SGLang: `nsys_sglang.sh`
- vLLM: `nsys_vllm.sh`

## 3) YAML 模板

- NSYS YAML 模板: `nsys_framework_template.yaml`
- NCU YAML 模板: `ncu_framework_template.yaml`

YAML 用法：

```bash
python my_utils/profiling/templates/run_nsys_quick_yaml.py \
  --config my_utils/profiling/examples/framework_playbook_samples/nsys_framework_template.yaml
```

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/examples/framework_playbook_samples/ncu_framework_template.yaml
```

## 4) 注意事项

1. 这些脚本默认只是模板，先改 `*_WORKDIR` 和模型/配置路径。
2. `run_nsys_quick.sh` 必须包在最外层，保证 `torchrun`/`deepspeed` 子进程也被采集。
3. 若用 `capture_range=cudaProfilerApi`，请确保 start/stop 在真正执行 CUDA 的 worker 进程里调用。
