# framework_playbook_samples

Companion samples for `../../docs/FRAMEWORK_INTEGRATION_PLAYBOOK.md`. Goal:
apply NSYS/NCU to common training/inference frameworks by changing only a few
variables.

## 1) Quick use

NSYS (start here):

```bash
bash my_utils/profiling/examples/framework_playbook_samples/nsys_torchtitan.sh
```

NCU (single-kernel deep dive):

```bash
bash my_utils/profiling/examples/framework_playbook_samples/ncu_generic_wrap.sh -- \
  torchrun --nproc_per_node=8 train.py ...
```

## 2) Framework-to-script map

- TorchTitan: `nsys_torchtitan.sh`
- Megatron-LM: `nsys_megatron.sh`
- DeepSpeed: `nsys_deepspeed.sh`
- HF Trainer: `nsys_hf_trainer.sh`
- VERL: `nsys_verl.sh`
- SLIME: `nsys_slime.sh`
- ROLL: `nsys_roll.sh`
- SGLang: `nsys_sglang.sh`
- vLLM: `nsys_vllm.sh`

## 3) YAML templates

- NSYS: `nsys_framework_template.yaml`
- NCU: `ncu_framework_template.yaml`

Usage:

```bash
python my_utils/profiling/templates/run_nsys_quick_yaml.py \
  --config my_utils/profiling/examples/framework_playbook_samples/nsys_framework_template.yaml
```

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/examples/framework_playbook_samples/ncu_framework_template.yaml
```

## 4) Notes

1. These scripts are templates: set `*_WORKDIR` and the model/config paths
   first.
2. `run_nsys_quick.sh` must wrap the outermost command so `torchrun` /
   `deepspeed` child processes are captured too.
3. With `capture_range=cudaProfilerApi`, make sure start/stop is called in
   the worker process that actually executes CUDA.

---

Chinese original: [docs/zh/profiling/examples/framework_playbook_samples/README.md](../../../../docs/zh/profiling/examples/framework_playbook_samples/README.md)
