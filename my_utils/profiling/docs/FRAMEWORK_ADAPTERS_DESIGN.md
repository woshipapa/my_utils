# 框架适配器设计方案

## 设计目标

1. **零侵入集成** - 不修改框架源码即可使用
2. **自动检测** - 自动识别当前使用的框架
3. **统一接口** - 所有适配器遵循相同的接口
4. **按需加载** - 只加载当前使用的框架适配器

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    FrameworkRegistry                         │
│  注册表，管理所有可用适配器                                   │
│  - 自动检测框架                                              │
│  - 按需加载适配器                                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Megatron     │   │  DeepSpeed    │   │ HuggingFace  │
│   Adapter     │   │   Adapter     │   │   Adapter     │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ - 模型并行    │   │ - ZeRO stages │   │ - Trainer     │
│ - 数据并行    │   │ - Offload     │   │ - Accelerate │
│ - 张量并行    │   │ - Pipeline    │   │ - PEFT        │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                  ┌──────────────────┐
                  │  BaseAdapter     │
                  │  - 通用接口      │
                  │  - 通用方法      │
                  └──────────────────┘
                              │
                              ▼
                  ┌──────────────────┐
                  │ MetricsCollector │
                  │ & Analyzer       │
                  └──────────────────┘
```

## 1. 适配器基类

```python
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import MetricsCollector

class FrameworkInfo:
    """框架信息"""
    name: str
    version: str | None
    detected: bool
    import_path: str

class FrameworkAdapter(ABC):
    """
    框架适配器基类

    所有框架适配器必须实现此接口
    """

    # 子类应该覆盖这些
    FRAMEWORK_NAME: str = "base"
    IMPORT_PATHS: list[str] = []  # 用于检测的导入路径

    def __init__(self, metrics_collector: "MetricsCollector"):
        self.collector = metrics_collector
        self._enabled = False
        self._hooks: dict[str, Callable] = {}

    @classmethod
    def detect(cls) -> FrameworkInfo:
        """
        检测框架是否可用

        Returns:
            FrameworkInfo: 框架信息
        """
        for path in cls.IMPORT_PATHS:
            try:
                __import__(path)
                return FrameworkInfo(
                    name=cls.FRAMEWORK_NAME,
                    version=cls._get_version(path),
                    detected=True,
                    import_path=path,
                )
            except ImportError:
                continue

        return FrameworkInfo(
            name=cls.FRAMEWORK_NAME,
            version=None,
            detected=False,
            import_path="",
        )

    @classmethod
    def _get_version(cls, import_path: str) -> str | None:
        """获取框架版本"""
        try:
            mod = __import__(import_path)
            version = getattr(mod, "__version__", None)
            if version is None:
                # 尝试从version模块获取
                try:
                    version_mod = __import__(f"{import_path}.version")
                    version = getattr(version_mod, "VERSION", None)
                except (ImportError, AttributeError):
                    pass
            return str(version) if version else None
        except Exception:
            return None

    @abstractmethod
    def setup(self) -> None:
        """
        设置适配器

        应该：
        1. 注册相关的MetricsProviders
        2. 注册训练循环hooks
        3. 设置框架特定的配置
        """
        pass

    @abstractmethod
    def get_training_loop_hooks(self) -> dict[str, Callable]:
        """
        获取训练循环的hook函数

        Returns:
            dict: {hook_name: hook_function}
                  例如: {"pre_forward": self._pre_forward_hook}
        """
        pass

    def get_forward_pre_hook(self) -> Callable | None:
        """获取forward前钩子"""
        return None

    def get_forward_post_hook(self) -> Callable | None:
        """获取forward后钩子"""
        return None

    def get_backward_pre_hook(self) -> Callable | None:
        """获取backward前钩子"""
        return None

    def get_backward_post_hook(self) -> Callable | None:
        """获取backward后钩子"""
        return None

    def cleanup(self) -> None:
        """清理资源，移除hooks"""
        for hook_id in self._hooks.values():
            if hasattr(hook_id, "remove"):
                hook_id.remove()
        self._hooks.clear()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *args):
        self.cleanup()
```

## 2. Megatron适配器

```python
class MegatronAdapter(FrameworkAdapter):
    """Megatron-LM框架适配器"""

    FRAMEWORK_NAME = "Megatron"
    IMPORT_PATHS = ["megatron", "megatron.core", "nvidia"]

    def __init__(self, metrics_collector: "MetricsCollector"):
        super().__init__(metrics_collector)
        self._parallel_state = None
        self._model_parallel_size = None
        self._tensor_model_parallel_size = None
        self._pipeline_model_parallel_size = None

    def setup(self) -> None:
        """设置Megatron适配器"""
        try:
            from megatron.core import parallel_state
            self._parallel_state = parallel_state

            # 获取并行配置
            self._tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
            self._pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
            self._model_parallel_size = self._tensor_model_parallel_size * self._pipeline_model_parallel_size

            # 注册provider
            self._register_providers()

            self._enabled = True
            logger.info(f"Megatron适配器已启用 (TP={self._tensor_model_parallel_size}, PP={self._pipeline_model_parallel_size})")

        except Exception as e:
            logger.warning(f"Megatron适配器设置失败: {e}")

    def _register_providers(self) -> None:
        """注册Megatron特定的MetricsProviders"""
        from ..providers.megatron import (
            MegatronTimerProvider,
            MegatronMemoryProvider,
            MegatronCommunicationProvider,
            MegatronParallelProvider,
        )

        # 计时provider
        self.collector.register_provider(MegatronTimerProvider(
            parallel_state=self._parallel_state,
        ))

        # 内存provider
        self.collector.register_provider(MegatronMemoryProvider(
            parallel_state=self._parallel_state,
        ))

        # 通信provider
        self.collector.register_provider(MegatronCommunicationProvider(
            parallel_state=self._parallel_state,
        ))

        # 并行效率provider
        self.collector.register_provider(MegatronParallelProvider(
            parallel_state=self._parallel_state,
        ))

    def get_training_loop_hooks(self) -> dict[str, Callable]:
        """获取训练循环hooks"""
        return {
            "forward_step_start": self._on_forward_step_start,
            "forward_step_end": self._on_forward_step_end,
            "training_step_end": self._on_training_step_end,
            "validation_start": self._on_validation_start,
            "validation_end": self._on_validation_end,
        }

    def _on_forward_step_start(self, model, input_tensor, losses_reduced) -> None:
        """Forward步骤开始时的hook"""
        # 标记NVTX范围
        from my_utils import create_labeler
        labeler = create_labeler()
        labeler.push("megatron.forward_step")

        # 记录输入tensor大小
        if hasattr(input_tensor, "size"):
            self.collector.record_event("input_size", input_tensor.size())

    def _on_forward_step_end(self, model, input_tensor, output_tensor, losses_reduced) -> None:
        """Forward步骤结束时的hook"""
        from my_utils import create_labeler
        labeler = create_labeler()
        labeler.pop()

        # 触发指标收集
        self.collector.collect()

    def _on_training_step_end(self, iteration, losses_reduced) -> None:
        """训练步骤结束时的hook"""
        # 记录损失
        if losses_reduced is not None:
            self.collector.record_event("loss", losses_reduced.item())

    def _on_validation_start(self) -> None:
        """验证开始时的hook"""
        from my_utils import create_labeler
        labeler = create_labeler()
        labeler.push("megatron.validation")

    def _on_validation_end(self) -> None:
        """验证结束时的hook"""
        from my_utils import create_labeler
        labeler = create_labeler()
        labeler.pop()

    def get_model_forward_hook(self) -> Callable:
        """获取模型forward hook"""
        def forward_hook(module, input, output):
            # 记录每层的输出大小
            self.collector.record_event(
                f"layer.{module.__class__.__name__}.output_size",
                output.size() if hasattr(output, "size") else None,
            )
        return forward_hook

    def attach_model_hooks(self, model) -> None:
        """给模型附加hooks"""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                hook_id = module.register_forward_hook(self.get_model_forward_hook())
                self._hooks[f"module_{name}"] = hook_id
```

## 3. DeepSpeed适配器

```python
class DeepSpeedAdapter(FrameworkAdapter):
    """DeepSpeed框架适配器"""

    FRAMEWORK_NAME = "DeepSpeed"
    IMPORT_PATHS = ["deepspeed"]

    def __init__(self, metrics_collector: "MetricsCollector"):
        super().__init__(metrics_collector)
        self._engine = None
        self._config = None
        self._zero_stage = None

    def setup(self) -> None:
        """设置DeepSpeed适配器"""
        try:
            import deepspeed
            from deepspeed.utils import logger as ds_logger

            # 尝试获取当前初始化的engine
            # 注意：DeepSpeed通常在训练脚本中初始化，这里需要延迟初始化
            self._enabled = True
            logger.info("DeepSpeed适配器已启用")

        except Exception as e:
            logger.warning(f"DeepSpeed适配器设置失败: {e}")

    def set_engine(self, engine) -> None:
        """设置DeepSpeed engine（在初始化后调用）"""
        self._engine = engine
        self._config = engine.deepspeed_config
        self._zero_stage = self._config.get("zero_optimization", {}).get("stage", 0)

        logger.info(f"DeepSpeed引擎已设置 (ZERO Stage={self._zero_stage})")

        # 注册provider
        self._register_providers()

    def _register_providers(self) -> None:
        """注册DeepSpeed特定的MetricsProviders"""
        from ..providers.deepspeed import (
            DeepSpeedMemoryProvider,
            DeepSpeedZeroProvider,
            DeepSpeedOffloadProvider,
        )

        # ZeRO内存provider
        self.collector.register_provider(DeepSpeedMemoryProvider(
            engine=self._engine,
            zero_stage=self._zero_stage,
        ))

        # Offload provider
        if self._is_offload_enabled():
            self.collector.register_provider(DeepSpeedOffloadProvider(
                engine=self._engine,
            ))

    def _is_offload_enabled(self) -> bool:
        """检查是否启用了offload"""
        zero_config = self._config.get("zero_optimization", {})
        return any(
            zero_config.get(f"offload_{kind}", {}).get("device", "none") != "none"
            for kind in ["optimizer", "param"]
        )

    def get_training_loop_hooks(self) -> dict[str, Callable]:
        """获取训练循环hooks"""
        return {
            "pre_training_step": self._pre_training_step,
            "post_training_step": self._post_training_step,
            "pre_micro_batch": self._pre_micro_batch,
            "post_micro_batch": self._post_micro_batch,
        }

    def _pre_training_step(self) -> None:
        """训练步骤前的hook"""
        from my_utils import create_labeler
        labeler = create_labeler()
        labeler.push("deepspeed.training_step")

    def _post_training_step(self, loss) -> None:
        """训练步骤后的hook"""
        from my_utils import create_labeler
        labeler = create_labeler()
        labeler.pop()

        # 记录loss
        self.collector.record_event("loss", loss.item())

        # 收集指标
        self.collector.collect()

    def _pre_micro_batch(self, micro_batch_idx) -> None:
        """Micro batch前的hook"""
        self.collector.record_event("micro_batch_start", micro_batch_idx)

    def _post_micro_batch(self, micro_batch_idx) -> None:
        """Micro batch后的hook"""
        self.collector.record_event("micro_batch_end", micro_batch_idx)

    def get_gradient_accumulation_hooks(self) -> dict[str, Callable]:
        """获取梯度累积相关hooks"""
        return {
            "pre_accumulation": self._pre_gradient_accumulation,
            "post_accumulation": self._post_gradient_accumulation,
        }

    def _pre_gradient_accumulation(self) -> None:
        """梯度累积前"""
        self.collector.record_event("grad_accum_start", True)

    def _post_gradient_accumulation(self) -> None:
        """梯度累积后"""
        # 记录梯度统计
        if self._engine is not None:
            grad_norm = self._engine.get_global_grad_norm()
            self.collector.record_event("grad_norm", grad_norm)
```

## 4. HuggingFace适配器

```python
class HuggingFaceAdapter(FrameworkAdapter):
    """HuggingFace框架适配器（支持Trainer和Accelerate）"""

    FRAMEWORK_NAME = "HuggingFace"
    IMPORT_PATHS = ["transformers", "accelerate"]

    def __init__(self, metrics_collector: "MetricsCollector"):
        super().__init__(metrics_collector)
        self._use_trainer = False
        self._use_accelerate = None
        self._trainer = None

    def setup(self) -> None:
        """设置HuggingFace适配器"""
        try:
            import transformers
            import accelerate

            # 检测使用的是Trainer还是Accelerate
            self._use_accelerate = accelerate

            self._enabled = True
            logger.info("HuggingFace适配器已启用")

        except Exception as e:
            logger.warning(f"HuggingFace适配器设置失败: {e}")

    def setup_trainer(self, trainer) -> None:
        """设置Trainer适配（在Trainer初始化后调用）"""
        self._trainer = trainer
        self._use_trainer = True

        # 注册Trainer回调
        from .callbacks import MetricsCollectorCallback
        trainer.add_callback(MetricsCollectorCallback(self.collector))

        # 注册providers
        self._register_trainer_providers()

        logger.info("HuggingFace Trainer适配已设置")

    def _register_trainer_providers(self) -> None:
        """注册Trainer特定的Providers"""
        from ..providers.huggingface import (
            HFTrainerMetricsProvider,
            HFLoggingProvider,
        )

        self.collector.register_provider(HFTrainerMetricsProvider(
            trainer=self._trainer,
        ))

        # 如果使用自定义日志，可以记录日志指标
        self.collector.register_provider(HFLoggingProvider(
            trainer=self._trainer,
        ))

    def setup_accelerate(self) -> None:
        """设置Accelerate适配"""
        if self._use_accelerate is None:
            return

        from ..providers.huggingface import HFAccelerateProvider
        self.collector.register_provider(HFAccelerateProvider())

        logger.info("HuggingFace Accelerate适配已设置")

    def get_training_loop_hooks(self) -> dict[str, Callable]:
        """获取训练循环hooks"""
        if self._use_trainer:
            # Trainer使用callback机制，这里返回空
            return {}
        elif self._use_accelerate:
            return {
                "pre_step": self._accelerate_pre_step,
                "post_step": self._accelerate_post_step,
            }
        return {}

    def _accelerate_pre_step(self, model, optimizer, input_batch) -> None:
        """Accelerate训练步骤前"""
        from my_utils import create_labeler
        labeler = create_labeler()
        labeler.push("hf.training_step")

    def _accelerate_post_step(self, model, optimizer, loss) -> None:
        """Accelerate训练步骤后"""
        from my_utils import create_labeler
        labeler = create_labeler()
        labeler.pop()

        self.collector.record_event("loss", loss.item())
        self.collector.collect()

    def get_model_forward_hooks(self, model) -> dict[str, Callable]:
        """获取模型forward hooks"""
        hooks = {}

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                def make_hook(module_name):
                    def hook(module, input, output):
                        self.collector.record_event(
                            f"hf.layer.{module_name}.output",
                            output,
                        )
                    return hook

                hooks[name] = make_hook(name)

        return hooks
```

## 5. 框架注册表

```python
class FrameworkRegistry:
    """
    框架注册表

    管理所有可用的框架适配器
    """

    _adapters: dict[str, type[FrameworkAdapter]] = {}
    _instances: dict[str, FrameworkAdapter] = {}

    @classmethod
    def register(cls, adapter_class: type[FrameworkAdapter]) -> None:
        """注册适配器类"""
        cls._adapters[adapter_class.FRAMEWORK_NAME] = adapter_class

    @classmethod
    def detect_all(cls) -> list[FrameworkInfo]:
        """检测所有可用框架"""
        return [
            adapter_class.detect()
            for adapter_class in cls._adapters.values()
        ]

    @classmethod
    def get_adapter(cls, name: str, metrics_collector: "MetricsCollector") -> FrameworkAdapter:
        """获取适配器实例"""
        if name not in cls._instances:
            adapter_class = cls._adapters.get(name)
            if adapter_class is None:
                raise ValueError(f"Unknown adapter: {name}")

            instance = adapter_class(metrics_collector)
            cls._instances[name] = instance

        return cls._instances[name]

    @classmethod
    def auto_setup(cls, metrics_collector: "MetricsCollector") -> list[FrameworkAdapter]:
        """自动检测并设置所有可用适配器"""
        detected = cls.detect_all()
        setup_adapters = []

        for info in detected:
            if info.detected:
                adapter = cls.get_adapter(info.name, metrics_collector)
                try:
                    adapter.setup()
                    setup_adapters.append(adapter)
                    logger.info(f"已设置 {info.name} 适配器 (版本: {info.version})")
                except Exception as e:
                    logger.warning(f"{info.name} 适配器设置失败: {e}")

        return setup_adapters

    @classmethod
    def cleanup_all(cls) -> None:
        """清理所有适配器"""
        for adapter in cls._instances.values():
            adapter.cleanup()
        cls._instances.clear()

# 注册内置适配器
FrameworkRegistry.register(MegatronAdapter)
FrameworkRegistry.register(DeepSpeedAdapter)
FrameworkRegistry.register(HuggingFaceAdapter)
```

## 6. Provider示例

```python
# my_utils/profiling/providers/megatron.py

class MegatronCommunicationProvider(MetricsProvider):
    """Megatron通信指标Provider"""

    provider_id = "megatron_comm"

    def __init__(self, parallel_state):
        self.parallel_state = parallel_state
        self._enabled = True
        self._events = []

    def get_metrics(self) -> list[MetricEvent]:
        """获取通信指标"""
        return self._events.copy()

    def record_allreduce(self, tensor_size: int, duration_ms: float) -> None:
        """记录allreduce操作"""
        # 计算带宽
        bandwidth_gb_s = (tensor_size * 4) / (duration_ms / 1000) / 1e9  # 假设float32

        self._events.append(MetricEvent(
            timestamp=time.time(),
            name="comm.allreduce",
            value=duration_ms,
            unit="ms",
            tags={
                "tensor_size": str(tensor_size),
                "bandwidth_gb_s": f"{bandwidth_gb_s:.2f}",
                "comm_type": "allreduce",
            }
        ))

class MegatronParallelProvider(MetricsProvider):
    """Megatron并行效率Provider"""

    provider_id = "megatron_parallel"

    def __init__(self, parallel_state):
        self.parallel_state = parallel_state
        self._enabled = True
        self._step_metrics = []

    def record_step_metrics(
        self,
        iteration_time: float,
        compute_time: float,
        comm_time: float,
    ) -> None:
        """记录步骤级别的并行效率指标"""
        # 计算并行效率
        efficiency = compute_time / (compute_time + comm_time) if (compute_time + comm_time) > 0 else 0

        self._step_metrics.append(MetricEvent(
            timestamp=time.time(),
            name="parallel.efficiency",
            value=efficiency,
            unit="ratio",
            tags={
                "compute_time_ms": str(compute_time),
                "comm_time_ms": str(comm_time),
                "overlap_ratio": str(comm_time / iteration_time if iteration_time > 0 else 0),
            }
        ))

    def get_metrics(self) -> list[MetricEvent]:
        return self._step_metrics.copy()
```

## 7. 回调示例

```python
# my_utils/profiling/callbacks.py

from transformers import TrainerCallback

class MetricsCollectorCallback(TrainerCallback):
    """Transformers Trainer回调，自动收集指标"""

    def __init__(self, metrics_collector):
        self.collector = metrics_collector
        from my_utils import create_labeler
        self.labeler = create_labeler()

    def on_step_begin(self, args, state, control, **kwargs):
        """步骤开始"""
        self.labeler.push("hf.step")
        self.collector.record_event("step_begin", state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        """步骤结束"""
        self.labeler.pop()

        # 记录损失
        if isinstance(state.log_history, list) and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                self.collector.record_event("loss", last_log["loss"])

        # 收集指标
        self.collector.collect(step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估时"""
        self.labeler.push("hf.evaluation")

    def on_prediction_step(self, args, state, control, **kwargs):
        """预测步骤"""
        pass
```

## 8. 使用示例

### 8.1 自动检测和设置

```python
from my_utils.profiling import (
    MetricsCollector,
    FrameworkRegistry,
)

# 创建collector
collector = MetricsCollector(output_dir="./metrics_logs")

# 自动检测并设置所有可用适配器
adapters = FrameworkRegistry.auto_setup(collector)

# 输出检测到的框架
for adapter in adapters:
    info = adapter.detect()
    print(f"检测到: {info.name} v{info.version}")

# 如果需要手动设置特定适配器
if adapters and adapters[0].FRAMEWORK_NAME == "Megatron":
    adapters[0].setup()
```

### 8.2 Megatron集成

```python
# 在Megatron训练脚本中

from my_utils.profiling import MetricsCollector, FrameworkRegistry

# 初始化
collector = MetricsCollector()
megatron_adapter = FrameworkRegistry.get_adapter("Megatron", collector)
megatron_adapter.setup()

# 获取hooks
hooks = megatron_adapter.get_training_loop_hooks()

# 在训练循环中使用
for iteration in range(...):
    hooks["forward_step_start"](model, input_tensor, None)

    output = model(input_tensor)
    loss = criterion(output, target)

    hooks["forward_step_end"](model, input_tensor, output, loss)

# 清理
megatron_adapter.cleanup()
```

### 8.3 HuggingFace Trainer集成

```python
# 在HuggingFace训练脚本中

from transformers import Trainer
from my_utils.profiling import MetricsCollector, FrameworkRegistry

collector = MetricsCollector()
hf_adapter = FrameworkRegistry.get_adapter("HuggingFace", collector)
hf_adapter.setup()

# 标准Trainer初始化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # ...
)

# 设置适配器
hf_adapter.setup_trainer(trainer)

# 正常训练
trainer.train()
```

### 8.4 DeepSpeed集成

```python
# 在DeepSpeed训练脚本中

import deepspeed
from my_utils.profiling import MetricsCollector, FrameworkRegistry

collector = MetricsCollector()
ds_adapter = FrameworkRegistry.get_adapter("DeepSpeed", collector)
ds_adapter.setup()

# 初始化DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=optimizer,
)

# 设置engine
ds_adapter.set_engine(model_engine)

# 训练循环
hooks = ds_adapter.get_training_loop_hooks()
for step, batch in enumerate(data_loader):
    hooks["pre_training_step"]()

    loss = model_engine(batch)

    hooks["post_training_step"](loss)
```

## 9. 配置文件支持

```yaml
# framework_adapters_config.yaml

adapters:
  megatron:
    enabled: true
    auto_detect: true
    providers:
      timer: true
      memory: true
      communication: true
      parallel_efficiency: true

  deepspeed:
    enabled: true
    auto_detect: true
    providers:
      zero_memory: true
      offload: true

  huggingface:
    enabled: true
    auto_detect: true
    use_trainer_callback: true
    providers:
      trainer_metrics: true
      logging: true

auto_setup: true  # 自动检测并设置
```

```python
# 从配置加载
import yaml

with open("framework_adapters_config.yaml") as f:
    config = yaml.safe_load(f)

collector = MetricsCollector()

if config.get("auto_setup"):
    adapters = FrameworkRegistry.auto_setup(collector)
else:
    # 手动选择
    adapters = []
    for name, cfg in config["adapters"].items():
        if cfg.get("enabled", False):
            adapter = FrameworkRegistry.get_adapter(name.capitalize(), collector)
            adapter.setup()
            adapters.append(adapter)
```

## 总结

框架适配器系统通过以下方式实现框架无感知：

1. **统一接口** - 所有适配器继承BaseAdapter
2. **自动检测** - FrameworkRegistry自动发现可用框架
3. **延迟初始化** - 只加载实际使用的框架
4. **零侵入** - 通过hooks和callbacks集成，不修改源码

下一步：
1. 实现基础适配器（Megatron、DeepSpeed、HuggingFace）
2. 扩展更多框架适配器（PyTorch Lightning、Determined AI等）
3. 完善Provider实现
4. 添加更多示例和文档
