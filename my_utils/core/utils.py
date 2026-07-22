# SPDX-License-Identifier: Apache-2.0
import torch
import sys
import torch.distributed as dist
import time
from typing import Any, Optional

# from megatron.core.tensor_parallel.mappings import (
#     gather_from_tensor_model_parallel_region,
# )
import hashlib
import torch.nn as nn

from contextlib import contextmanager
import logging


def get_logger() -> logging.Logger:
    return logging.getLogger("my_utils")


def print_model_params(model):
    print("Model Parameters:")
    print("=" * 50)
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor):
            print(f"Layer: {name}")
            print(f"Shape: {param.shape}")
            print(param.data)  # 仅打印数值，不计算梯度
            print("-" * 50)


def tensor_md5(tensor: torch.Tensor) -> str:
    tensor = tensor.to(torch.float64)
    # 确保 Tensor 在 CPU 上，并转换为 numpy 数组
    tensor_np = tensor.detach().cpu().numpy()
    # 将 numpy 数组转换为 bytes
    tensor_bytes = tensor_np.tobytes()
    # 计算 MD5
    md5_hash = hashlib.md5(tensor_bytes).hexdigest()
    return md5_hash


class DebugLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


filename = "_output_14_backward_2.log"


def register_hooks(model, print_values=True, max_elements=20):
    hooks = []

    def print_shape_and_values(tensor, label, max_elements=20):
        """Helper function to print shape and values of a tensor"""
        rank = 0
        with open(str(rank) + filename, "a") as f:
            if isinstance(tensor, torch.Tensor):
                get_logger().info(f"[rank {rank}] {label} shape: {tensor.shape}")
                if print_values:
                    if tensor.numel() < max_elements:
                        max_elements = tensor.numel()
                    if tensor.flatten()[0].dtype != torch.bool:
                        # get_logger().info(f"[rank {rank}]  {label} first 20 values: {tensor.flatten()[:max_elements]}{'...' if tensor.numel() > max_elements else ''} ")
                        # x, _ = torch.topk(tensor.flatten(), max_elements)
                        # get_logger().info(f"[rank {rank}]  {label} max 20 values: {x} ")
                        # x, _ = torch.topk(-tensor.flatten(), max_elements)
                        # get_logger().info(f"[rank {rank}]  {label} min 20 values: {x} ")
                        tensor = tensor.float()
                        x = torch.norm(tensor)
                        get_logger().info(f"[rank {rank}] {label} norm Values: {x} ")
                        # get_logger().info(f'[rank {rank}] {label} md5 value: {tensor_md5(tensor)} ')
                        get_logger().info(
                            f"[rank {rank}] {label} shape: {tensor.shape} "
                        )
            elif False:
                get_logger().info(f"[rank {rank}]  {label} is not tensor, is {tensor} ")

    def forward_hook_fn(module, input, output, name):
        """Hook 函数，打印输入和输出的 shape 和具体数值"""
        # print(f"Layer: {module.__class__.__name__}")
        rank = 0
        with open(str(rank) + filename, "a") as f:
            get_logger().info(f"[rank {rank}] Layer: {name}")
            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight.data
                print_shape_and_values(weight, f"{name}_Weight")
            elif False:
                get_logger().info(f"[rank {rank}] Layer: {name} has no weight ")
        # 打印输入
        if isinstance(input, tuple):
            for idx, inp in enumerate(input):
                if isinstance(inp, tuple):  # Check for nested tuple
                    for sub_idx, sub_inp in enumerate(inp):
                        print_shape_and_values(sub_inp, f"Input {idx}-{sub_idx}")
                else:
                    print_shape_and_values(inp, f"{name}_Input {idx}")
        else:
            print_shape_and_values(input, "Input")

        # Print output shapes and values
        if isinstance(output, tuple):
            for i, tensor in enumerate(output):
                if isinstance(tensor, tuple):  # Check for nested tuple
                    for sub_idx, sub_tensor in enumerate(tensor):
                        print_shape_and_values(
                            sub_tensor, f"{name}_Output {i}-{sub_idx}"
                        )
                else:
                    print_shape_and_values(tensor, f"{name}_Output {i}")

        else:
            print_shape_and_values(output, "Output")
        with open(str(rank) + filename, "a") as f:
            get_logger().info("-" * 100)
            get_logger().info(" ")

    def backward_hook_fn(module, grad_input, grad_output, name):
        """Hook 函数，打印反向传播时的梯度"""
        # print(f"Layer: {module.__class__.__name__} (backward)")
        rank = 0
        with open(str(rank) + filename, "a") as f:
            get_logger().info(f"[rank {rank}] Layer: {name} (backward)")
            # 打印输入梯度
            for idx, grad in enumerate(grad_input):
                if grad is not None:
                    print_shape_and_values(grad, f"{name}_Grad Input {idx}")

            # 打印输出梯度
            for idx, grad in enumerate(grad_output):
                if grad is not None:
                    print_shape_and_values(grad, f"{name}_Grad Output {idx}")

            # if hasattr(module, 'weight') and module.weight is not None:
            #     print_shape_and_values(module.weight.grad, f"{name}_weight_grad")
            # if hasattr(module, 'bias') and module.bias is not None:
            #     print_shape_and_values(module.bias.grad, f"{name}_bias_grad")

            get_logger().info("-" * 100)
            get_logger().info(" ")

    # 遍历模型的所有层并注册 hook
    # for layer in model.modules():
    #     if not isinstance(layer, torch.nn.Sequential) and not isinstance(layer, torch.nn.ModuleList):
    #         forward_hook = layer.register_forward_hook(forward_hook_fn)
    #         backward_hook = layer.register_backward_hook(backward_hook_fn)
    #         hooks.append(forward_hook)
    #         hooks.append(backward_hook)

    # if dist.is_available() and dist.is_initialized():
    #     rank = 0
    # else:
    #     rank = 0

    def watch_parameter(param_name, param):
        rank = 0

        def param_hook(grad):
            if grad is not None:
                with open(str(rank) + filename + "_grad", "a") as f:
                    get_logger().info(f"{param_name} grad norm: {grad.norm()} ")
            else:
                with open(str(rank) + filename + "_grad", "a") as f:
                    get_logger().info(f"{param_name} grad is None ")

        param.register_hook(param_hook)

    for name, module in model.named_modules():
        forward_hook = module.register_forward_hook(
            lambda m, i, o, name=name: forward_hook_fn(m, i, o, name)
        )
        backward_hook = module.register_full_backward_hook(
            lambda m, i, o, name=name: backward_hook_fn(m, i, o, name)
        )
        hooks.append(forward_hook)
        hooks.append(backward_hook)

    for name, param in model.named_parameters():
        if param.requires_grad:
            watch_parameter(name, param)

    return hooks  # 返回 hook 句柄列表，方便后续清理


import os
import re
from collections import defaultdict
import numpy as np
import logging

# from t2v_flow.executor.DynamicForwardStepHandler import DynamicForwardStepHandler
from logging import LoggerAdapter
from ..tracing.nvtx_utils import LabelerProtocol, NoOpLabeler, create_labeler

try:
    import psutil  # noqa: F401 -- availability probe

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MyTimer:
    # __init__, start, stop, next_iteration, _gather_records, summarize, summarize_per_rank, dump 等方法保持不变...
    # (此处省略了之前已展示的、未改动的方法代码，以保持简洁)
    def __init__(
        self,
        use_cuda=True,
        tag="timer",
        verbose=True,
        log_dir="my_timer_log/",
        profile_memory=False,
        use_nvtx=False,
        labeler: Optional[LabelerProtocol] = None,
        nvtx_domain: Optional[str] = None,
    ):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.tag = tag
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.log_dir = log_dir

        self.records = []
        self.current_iteration = 0

        self.profile_memory = profile_memory and PSUTIL_AVAILABLE and self.use_cuda
        self.log_context = {"log_type": "timing"}
        # self.logger = self._get_logger()

        # 1. 等待外部的配置参数传入或者2. 自己去主动寻找（耦合版）
        self.logger = None
        # if verbose: self.logger.setLevel(logging.INFO)
        # else: self.logger.setLevel(logging.WARNING)

        self.set_labeler(
            labeler
            if labeler is not None
            else create_labeler(
                enabled=bool(use_nvtx),
                default_domain=nvtx_domain,
            )
        )

        # async
        # 新增：用于存放本轮迭代中已完成但尚未计算时间的记录
        self._pending_records = []

        # self._stage_times = {}

        # --- [!!] 新的层级堆栈逻辑 [!!] ---

        # 1. (替换) 移除 self._stage_times = {}
        # 2. (新增) 我们需要一个唯一的节点 ID 来在
        #    'summarize' 阶段重建父子关系。
        self.next_node_id = 1  # 0 是 root

        # 3. (新增) 创建根节点 (root_node)。
        #    current_node 始终指向堆栈的顶部。
        self.root_node = {
            "name": "root",
            "node_id": 0,
            "parent_id": None,
            "start_cpu": time.perf_counter(),
            "children": [],  # (用于调试, 主要数据在 records 中)
        }

        self.current_node = self.root_node

    def _get_domain(self, domain_name: str):
        raise RuntimeError(
            "MyTimer no longer exposes raw NVTX domains; use register_stage() via the configured labeler."
        )

    def set_labeler(self, labeler: Optional[LabelerProtocol]) -> None:
        self.labeler = labeler if labeler is not None else NoOpLabeler()
        self.use_nvtx = bool(getattr(self.labeler, "enabled", False))

    def register_stage(
        self,
        stage_name: str,
        color: str = "blue",
        domain_name: str = None,
        category=None,
    ):
        if not self.use_nvtx:
            return
        self.labeler.register_label(
            stage_name,
            color=color,
            domain_name=domain_name,
            category=category,
        )
        return

        if domain_name is None:
            raise ValueError(
                "Pre-registration for NVTX optimization requires a valid `domain_name`."
            )

        domain = self._get_domain(domain_name)

        # nvtx.Domain的方法 get_event_attributes
        attrs = domain.get_event_attributes(
            message=stage_name, color=color, category=category
        )

        self._registered_attrs[stage_name] = (domain, attrs)

    def disable_cuda_time(self):
        self.use_cuda = False

    def set_logger(self, logger_instance: logging.Logger):
        """
        【公共接口】允许外部项目注入自己的 logger 实例。
        注入后会立即用 LoggerAdapter 包装，以支持过滤器。
        """
        # 即使外部注入，也用 Adapter 包装以确保 log_context 存在
        # logger_instance是global_logger.get_logger()返回的logger
        self.logger = LoggerAdapter(logger_instance, self.log_context)

        # 单例
        global_logger = GlobalLogger()

        setattr(self.logger, "log_profile_event", global_logger.log_profile_event)

    def _create_default_logger(self) -> logging.Logger:
        """
        【全新】按您 handler 项目的样式，创建一个功能完备的默认 logger。
        """
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        logger = logging.getLogger(f"MyTimer.default_rank_{self.rank}")

        # 如果已经配置过，则直接返回，防止重复添加 handler
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        logger.propagate = False

        formatter = logging.Formatter(
            f"[%(asctime)s] [Rank {self.rank}] [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. 配置控制台 Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 2. 配置文件 Handler (所有日志)
        main_log_file = os.path.join(self.log_dir, f"timer_rank_{self.rank}.log")
        main_file_handler = logging.FileHandler(main_log_file, mode="a")
        main_file_handler.setFormatter(formatter)
        logger.addHandler(main_file_handler)

        return logger

    # def _get_logger(self):
    #     handler = DynamicForwardStepHandler()
    #     if handler.logger is None:
    #         return None
    #     adapter = LoggerAdapter(handler.logger, self.log_context)
    #     return adapter

    def _ensure_logger(self):
        """
        【全新核心逻辑】即时解析 Logger，解决初始化时序问题。
        """
        # 如果 logger 已经被外部通过 set_logger 注入，则什么都不做
        if self.logger:
            return

        # 1. 优先尝试获取 handler 项目的 logger
        raw_logger = None
        try:
            from t2v_flow.executor.DynamicForwardStepHandler import (
                DynamicForwardStepHandler,
            )

            handler = DynamicForwardStepHandler()
            if handler.logger:
                raw_logger = handler.logger
        except (ImportError, AttributeError):
            # 导入失败或属性不存在，说明不在 handler 项目中，正常现象
            pass

        # 2. 如果没获取到，则创建系统初始化时创建的默认 global logger
        if raw_logger is None:
            from .logger import GlobalLogger

            raw_logger = GlobalLogger().get_logger()

        # 3. 无论来源如何，都用 LoggerAdapter 包装以添加上下文，使过滤器生效
        self.logger = LoggerAdapter(raw_logger, self.log_context)

    @contextmanager
    def time_stage(self, stage_name: str):
        """
        一个上下文管理器，用于方便、安全地对代码块进行计时。

        Args:
            stage_name (str): 要计时的阶段或代码块的名称。

        Usage:
            timer = MyTimer()
            with timer.time_stage('data_loading'):
                # your code to be timed
                time.sleep(1)
        """
        self.start(stage_name)
        try:
            yield
        finally:
            self.stop(stage_name)

    def start(self, stage_name: str, color: str = "blue", domain_name: str = None):
        # if self.logger is None:
        #     self.logger = self._get_logger()
        # 此处耦合了handler的logger在这里设定，目前已经迁移到初始化完megatron分布式环境后传入logger
        self._ensure_logger()
        if torch.distributed.is_initialized() and self.rank != dist.get_rank():
            self.rank = dist.get_rank()
        # entry = {"cpu_start": time.time()}
        # CPU code start time
        entry = {"cpu_start": time.perf_counter()}

        # GPU profile using cudaEvent
        if self.use_cuda:
            entry["cuda_start"] = torch.cuda.Event(enable_timing=True)
            entry["cuda_end"] = torch.cuda.Event(enable_timing=True)
            entry["cuda_start"].record()

        if self.use_nvtx:
            entry["nvtx_token"] = self.labeler.start(
                stage_name,
                color=color,
                domain_name=domain_name,
            )
            entry["nvtx_domain"] = None  # 用于在 stop 时判断调用哪个 end_range
            entry["nvtx_range_id"] = None

            if False:
                # 优化路径: 使用预缓存的 (domain, attrs)
                domain, attrs = self._registered_attrs[stage_name]
                entry["nvtx_domain"] = domain
                entry["nvtx_range_id"] = domain.start_range(attributes=attrs)
            elif False:
                # 探索路径: 即时创建
                entry["nvtx_range_id"] = None
                # 此时 entry["nvtx_domain"] 保持为 None，作为使用全局函数的标记

        new_node_id = self.next_node_id
        self.next_node_id += 1

        new_node = {
            "name": stage_name,
            "node_id": new_node_id,
            "parent_id": self.current_node["node_id"],
            "children": [],
            # (用于 stop() 方法的堆栈上浮)
            "parent": self.current_node,
            "abs_start_time": time.time(),
            # (存储你 'entry' 字典中的所有内容)
            "cpu_start": entry["cpu_start"],
            "cuda_start": entry.get("cuda_start"),
            "cuda_end": entry.get("cuda_end"),
            "nvtx_domain": entry.get("nvtx_domain"),
            "nvtx_range_id": entry.get("nvtx_range_id"),
            "nvtx_token": entry.get("nvtx_token"),
        }

        self.current_node["children"].append(new_node)
        self.current_node = new_node
        # self._stage_times[stage_name] = entry

    # def stop(self, stage_name):
    #     if stage_name not in self._stage_times:
    #         return
    #     entry = self._stage_times.pop(stage_name)
    #     # cpu_end = time.time()
    #     cpu_end = time.perf_counter()
    #     cpu_elapsed_ms = (cpu_end - entry.get("cpu_start", cpu_end)) * 1000
    #     cuda_elapsed_ms = None
    #     if self.use_cuda and "cuda_end" in entry:
    #         entry["cuda_end"].record()
    #         # torch.cuda.synchronize()
    #         # cuda_elapsed_ms = entry["cuda_start"].elapsed_time(entry["cuda_end"])

    #     if self.use_nvtx and "nvtx_range_id" in entry:
    #         domain = entry.get("nvtx_domain")
    #         range_id = entry.get("nvtx_range_id")

    #         if range_id is not None:
    #             if domain:
    #                 # 如果 start 是在 domain 对象上调用的，end 也必须在同一个对象上调用
    #                 domain.end_range(range_id)
    #             else:
    #                 # 如果 start 是用全局函数调用的，end 也必须用全局函数
    #                 nvtx.end_range(range_id)

    #      # 暂存记录，CPU 时间已知，GPU 事件已记录但时间未知
    #     pending_record = {
    #         "stage": stage_name,
    #         "cpu_duration_ms": (cpu_end - entry["cpu_start"]) * 1000,
    #         "cuda_events": (entry.get("cuda_start"), entry.get("cuda_end"))
    #     }
    #     self._pending_records.append(pending_record)
    #     # self.records.append(
    #     #     {
    #     #         "stage": stage_name,
    #     #         "rank": self.rank,
    #     #         "iteration": self.current_iteration,
    #     #         "cpu_duration_ms": cpu_elapsed_ms,
    #     #         "cuda_duration_ms": cuda_elapsed_ms,
    #     #     }
    #     # )

    #     # # 使用传入的logger来记录timer中的数据信息
    #     # if self.verbose:
    #     #     self.logger.info(
    #     #         f"[Iter {self.current_iteration}] Stage '{stage_name}': CPU {cpu_elapsed_ms:.3f}ms, CUDA {cuda_elapsed_ms or 0.0:.3f}ms"
    #     #     )
    def stop(self, stage_name: str):
        cpu_end = time.perf_counter()

        # [!!] (V2) 1. 堆栈检查 (最关键的部分) [!!]
        if self.current_node["name"] != stage_name:
            # (健壮性处理：如果名称不匹配, 我们尝试在堆栈中向上查找)
            # (这可以处理 "stop('A')" 自动关闭 "B" 的情况)

            print(
                f"TimerWarning: Mismatched stop call on Rank {self.rank}! "
                f"Expected to stop '{self.current_node['name']}' but got '{stage_name}'."
            )

            node_to_stop = self._find_node_in_stack(stage_name)

            if node_to_stop is None:
                print(
                    f"TimerError: Could not find active timer '{stage_name}' in the stack."
                )
                return

            # 如果找到了, 我们必须自动关闭所有子节点, 直到我们到达
            # 'node_to_stop'。
            while self.current_node != node_to_stop:
                print(
                    f"TimerWarning: Auto-stopping child '{self.current_node['name']}' "
                    f"due to explicit stop of ancestor '{stage_name}'."
                )
                # (传入 cpu_end, 因为这是唯一的 "stop" 时间)
                self._finalize_and_record_node(self.current_node, cpu_end)
                self.current_node = self.current_node["parent"]

        # [!!] (V2) 2. 最终确定并记录当前节点
        self._finalize_and_record_node(self.current_node, cpu_end)

        # [!!] (V2) 3. 上浮 (Ascend)
        # (我们 *总是* 上浮到当前节点的父节点)
        if self.current_node["parent"] is not None:
            self.current_node = self.current_node["parent"]
        else:
            print("TimerError: Attempted to stop root node?")

    def _find_node_in_stack(self, name: str):
        """(新增) 辅助函数: 从当前节点向上搜索堆栈"""
        temp_node = self.current_node
        while temp_node is not None and temp_node["name"] != "root":
            if temp_node["name"] == name:
                return temp_node
            temp_node = temp_node["parent"]
        return None

    def _finalize_and_record_node(self, node: dict, cpu_end: float):
        """(新增) 辅助函数: 包含你 'stop' 方法的所有核心逻辑"""

        # [!!] (V2) 这部分代码 *就是* 你 'stop' 方法的 90% [!!]

        # (来自你 'stop' 的逻辑)
        if self.use_cuda and "cuda_end" in node and node["cuda_end"] is not None:
            node["cuda_end"].record()

        if self.use_nvtx:
            self.labeler.stop(node.get("nvtx_token"))

        if self.use_nvtx and "nvtx_range_id" in node:
            domain = node.get("nvtx_domain")
            range_id = node.get("nvtx_range_id")
            if range_id is not None:
                if domain:
                    domain.end_range(range_id)
                else:
                    # print(f"Timer NVTX: Using global end_range for '{node['name']}'")
                    pass

        # (来自你 'stop' 的逻辑: 创建 pending_record)
        pending_record = {
            "stage": node["name"],
            "cpu_duration_ms": (cpu_end - node["cpu_start"]) * 1000,
            "cuda_events": (node.get("cuda_start"), node.get("cuda_end")),
            "abs_start_time": node.get("abs_start_time", 0.0),
            # [!!] (V2) 新增的层级数据 [!!]
            # (这允许 'summarize' 重建树)
            "node_id": node["node_id"],
            "parent_id": node["parent_id"],
        }
        self._pending_records.append(pending_record)

    def set_step(self, iteration: int):
        self.current_iteration = iteration

    def synchronize_and_log(self):
        """
        [V2 - 层级版本]
        在迭代结束时调用，执行同步，并完成三件事：
        1. 计算所有 pending records 的 CUDA 时间并存入 self.records。
        2. 从 self.records (扁平列表) 重建本次迭代的调用树。
        3. 遍历该树，计算 Self Time 并以层级（缩进）格式记录日志。
        """
        if self.use_cuda:
            torch.cuda.synchronize()

        # --- 步骤 1: 处理 _pending_records 并填充 self.records ---

        # 清空本次迭代的 records
        self.records = []

        for record in self._pending_records:
            cuda_elapsed_ms = None
            if self.use_cuda:
                start_event, end_event = record["cuda_events"]
                if start_event and end_event:
                    # 确保事件已准备好
                    try:
                        cuda_elapsed_ms = start_event.elapsed_time(end_event)
                    except torch.cuda.Error as e:
                        # (处理可能的 CUDA 错误)
                        self.logger.warning(
                            f"CUDA event error for {record['stage']}: {e}"
                        )

            abs_start_ts = record.get("abs_start_time", 0.0)

            # B. 确定耗时 (优先用 CUDA 耗时，如果是纯 CPU 操作则用 CPU 耗时)
            final_duration_ms = (
                cuda_elapsed_ms
                if cuda_elapsed_ms is not None
                else record["cpu_duration_ms"]
            )

            # C. 推算结束时间点 (Start + Duration)
            # 注意 ms 转 s
            abs_end_ts = abs_start_ts + (final_duration_ms / 1000.0)

            self.logger.log_profile_event(
                timestamp=abs_start_ts,
                step=self.current_iteration,
                event_name=record["stage"],
                event_type="START",
                metadata=f"node_id={record['node_id']}",
            )

            # E. 写入 END 事件
            self.logger.log_profile_event(
                timestamp=abs_end_ts,
                step=self.current_iteration,
                event_name=record["stage"],
                event_type="END",
                duration_ms=final_duration_ms,
                metadata=f"node_id={record['node_id']}",  # 可以记录更多 meta
            )
            full_record = {
                "stage": record["stage"],
                "rank": self.rank,
                "iteration": self.current_iteration,
                "cpu_duration_ms": record["cpu_duration_ms"],
                "cuda_duration_ms": cuda_elapsed_ms,
                # [!!] 关键: 保留 V2 堆栈 timer 提供的层级 ID
                "node_id": record["node_id"],
                "parent_id": record["parent_id"],
                # (用于步骤 2 的临时字段)
                "children": [],
            }
            self.records.append(full_record)

        self._pending_records.clear()

        if not self.records or not self.verbose:
            # 如果没有记录, 或者我们处于非 verbose 模式, 就提前退出
            # (self.records 仍然被填充, 只是不记录日志)
            return

        # --- 步骤 2: 从 self.records (扁平列表) 重建调用树 ---

        # 使用字典以便快速查找
        nodes_map = {node["node_id"]: node for node in self.records}

        # (注意: 我们假设 self.root_node (id=0) 是全局根,
        #  我们在这里只构建本次迭代的子树)

        tree_roots = []  # 存放本次迭代的"顶层"调用

        for node in self.records:
            parent_id = node["parent_id"]
            if parent_id in nodes_map:
                # 这是一个子节点, 将它添加到其父节点的 'children' 列表中
                parent_node = nodes_map[parent_id]
                parent_node["children"].append(node)
            else:
                # 这是一个顶层节点 (其父节点不是本次迭代的记录,
                # 可能是全局根节点 'id=0')
                tree_roots.append(node)

        # --- 步骤 3: 递归计算 Self Time ---

        def calculate_self_time_recursive(node):
            """
            遍历树, 计算每个节点的 Self Time。
            返回: 该节点的 Total CUDA Time (用于父节点的计算)。
            """
            # 我们使用 CUDA 时间作为 "Total Time"
            total_time_ms = node.get("cuda_duration_ms") or 0.0

            if not node["children"]:
                # 如果是叶节点, Self Time == Total Time
                node["self_time_ms"] = total_time_ms
                return total_time_ms

            # 递归计算所有子节点的总时间
            children_total_time = 0.0
            for child in node["children"]:
                children_total_time += calculate_self_time_recursive(child)

            # Self Time = Total Time - Children's Total Time
            node["self_time_ms"] = total_time_ms - children_total_time

            # 返回 *Total Time* 给父节点
            return total_time_ms

        # 遍历所有顶层节点来启动计算
        for root_node in tree_roots:
            calculate_self_time_recursive(root_node)

        # --- 步骤 4: 递归地记录层级日志 ---

        def log_tree_recursive(node, indent_prefix=""):
            """
            遍历树, 并以缩进格式打印日志。
            """
            # 准备日志条目
            stage = node["stage"]
            cpu_ms = node["cpu_duration_ms"]
            cuda_ms = node.get("cuda_duration_ms") or 0.0
            self_ms = node.get("self_time_ms", 0.0)  # self_time 是我们刚计算的

            # [!!] 这就是你的 "Log Parsing" 解决方案 [!!]
            node_id = node["node_id"]
            parent_id = node["parent_id"]

            # 格式化 Self Time (仅在有子节点且 Self > 0 时显示)
            self_time_str = ""
            if node["children"] and self_ms > 0.001:
                self_time_str = f", Self {self_ms:.3f}ms"

            # [!!] 最终的、可解析的、层级的日志消息 [!!]
            log_msg = (
                f"[Iter {self.current_iteration}] {indent_prefix}"
                f"Stage '{stage}' [id={node_id}, p_id={parent_id}]: "
                f"CPU {cpu_ms:.3f}ms, CUDA {cuda_ms:.3f}ms{self_time_str}"
            )

            self.logger.info(log_msg)

            # 递归地打印子节点
            new_indent = indent_prefix + "  L "

            # (按 node_id 排序, 以确保日志顺序与调用顺序大致匹配)
            sorted_children = sorted(node["children"], key=lambda x: x["node_id"])

            for child in sorted_children:
                log_tree_recursive(child, indent_prefix=new_indent)

        # 确保 self.logger 可用
        self._ensure_logger()

        # (按 node_id 排序根节点)
        sorted_roots = sorted(tree_roots, key=lambda x: x["node_id"])

        # 启动日志记录
        for root_node in sorted_roots:
            log_tree_recursive(root_node, indent_prefix="")  # 顶层没有缩进

    def step(self):
        self.synchronize_and_log()

    def next_iteration(self):
        self.current_iteration += 1
        self.logger.info(
            f"--- MyTimer: Switched to iteration {self.current_iteration} ---"
        )

    def _gather_records(self):
        if not dist.is_initialized() or self.world_size == 1:
            return self.records
        all_records = None
        if self.rank == 0:
            all_records_list = [None] * self.world_size
            dist.gather_object(self.records, all_records_list, dst=0)
            all_records = [item for sublist in all_records_list for item in sublist]
        else:
            dist.gather_object(self.records, None, dst=0)
        return all_records

    def dump(self, sort_records: bool = False):
        """
        【修改后】将当前 Rank 的原始计时记录追加到日志文件中。

        Args:
            sort_records (bool, optional): 是否在写入前对记录进行排序。
                                         默认为 False，即按原始执行顺序写入。
                                         设置为 True 则按 iteration 和 stage name 排序。
        """
        if self.log_dir is None:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"{self.tag}_rank{self.rank}.log")

        # 【修改点 1】: 根据 sort_records 参数决定是否排序
        if sort_records:
            records_to_write = sorted(
                self.records, key=lambda x: (x["iteration"], x["stage"])
            )
            sort_info = "(Sorted)"
        else:
            # 默认情况下，直接使用原始记录列表，保留执行顺序
            records_to_write = self.records
            sort_info = "(Execution Order)"

        with open(log_path, "a") as f:
            f.write(
                f"\n==================== DUMP {sort_info} (Up to Iteration {self.current_iteration}) ====================\n"
            )

            # 【修改点 2】: 遍历处理后的列表
            for r in records_to_write:
                # 确保 cuda_duration_ms 存在，提供一个默认值以避免错误
                cuda_time = r.get("cuda_duration_ms")
                cuda_str = (
                    f"{cuda_time:>8.3f}ms" if cuda_time is not None else "N/A".rjust(8)
                )

                f.write(
                    f"[Iter {r['iteration']}][Rank {r['rank']}] Stage: {r['stage']:<30} | "
                    f"CPU: {r['cpu_duration_ms']:>8.3f}ms, CUDA: {cuda_str}\n"
                )

    def generate_report(self, stage_pattern, output_filename, iteration_filter=None):
        """
        【最终正确版 V2】生成详细的性能分析报告，并保存到文件。
        此版本会聚合所有 rank 的数据，按独立的 Stage Name 进行统一统计。
        """
        # _gather_records() 应该返回一个包含所有 rank 记录的列表
        all_records = self._gather_records()

        if self.rank == 0:
            # 1. 筛选符合条件的记录 (逻辑不变)
            pattern = re.compile(stage_pattern)
            filtered_records = [
                r
                for r in all_records
                if pattern.match(r["stage"])
                and (iteration_filter is None or iteration_filter(r["iteration"]))
            ]

            if not filtered_records:
                self.logger.warning(
                    f"No records found for pattern '{stage_pattern}' to generate report."
                )
                return {}

            # --- 核心修改区域 ---

            # 2. 按 stage_name 对所有 rank 的数据进行分组
            # 新逻辑：将所有 rank 中 name 相同的 stage 聚合在一起
            grouped_data = defaultdict(list)
            for r in filtered_records:
                # 不再关心 r["rank"]，只要 stage name 相同，就聚合
                if r["cuda_duration_ms"] is not None:
                    grouped_data[r["stage"]].append(r["cuda_duration_ms"])

            # 3. 计算每个聚合后 stage 的统计数据
            report_data = {}  # 不再需要按 rank 分组
            for stage_name, durations in grouped_data.items():
                # 每个 stage_name 都是一个独立的条目，其 durations 是来自所有 rank 的数据列表
                report_data[stage_name] = {
                    "count": len(durations),
                    "mean": np.mean(durations),
                    "median": np.median(durations),
                    "std": np.std(durations),
                    "min": np.min(durations),
                    "max": np.max(durations),
                }

            # --- 修改结束 ---

            # 4. 生成格式化的报告字符串 (现在只有一个聚合后的总表)
            report_string = ""
            report_header = (
                f"--- 📊 Aggregated Performance Report (All Ranks) ---\n"
                f"Pattern: '{stage_pattern}'\n"
                f"Filename: {output_filename}\n"
                f"{'-' * 80}\n"
            )
            report_string += report_header

            # 不再需要 for rank_id in ... 的循环
            report_string += "\n[Aggregated Statistics]\n"
            report_string += f" {'STAGE':<60} {'COUNT':<7} {'MEAN (ms)':<12} {'MEDIAN (ms)':<13} {'STD (ms)':<12}\n"
            report_string += f" {'-' * 59} {'-' * 6} {'-' * 11} {'-' * 12} {'-' * 11}\n"

            # 表格内容
            for stage_name in sorted(report_data.keys()):
                stats = report_data[stage_name]
                report_string += (
                    f" {stage_name:<60} {stats['count']:<7} "
                    f"{stats['mean']:<12.3f} {stats['median']:<13.3f} {stats['std']:<12.3f}\n"
                )

            # 5. 打印到控制台 (无需改动)
            print(report_string)

            # 6. 保存到文件 (无需改动)
            if self.log_dir:
                log_path = os.path.join(self.log_dir, output_filename)
                # os.makedirs(self.log_dir, exist_ok=True)
                # log_path = os.path.join(self.log_dir, output_filename)
                try:
                    with open(log_path, "w") as f:
                        f.write(report_string)
                    self.logger.info(f"Report successfully saved to '{log_path}'")
                except IOError as e:
                    self.logger.error(f"Failed to save report to '{log_path}': {e}")
            else:
                self.logger.warning("log_dir not set, cannot save report file.")

            return report_data

        return None

    def generate_csv(
        self, report_data: dict, csv_filename: str = "suffix_median_report.csv"
    ):
        """
        从 generate_report 的结果中提取后缀参数 bs/f/h/w/sp 和中位数，并保存为 CSV 文件。

        Args:
            report_data (dict): generate_report 返回的字典，结构为 report_data[rank][suffix]。
            csv_filename (str): 要保存的 CSV 文件名（默认为 suffix_median_report.csv）
        """
        import csv

        if self.rank != 0:
            return  # 只在 Rank 0 上执行

        if self.log_dir is None:
            self.logger.warning("log_dir not set, cannot save CSV file.")
            return

        csv_path = os.path.join(self.log_dir, csv_filename)

        try:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["bs", "f", "h", "w", "sp", "median"])  # 表头

                for suffix, stats in report_data[0].items():
                    match = re.match(
                        r"bs_(\d+)_f_(\d+)_h_(\d+)_w_(\d+)_sp(\d+)", suffix
                    )
                    if match:
                        bs, f, h, w, sp = match.groups()
                        median = stats["median"]
                        writer.writerow([bs, f, h, w, sp, median])

            self.logger.info(f"CSV report successfully saved to '{csv_path}'")

        except Exception as e:
            self.logger.error(f"Failed to save CSV report to '{csv_path}': {e}")


class NoOpMyTimer:
    """
    一个与 MyTimer 接口兼容的伪计时器。
    它的所有方法都是空操作，用于在禁用性能分析时作为 MyTimer 的“替身”。
    使用 *args 和 **kwargs 确保它能接收任何参数而不会报错。
    """

    def __init__(self, *args, **kwargs):
        pass

    def set_logger(self, logger_instance: logging.Logger):
        pass

    def set_labeler(self, labeler: Optional[LabelerProtocol]):
        pass

    def start(self, stage_name: str, *args, **kwargs):
        pass

    def stop(self, stage_name: str, *args, **kwargs):
        pass

    def synchronize_and_log(self):
        pass

    def step(self):
        """synchronize_and_log 的别名"""
        pass

    def next_iteration(self):
        pass

    def dump(self, *args, **kwargs):
        pass

    def generate_report(self, *args, **kwargs):
        # 真实的方法在 rank 0 上返回 dict，非 rank 0 返回 None
        # 这里直接返回 None 以保持行为一致
        return None

    def generate_csv(self, *args, **kwargs):
        pass

    def disable_cuda_time(self, *args, **kwargs):
        pass

    # 上下文管理器也需要实现
    @contextmanager
    def time_stage(self, stage_name: str, *args, **kwargs):
        try:
            yield
        finally:
            pass  # 无需执行任何操作

    def register_stage(self, *args, **kwargs):
        """空操作的注册方法，接收任何参数但什么也不做。"""
        pass


# SPMD下使用的全局实例
PROFILING_ENABLED = os.environ.get("ENABLE_TIMER", "0") == "1"
if PROFILING_ENABLED:
    global_timer = MyTimer()
else:
    global_timer = NoOpMyTimer()


def get_global_timer():
    return global_timer


# (在 my_utils.init_utils.py 中)

import os
import logging
from .logger import GlobalLogger, get_global_logger
from ..memory.memory_snapshot import global_snapshotter


def setup_logging_and_timer(
    args, role_tag: str, use_cuda: bool, use_nvtx: bool, is_distributed: bool
):
    """
    为当前进程 (Worker 或 Driver) 初始化 GlobalLogger 和 MyTimer。

    返回:
        (logging.Logger, MyTimer/NoOpTimer): 配置好的 logger 和 timer 实例。
    """

    # --- 1. 配置 GlobalLogger ---
    logger_instance = GlobalLogger()

    if not logger_instance.is_configured:
        if is_distributed:
            # Worker 进程: 从 torch.dist 获取 rank
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        elif os.environ.get("LOCAL_RANK") is not None:
            # torchrun
            rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            print(
                f"Detected torchrun environment: LOCAL_RANK={rank}, WORLD_SIZE={world_size}"
            )
        else:
            # Driver 进程: 总是 0/1
            rank = 0
            world_size = 1

        if not hasattr(args, "log_dir") or args.logdir is None:
            base_log_dir = "logs"
        else:
            base_log_dir = args.logdir

        # e.g., "logs/Critic" 或 "logs/Trainer_Driver"
        log_dir = os.path.join(base_log_dir, str(role_tag))

        logger_instance.setup(
            log_dir=log_dir,
            level=logging.INFO,  # or args.log_level
            rank=rank,
            world_size=world_size,
            extra_log_label=str(role_tag),
        )

    logger = get_global_logger()
    logger.info(
        f"Logger for {role_tag} (Rank {rank} World_size {world_size}) configured."
    )
    timer = None
    # --- 2. 配置 MyTimer ---
    if hasattr(args, "use_ray") and args.use_ray:
        if os.environ.get("ENABLE_TIMER", "0") == "1":
            logger.info(
                f"Performance Timer ENABLED for {role_tag}, nvtx={use_nvtx}, use_cuda={use_cuda}."
            )

            timer = MyTimer(
                use_cuda=use_cuda,
                tag=str(role_tag),
                log_dir=log_dir,
                labeler=create_labeler(enabled=use_nvtx),
                profile_memory=False,
            )

            # [!!] 注入 Logger (已修复 Bug)

        else:
            timer = NoOpMyTimer()
        timer.set_logger(logger)

    global_timer.set_logger(logger)
    global_timer.use_cuda = use_cuda
    global_timer.tag = str(role_tag)
    # global_timer.log_dir = logger_instance.log_dir # (重用 logger 的 log_dir)
    # global_timer.rank = logger_instance.rank

    # (来自你 V1 的硬编码)
    global_timer.set_labeler(create_labeler(enabled=False))
    global_timer.profile_memory = False

    # 3. 检查 global_timer 是哪种类型并记录
    # (我们通过检查它是否是 NoOpTimer 来判断 ENABLE_TIMER 的状态)
    if isinstance(global_timer, NoOpMyTimer):
        logger.info(f"Performance Timer is DISABLED for {role_tag}.")
    else:
        logger.info(
            f"Performance Timer ENABLED for {role_tag} (Rank {global_timer.rank})."
        )

    global_snapshotter.set_logger(logger=logger)

    return logger, timer


def print_cuda_memory_gb(step_name=""):
    """
    打印当前进程（Rank）的已分配和已缓存的 CUDA 显存。
    单位为 GB。
    """
    # 确保 CUDA 可用且分布式环境已初始化
    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"✅ [Rank {rank}] [CUDA Memory] {step_name}: "
            f"Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB"
        )
    else:
        # 如果没有分布式环境或 CUDA，只打印普通信息
        print(f"✅ {step_name}: CUDA not available or distributed not initialized.")


import threading
import traceback


class DebuggingEvent(threading.Event):
    """
    一个增强的 Event 类，用于调试。
    它在初始化时接收一个 logger 对象，并在 .set() 方法被调用时，
    使用该 logger 记录堆栈信息。
    """

    def __init__(self, *args, logger=None, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)

        # 保存 logger 对象，如果未提供，则创建一个默认的 print logger
        if logger:
            self.logger = logger
        else:
            # Fallback: 如果没有提供 logger，就退回到打印到控制台的行为
            self.logger = logging.getLogger("DebuggingEvent")
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

    def set(self):
        # 使用 StringIO 来捕获堆栈信息，而不是直接打印
        import io

        s = io.StringIO()
        traceback.print_stack(file=s)
        stack_info = s.getvalue()
        s.close()

        # 使用我们保存的 logger 来记录信息
        self.logger.info(
            f"\n{'=' * 30} [EVENT SET TRACE] {'=' * 30}\n"
            f">>> Event object {id(self)} is being set() by:\n"
            f"{stack_info}"
            f"{'=' * 80}"
        )

        # 调用父类的原始 set 方法
        super().set()


def record_oom_threshold(failing_bs: int, failing_frame: int, step: int = 4):
    """
    当发生OOM时，记录下对应batch size的安全帧数上限。

    这个函数是幂等的：
    1. 如果文件不存在，会自动创建。
    2. 如果记录已存在，只有在新的上限更严格（更小）时才会更新。

    Args:
        failing_bs (int): 导致OOM的batch size。
        failing_frame (int): 导致OOM的帧数。
        step (int): 帧数递增的步长，用于计算上一个安全点。
    """
    import json

    # from megatron.core import mpu
    sp = os.environ.get("sp")
    model = os.environ.get("model_type")
    resolution = os.environ.get("resolution")

    threshold_file = f"oom_thresholds_{model}_{resolution}_sp{sp}.json"
    print(f"--- OOM Detected! Recording threshold for bs={failing_bs} ---")

    # 根据失败的帧数，计算上一个已知的“安全”点
    # 例如：frame=73 失败了, 上限则为 69 (73-4)
    new_max_frame = failing_frame - step

    # 读取已有的阈值文件，如果不存在或为空则创建一个新的字典
    thresholds = {}
    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, "r", encoding="utf-8") as f:
                content = f.read()
                if content:  # 确保文件不为空
                    thresholds = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            print(
                f"Warning: Could not read or parse '{threshold_file}'. Starting with empty thresholds."
            )
            thresholds = {}  # 出错时重置

    # JSON的key必须是字符串
    failing_bs_str = str(failing_bs)

    # 获取当前bs已记录的上限，如果不存在则设为无穷大
    current_max = thresholds.get(failing_bs_str, float("inf"))

    # 只有在新的上限比旧的更严格（更小）时才更新
    if new_max_frame < current_max:
        print(
            f"Updating bs={failing_bs} max frame from {current_max} to {new_max_frame}"
        )
        thresholds[failing_bs_str] = new_max_frame

        # 将更新后的阈值漂亮地写回文件
        with open(threshold_file, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=4)
            print(f"Successfully saved new thresholds to '{threshold_file}'.")
    else:
        print(
            f"New max frame ({new_max_frame}) is not stricter than existing ({current_max}). No update needed."
        )


def print_tensor_info(tensor: torch.Tensor, name: str = ""):
    """
    将一个 PyTorch Tensor 的 Rank, Shape, Device, 和 Dtype 打印在同一行。

    Args:
        tensor (torch.Tensor): 需要检查的 PyTorch Tensor.
        name (str, optional): Tensor 的名字，用于在打印时区分. Defaults to "".
    """
    if not isinstance(tensor, torch.Tensor):
        print(f"提供的输入 '{name}' 不是一个有效的 PyTorch Tensor。")
        return

    # 使用 f-string 将所有信息格式化到一行
    # 如果提供了 name，则在前面加上 "name: "
    prefix = f"{name}: " if name else ""
    print(
        f"{prefix}"
        f"Rank {dist.get_rank()}, "
        f"Shape={tensor.shape}, "
        f"Device='{tensor.device}', "
        f"Dtype={tensor.dtype}"
    )


try:
    from tensordict import TensorDict  # optional dependency
except Exception:
    TensorDict = Any


IS_ENABLED = os.environ.get("DEBUG_DATA_CONSISTENCY", "0") == "1"
CSUM_PREFIX = "_csum_"


def _get_checksum_for_slice(tensor_slice: torch.Tensor) -> float:
    """
    [V6] 计算 *单个* 批次项 (slice) 的校验和。
    """
    if not IS_ENABLED:
        return 0.0
    try:
        # (确保 .float() 以防止 BFloat16 等的精度问题)
        return torch.sum(tensor_slice.cpu().float()).item()
    except Exception:
        return -1.0


class ChecksumUtils:
    @staticmethod
    def sign(payload: dict):
        """
        [在 *发送* 端调用 - V6 逐切片版]

        [!!] 核心修改:
        1. 遍历 payload 中的所有张量。
        2. 遍历该张量的 *每一项* (e.g., 0 到 batch_size-1)。
        3. 为 *每一项* 计算校验和。
        4. 将 [csum0, csum1, ...] 列表包装成一个 [BS] 形状的张量。
        """
        if not IS_ENABLED:
            return

        checksums_to_add = {}
        for key, value in payload.items():
            if not isinstance(value, torch.Tensor):
                continue

            csum_key = f"{CSUM_PREFIX}{key}"

            # [!!] V6 关键修复: 遍历 Batch [!!]
            batch_size = value.shape[0]
            csum_list = []
            for i in range(batch_size):
                # (为第 i 个切片计算校验和)
                csum_list.append(_get_checksum_for_slice(value[i]))

            # (创建 [BS] 形状的张量)
            checksums_to_add[csum_key] = torch.tensor(
                csum_list,
                dtype=torch.float32,
                device=value.device,  # (或 .cpu(), 但 .device 匹配更好)
            )

        # [!!] 关键: *修改* payload [!!]
        payload.update(checksums_to_add)

    @staticmethod
    def verify(batch: TensorDict, logger: logging.Logger):
        """
        [在 *接收* 端调用 - V6 逐切片版]

        在 *已切片* 的 'TensorDict' (.batch) 内部比较校验和。
        """
        if not IS_ENABLED:
            return

        if not logger:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] ChecksumUtils.verify: No logger provided, skipping.")
            return

        csum_keys = [k for k in batch.keys() if k.startswith(CSUM_PREFIX)]
        if not csum_keys:
            logger.info("[Checksum] No checksum tensors found in TensorDict.")
            return

        for csum_key in csum_keys:
            original_key = csum_key[len(CSUM_PREFIX) :]

            # 2. 获取 *校验和张量* (e.g., Shape [2])
            expected_csums_tensor = batch[csum_key]

            # 3. 获取 *数据张量* (e.g., Shape [2, F, C, H, W])
            received_tensors = batch.get(original_key)

            if received_tensors is None:
                logger.warning(
                    f"[Checksum] Found '{csum_key}' but "
                    f"missing '{original_key}' in TensorDict!"
                )
                continue

            batch_size = received_tensors.shape[0]
            if expected_csums_tensor.shape[0] != batch_size:
                logger.error(
                    f"[Checksum] FAILED: Mismatched batch size for '{original_key}'. "
                    f"Data has {batch_size} but csum has {expected_csums_tensor.shape[0]}."
                )
                continue

            # [!!] 4. V6 核心逻辑: 遍历 *切片后* 的 Batch [!!]
            for i in range(batch_size):
                tensor_slice = received_tensors[i]  # (获取第 i 个张量 (BS=1))
                expected_csum = expected_csums_tensor[
                    i
                ].item()  # (获取第 i 个校验和 (float))

                try:
                    # 重新计算 *该切片* 的校验和
                    new_csum = _get_checksum_for_slice(tensor_slice)

                    if not torch.allclose(
                        torch.tensor(expected_csum), torch.tensor(new_csum)
                    ):
                        logger.error(
                            f"[!!] CHECKSUM MISMATCH (Key: {original_key}, Batch Index: {i}) [!!]\n"
                            f"  Sender (Generator)   Calculated: {expected_csum}\n"
                            f"  Receiver (Critic)  Re-calculated: {new_csum}"
                        )
                    else:
                        logger.info(
                            f"[Checksum OK] Key: {original_key} (Index: {i}, Sum: {new_csum})"
                        )
                except Exception as e:
                    logger.error(f"[Checksum] FAILED to verify '{original_key}': {e}")
