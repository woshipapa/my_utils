import os
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity
from contextlib import contextmanager
import logging
import json
from typing import List, Union, Dict, Optional
import re
from collections import defaultdict
# It's good practice for a library module to not configure the root logger.
# It should use its own logger. The calling application can configure logging.
logger = logging.getLogger(__name__)

class FlexibleProfiler:
    """
    An internal class managing the profiling lifecycle and analysis.
    It's designed to be instantiated and used by the create_profiler_context factory.
    """
    def __init__(self,
                 is_enabled: bool,
                 output_dir: str,
                 logger_instance,
                 task_type: str ,
                 ops_to_analyze: Optional[Dict[str, str]] = None):
        
        self.is_enabled = is_enabled
        self.output_dir = output_dir
        self.logger = logger_instance
        self.ops_to_analyze = ops_to_analyze or {}
        self.profiler = None
        self.task_type = task_type

        if self.is_enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self.logger.info(f"INFO: Profiler is ENABLED for Rank {self.rank}.")

# 在 FlexibleProfiler 类中
    def _trace_handler(self, p):
        """Saves the standard Chrome trace and summary table."""

        # --- 新增的条件判断逻辑 ---
        if self.task_type == "VAE":
            self.logger.info(f"任务类型为 VAE，跳过生成 JSON Trace 文件 (Rank {self.rank}).")
        else:
            # 只有当任务类型不是 VAE 时，才导出 JSON trace
            trace_file = os.path.join(self.output_dir, f"torch_prof_rank{self.rank}.json")
            p.export_chrome_trace(trace_file)
            self.logger.info(f"Profiler trace for rank {self.rank} saved to {trace_file}")

        # --- Summary 的生成逻辑保持不变，不受影响 ---
        summary_file = os.path.join(self.output_dir, f'prof_summary_rank{self.rank}.txt')
        with open(summary_file, 'w') as f:
            f.write(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
        self.logger.info(f"Profiler summary for rank {self.rank} saved to {summary_file}")

    def __enter__(self):
        if not self.is_enabled:
            return self

        self.profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_enabled or self.profiler is None:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        
        # --- NEW: Perform and persist detailed analysis upon exit ---
        if self.ops_to_analyze:
            self._analyze_and_persist_ops()

    def _get_op_metrics(self, operation_name: str, metric: str) -> List[Union[int, float]]:
        """Internal method to extract metrics for a single operation."""
        all_events = self.profiler.events()
        metric_values = [getattr(event, metric, 0) for event in all_events if operation_name in event.name]
        return metric_values
        
    def _analyze_and_persist_ops(self):
        """
        [最终版] 分析指定的操作，支持正则表达式，并按每个独一无二的事件名独立输出统计结果。
        """
        self.logger.info(f"Performing detailed analysis (grouped by unique name) for specified ops on Rank {self.rank}...")

        all_events = self.profiler.events()
        
        # =================================================================
        # 阶段 1: 收集所有匹配模式的事件，并按其完整的、独特的名字分组
        # =================================================================
        # 结构: {"tile_encoder_0": {"metric": "...", "events": [...]}, "tile_encoder_1": ...}
        events_grouped_by_name = defaultdict(lambda: {"metric": "", "events": []})

        # 遍历你配置的模式 (e.g., r"tile_encoder_\d+")
        for pattern, metric in self.ops_to_analyze.items():
            # 遍历所有 profiler 事件
            for event in all_events:
                # 如果事件名匹配你的模式
                if re.match(pattern, event.name):
                    full_event_name = event.name  # 获取完整的名字，如 "tile_encoder_0"
                    # 以完整名字为 key 进行存储
                    events_grouped_by_name[full_event_name]["metric"] = metric
                    events_grouped_by_name[full_event_name]["events"].append(event)

        if not events_grouped_by_name:
            self.logger.warning("No events found matching the specified patterns.")
            return

        # =================================================================
        # 阶段 2: 遍历分组后的事件，计算每个独立组的统计数据
        # =================================================================
        final_analysis = {}
        
        for full_name, group_data in events_grouped_by_name.items():
            metric_to_use = group_data["metric"]
            events_list = group_data["events"]
            
            all_values_us = [getattr(event, metric_to_use, 0) for event in events_list]
            non_zero_values_us = [v for v in all_values_us if v > 0]

            if non_zero_values_us:
                count = len(non_zero_values_us)
                total_us = sum(non_zero_values_us)
                average_us = total_us / count
                
                total_ms = total_us / 1000.0
                average_ms = average_us / 1000.0
                values_ms = [v / 1000.0 for v in non_zero_values_us]

                final_analysis[full_name] = {
                    "metric": metric_to_use,
                    "count": count,
                    "average_ms": average_ms,
                    "total_ms": total_ms,
                    "values_ms": values_ms
                }
                self.logger.info(f"  - Analyzed '{full_name}': Found {count} instances, Avg: {average_ms:.3f} ms.")
            else:
                self.logger.warning(f"  - For '{full_name}', all recorded instances had a zero value for metric '{metric_to_use}'.")
        
        # =================================================================
        # 阶段 3: 保存最终结果
        # =================================================================
        if final_analysis:
            output_file = os.path.join(self.output_dir, f"detailed_metrics_rank_{self.rank}.json")
            with open(output_file, 'w') as f:
                json.dump(final_analysis, f, indent=4, sort_keys=True) # 按key排序更美观
            self.logger.info(f"Detailed analysis saved to {output_file}")

@contextmanager
def create_profiler_context(current_task_type: str,
                            logger,
                            enabled_env_var: str = "PROFILE_TASK_TYPES",
                            base_output_dir: str = "prof_results",
                            ops_to_analyze: Optional[Dict[str, str]] = None):
    """
    一个根据任务类型创建 profiling 上下文的工厂函数。

    如果环境变量中指定了当前任务类型，它会返回一个激活的 profiler，
    该 profiler 会在退出时自动保存摘要、trace 文件和指定操作的详细分析。

    Args:
        current_task_type (str): 当前正在执行的任务类型 (e.g., "DIT", "VAE").
        logger: 日志记录器实例.
        enabled_env_var (str): 用于指定要 profile 的任务类型的环境变量名.
                               其值应为逗号分隔的字符串, e.g., "DIT,VAE".
        base_output_dir (str): 保存所有 profiling 输出的基础目录.
        ops_to_analyze (dict): 指定要详细分析的操作的字典.
    """
    # 从环境变量读取需要 profile 的任务类型列表
    profilable_tasks_str = os.environ.get(enabled_env_var, "")
    profilable_tasks = [t.strip() for t in profilable_tasks_str.split(',') if t.strip()]

    # 检查当前任务类型是否在列表中
    is_profiling_enabled = current_task_type in profilable_tasks

    if not is_profiling_enabled:
        # 如果不启用，返回一个什么都不做的空上下文
        yield
        return

    # 为当前任务类型创建独立的输出目录
    task_specific_output_dir = os.path.join(base_output_dir, current_task_type)

    # 创建并 yield 功能强大的 profiler 实例
    profiler_instance = FlexibleProfiler(
        is_enabled=True,
        output_dir=task_specific_output_dir, # 使用任务专属的目录
        logger_instance=logger,
        task_type = current_task_type,
        ops_to_analyze=ops_to_analyze
    )
    
    with profiler_instance:
        yield