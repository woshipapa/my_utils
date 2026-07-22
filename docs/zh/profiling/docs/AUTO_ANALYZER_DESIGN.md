# 自动化性能分析器设计方案

## 设计目标

1. **智能识别瓶颈** - 自动发现性能热点和问题
2. **可解释的建议** - 不仅指出问题，还给出具体优化方向
3. **多维度分析** - 时间、内存、通信、计算利用率
4. **趋势预测** - 基于历史数据预测性能变化

## 核心分析框架

```
┌─────────────────────────────────────────────────────────────┐
│                    AnalyzerPipeline                          │
│  管道式处理: 数据清洗 → 特征提取 → 分析 → 报告生成          │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ TimeAnalyzer  │   │ MemoryAnalyzer│   │ CommAnalyzer  │
│ - 瓶颈检测    │   │ - 泄漏检测    │   │ - 热点检测    │
│ - 负载均衡    │   │ - 碎片化分析  │   │ - 带宽利用率  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                  ┌──────────────────┐
                  │ RuleEngine       │
                  │ - 模式匹配       │
                  │ - 阈值判定       │
                  │ - 异常检测       │
                  └──────────────────┘
                              │
                              ▼
                  ┌──────────────────┐
                  │ Recommendation  │
                  │   Generator      │
                  │ - 优先级排序     │
                  │ - 影响评估       │
                  │ - 具体方案       │
                  └──────────────────┘
```

## 1. 核心数据结构

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np

class Severity(Enum):
    CRITICAL = "critical"  # 需要立即处理
    HIGH = "high"         # 严重影响性能
    MEDIUM = "medium"     # 有一定影响
    LOW = "low"           # 轻微影响
    INFO = "info"         # 信息性

class Category(Enum):
    COMPUTE = "compute"           # 计算相关
    MEMORY = "memory"             # 内存相关
    COMMUNICATION = "communication"  # 通信相关
    IO = "io"                     # IO相关
    ARCHITECTURE = "architecture"  # 架构相关

@dataclass
class Finding:
    """分析发现"""
    id: str
    title: str
    description: str
    severity: Severity
    category: Category
    evidence: dict[str, Any]       # 支持证据
    affected_components: list[str] # 受影响的组件
    metrics: dict[str, float]     # 相关指标

@dataclass
class Recommendation:
    """优化建议"""
    id: str
    title: str
    description: str
    priority: int  # 1-10, 10最高
    estimated_impact: str  # 如 "可提升20-30%"
    effort: str  # "低", "中", "高"
    actions: list[str]  # 具体操作步骤
    references: list[str] = field(default_factory=list)  # 参考链接

@dataclass
class AnalysisReport:
    """完整分析报告"""
    metadata: dict[str, Any]
    findings: list[Finding]
    recommendations: list[Recommendation]
    summary: str
    overall_score: float  # 0-100

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "findings": [asdict(f) for f in self.findings],
            "recommendations": [asdict(r) for r in self.recommendations],
            "summary": self.summary,
            "overall_score": self.overall_score,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
```

## 2. 基础分析器接口

```python
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    """分析器基类"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._thresholds = self._load_thresholds()

    @abstractmethod
    def analyze(self, events: list[MetricEvent]) -> list[Finding]:
        """执行分析，返回发现"""
        pass

    def _load_thresholds(self) -> dict:
        """加载阈值配置"""
        return self.config.get("thresholds", {})

    def _create_finding(
        self,
        title: str,
        severity: Severity,
        evidence: dict,
    ) -> Finding:
        """创建Finding的工厂方法"""
        return Finding(
            id=self._generate_id(),
            title=title,
            description=self._describe(evidence),
            severity=severity,
            category=self.category,
            evidence=evidence,
            affected_components=evidence.get("components", []),
            metrics=evidence.get("metrics", {}),
        )
```

## 3. 时间分析器

```python
class TimeAnalyzer(BaseAnalyzer):
    """时间相关分析器"""

    category = Category.COMPUTE

    def analyze(self, events: list[MetricEvent]) -> list[Finding]:
        findings = []

        # 1. 瓶颈检测
        findings.extend(self._detect_bottlenecks(events))

        # 2. 负载不均衡检测
        findings.extend(self._detect_imbalance(events))

        # 3. Kernel效率分析
        findings.extend(self._analyze_kernel_efficiency(events))

        # 4. 计算密度分析
        findings.extend(self._analyze_compute_intensity(events))

        return findings

    def _detect_bottlenecks(self, events: list[MetricEvent]) -> list[Finding]:
        """
        检测性能瓶颈

        算法:
        1. 按操作名分组
        2. 计算每组占总时间比例
        3. 超过阈值的标记为瓶颈
        """
        # 过滤时间相关事件
        time_events = [e for e in events if self._is_time_metric(e)]
        if not time_events:
            return []

        # 按名称分组聚合
        grouped = {}
        for evt in time_events:
            name = evt.name
            if name not in grouped:
                grouped[name] = {"values": [], "unit": evt.unit}
            grouped[name]["values"].append(evt.value)

        # 计算统计
        total_time = sum(sum(g["values"]) for g in grouped.values())
        bottlenecks = []

        for name, data in grouped.items():
            total = sum(data["values"])
            ratio = total / total_time if total_time > 0 else 0

            threshold = self._thresholds.get("bottleneck_ratio", 0.1)  # 默认10%
            if ratio > threshold:
                avg = np.mean(data["values"])
                std = np.std(data["values"])
                cv = std / avg if avg > 0 else 0  # 变异系数

                bottlenecks.append(self._create_finding(
                    title=f"性能瓶颈: {name}",
                    severity=Severity.HIGH if ratio > 0.3 else Severity.MEDIUM,
                    evidence={
                        "component": name,
                        "total_time": total,
                        "ratio": ratio,
                        "avg_time": avg,
                        "std_time": std,
                        "cv": cv,
                        "count": len(data["values"]),
                        "unit": data["unit"],
                        "components": [name],
                        "metrics": {
                            f"{name}_ratio": ratio * 100,
                            f"{name}_avg_ms": avg,
                            f"{name}_stability": 1 - min(cv, 1.0),  # 稳定性
                        }
                    }
                ))

        return bottlenecks

    def _detect_imbalance(self, events: list[MetricEvent]) -> list[Finding]:
        """
        检测负载不均衡

        算法:
        1. 提取rank标签
        2. 按rank分组计算总时间
        3. 计算变异系数(CV)
        4. CV超过阈值则存在不均衡
        """
        # 按rank分组
        by_rank: dict[int, list[float]] = {}
        for evt in events:
            if "rank" in evt.tags and self._is_time_metric(evt):
                rank = int(evt.tags["rank"])
                if rank not in by_rank:
                    by_rank[rank] = []
                by_rank[rank].append(evt.value)

        if len(by_rank) < 2:
            return []  # 单rank无法判断不均衡

        # 计算每个rank的总时间
        rank_totals = {r: sum(times) for r, times in by_rank.items()}
        times = list(rank_totals.values())
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time if mean_time > 0 else 0

        threshold = self._thresholds.get("imbalance_cv", 0.2)  # 默认20%
        if cv > threshold:
            # 找出最慢和最快的rank
            slowest_rank = max(rank_totals, key=rank_totals.get)
            fastest_rank = min(rank_totals, key=rank_totals.get)
            slowdown = rank_totals[slowest_rank] / rank_totals[fastest_rank]

            return [self._create_finding(
                title="负载不均衡",
                severity=Severity.HIGH if cv > 0.5 else Severity.MEDIUM,
                evidence={
                    "cv": cv,
                    "slowest_rank": slowest_rank,
                    "fastest_rank": fastest_rank,
                    "slowdown_factor": slowdown,
                    "rank_times": rank_totals,
                    "components": ["distributed_training"],
                    "metrics": {
                        "imbalance_cv": cv,
                        "slowdown": slowdown,
                    }
                }
            )]

        return []

    def _analyze_kernel_efficiency(self, events: list[MetricEvent]) -> list[Finding]:
        """
        分析Kernel效率

        关注点:
        1. SM利用率低
        2. Memory bound vs Compute bound
        3. Warp divergence
        """
        # 需要NCU或nsys提供的详细指标
        # 这里假设有相关事件
        kernel_events = [e for e in events if "kernel" in e.name.lower()]

        findings = []

        # 检测低效kernel (高执行时间但低FLOPS)
        for evt in kernel_events:
            efficiency = evt.tags.get("flops_per_cycle", 0)
            if efficiency and efficiency < 0.3:  # 经验阈值
                findings.append(self._create_finding(
                    title=f"低效Kernel: {evt.name}",
                    severity=Severity.MEDIUM,
                    evidence={
                        "kernel_name": evt.name,
                        "flops_per_cycle": efficiency,
                        "duration_ms": evt.value,
                        "components": [evt.name],
                        "metrics": {"efficiency": efficiency}
                    }
                ))

        return findings

    def _analyze_compute_intensity(self, events: list[MetricEvent]) -> list[Finding]:
        """
        计算强度分析 (Arithmetic Intensity)

        AI = FLOPs / Bytes
        - 低AI: 内存受限
        - 高AI: 计算受限
        """
        # 需要FLOPs和内存访问量的数据
        # 这里简化处理
        return []

    def _is_time_metric(self, evt: MetricEvent) -> bool:
        """判断是否是时间指标"""
        time_keywords = ["time", "duration", "latency", "timer"]
        return any(k in evt.name.lower() for k in time_keywords)

    def _describe(self, evidence: dict) -> str:
        """生成描述文本"""
        if "ratio" in evidence:
            return f"'{evidence['component']}' 占总执行时间的 {evidence['ratio']*100:.1f}%"
        if "cv" in evidence:
            return f"Rank间执行时间变异系数为 {evidence['cv']:.2%}"
        return ""
```

## 4. 内存分析器

```python
class MemoryAnalyzer(BaseAnalyzer):
    """内存相关分析器"""

    category = Category.MEMORY

    def analyze(self, events: list[MetricEvent]) -> list[Finding]:
        findings = []

        # 1. 内存泄漏检测
        findings.extend(self._detect_memory_leak(events))

        # 2. 内存碎片化
        findings.extend(self._detect_fragmentation(events))

        # 3. OOM风险
        findings.extend(self._detect_oom_risk(events))

        # 4. 内存使用模式
        findings.extend(self._analyze_memory_pattern(events))

        return findings

    def _detect_memory_leak(self, events: list[MetricEvent]) -> list[Finding]:
        """
        内存泄漏检测

        算法:
        1. 按step分组，获取内存峰值
        2. 线性回归检测趋势
        3. 斜率>阈值且R²高则判定泄漏
        """
        # 按step分组
        by_step: dict[int, float] = {}
        for evt in events:
            if "step" in evt.tags and "memory" in evt.name.lower():
                step = int(evt.tags["step"])
                if step not in by_step or evt.value > by_step[step]:
                    by_step[step] = evt.value

        if len(by_step) < 10:
            return []  # 数据点太少

        # 线性回归
        steps = np.array(sorted(by_step.keys()))
        mems = np.array([by_step[s] for s in steps])

        # 去除异常值
        z_scores = np.abs((mems - mems.mean()) / mems.std())
        mask = z_scores < 3
        steps_clean = steps[mask]
        mems_clean = mems[mask]

        if len(steps_clean) < 5:
            return []

        # 简单线性回归
        coeffs = np.polyfit(steps_clean, mems_clean, 1)
        slope = coeffs[0]

        # 计算R²
        pred = np.polyval(coeffs, steps_clean)
        ss_res = ((mems_clean - pred) ** 2).sum()
        ss_tot = ((mems_clean - mems_clean.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot)

        # 判断泄漏
        leak_threshold = self._thresholds.get("memory_leak_slope_mb_per_step", 1.0)  # MB/step
        r_squared_threshold = self._thresholds.get("memory_leak_r2", 0.7)

        if abs(slope) > leak_threshold and r_squared > r_squared_threshold:
            # 计算泄漏速度
            leak_rate_mb_per_step = abs(slope)
            leak_rate_gb_per_1000_steps = leak_rate_mb_per_step * 1000 / 1024

            return [self._create_finding(
                title=f"内存{'泄漏' if slope > 0 else '持续释放'}",
                severity=Severity.CRITICAL if slope > 0 else Severity.INFO,
                evidence={
                    "slope": slope,
                    "r_squared": r_squared,
                    "leak_rate_mb_per_step": leak_rate_mb_per_step,
                    "leak_rate_gb_per_1000_steps": leak_rate_gb_per_1000_steps,
                    "trend": "increasing" if slope > 0 else "decreasing",
                    "components": ["memory"],
                    "metrics": {
                        "leak_rate_mb_per_step": leak_rate_mb_per_step,
                        "trend_strength": r_squared,
                    }
                }
            )]

        return []

    def _detect_fragmentation(self, events: list[MetricEvent]) -> list[Finding]:
        """
        内存碎片化检测

        关注:
        1. allocated << reserved
        2. 频繁的分配释放
        """
        allocated_events = [e for e in events if "allocated" in e.name.lower()]
        reserved_events = [e for e in events if "reserved" in e.name.lower()]

        if not allocated_events or not reserved_events:
            return []

        # 对齐时间戳
        # 计算碎片化比例
        fragmentation_ratios = []
        for alloc_evt in allocated_events:
            # 找同一时间的reserved
            for res_evt in reserved_events:
                if abs(res_evt.timestamp - alloc_evt.timestamp) < 0.001:
                    if res_evt.value > 0:
                        frag_ratio = 1 - (alloc_evt.value / res_evt.value)
                        fragmentation_ratios.append(frag_ratio)

        if fragmentation_ratios:
            avg_frag = np.mean(fragmentation_ratios)
            max_frag = np.max(fragmentation_ratios)

            frag_threshold = self._thresholds.get("fragmentation_ratio", 0.3)  # 30%
            if avg_frag > frag_threshold:
                return [self._create_finding(
                    title="内存碎片化严重",
                    severity=Severity.MEDIUM,
                    evidence={
                        "avg_fragmentation_ratio": avg_frag,
                        "max_fragmentation_ratio": max_frag,
                        "components": ["memory_allocator"],
                        "metrics": {"fragmentation": avg_frag}
                    }
                )]

        return []

    def _detect_oom_risk(self, events: list[MetricEvent]) -> list[Finding]:
        """
        OOM风险检测

        算法:
        1. 获取峰值内存
        2. 检测增长趋势
        3. 预测何时OOM
        """
        # 获取GPU总内存
        try:
            import torch
            if torch.cuda.is_available():
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            else:
                return []
        except Exception:
            return []

        # 获取峰值内存
        peak_memory = 0
        for evt in events:
            if "memory" in evt.name.lower() and isinstance(evt.value, (int, float)):
                if evt.value > peak_memory:
                    peak_memory = evt.value

        # 计算使用率
        usage_ratio = peak_memory / total_memory_gb

        oom_threshold = self._thresholds.get("oom_usage_ratio", 0.9)  # 90%
        warning_threshold = self._thresholds.get("oom_warning_ratio", 0.8)  # 80%

        if usage_ratio > oom_threshold:
            return [self._create_finding(
                title="OOM风险极高",
                severity=Severity.CRITICAL,
                evidence={
                    "peak_memory_gb": peak_memory,
                    "total_memory_gb": total_memory_gb,
                    "usage_ratio": usage_ratio,
                    "headroom_gb": total_memory_gb - peak_memory,
                    "components": ["memory"],
                    "metrics": {"memory_usage_ratio": usage_ratio}
                }
            )]
        elif usage_ratio > warning_threshold:
            return [self._create_finding(
                title="OOM风险预警",
                severity=Severity.HIGH,
                evidence={
                    "peak_memory_gb": peak_memory,
                    "total_memory_gb": total_memory_gb,
                    "usage_ratio": usage_ratio,
                    "components": ["memory"],
                    "metrics": {"memory_usage_ratio": usage_ratio}
                }
            )]

        return []

    def _analyze_memory_pattern(self, events: list[MetricEvent]) -> list[Finding]:
        """分析内存使用模式"""
        # 检测是否存在周期性的内存增长和释放
        return []
```

## 5. 通信分析器

```python
class CommunicationAnalyzer(BaseAnalyzer):
    """通信相关分析器"""

    category = Category.COMMUNICATION

    def analyze(self, events: list[MetricEvent]) -> list[Finding]:
        findings = []

        # 1. 通信热点检测
        findings.extend(self._detect_comm_hotspots(events))

        # 2. 带宽利用率
        findings.extend(self._analyze_bandwidth_utilization(events))

        # 3. 通信重叠度
        findings.extend(self._analyze_compute_comm_overlap(events))

        return findings

    def _detect_comm_hotspots(self, events: list[MetricEvent]) -> list[Finding]:
        """检测通信热点"""
        comm_keywords = ["allreduce", "allgather", "broadcast", "send", "recv", "nccl"]
        comm_events = [e for e in events if any(k in e.name.lower() for k in comm_keywords)]

        if not comm_events:
            return []

        # 按操作类型聚合
        by_op: dict[str, list[float]] = {}
        for evt in comm_events:
            op = evt.name.split(".")[0]  # 简化处理
            if op not in by_op:
                by_op[op] = []
            by_op[op].append(evt.value)

        findings = []
        total_comm_time = sum(sum(times) for times in by_op.values())

        for op, times in by_op.items():
            op_time = sum(times)
            ratio = op_time / total_comm_time if total_comm_time > 0 else 0

            hotspot_threshold = self._thresholds.get("comm_hotspot_ratio", 0.3)
            if ratio > hotspot_threshold:
                findings.append(self._create_finding(
                    title=f"通信热点: {op}",
                    severity=Severity.HIGH,
                    evidence={
                        "operation": op,
                        "total_time_ms": op_time,
                        "ratio": ratio,
                        "avg_time_ms": np.mean(times),
                        "count": len(times),
                        "components": [f"communication.{op}"],
                        "metrics": {
                            f"{op}_time_ratio": ratio,
                            f"{op}_avg_ms": np.mean(times),
                        }
                    }
                ))

        return findings

    def _analyze_bandwidth_utilization(self, events: list[MetricEvent]) -> list[Finding]:
        """分析带宽利用率"""
        # 需要通信量和时间数据
        return []

    def _analyze_compute_comm_overlap(self, events: list[MetricEvent]) -> list[Finding]:
        """分析计算通信重叠度"""
        # 分析NVTX范围，看通信是否与计算重叠
        return []
```

## 6. 推荐生成器

```python
class RecommendationGenerator:
    """基于规则和建议库生成优化建议"""

    def __init__(self):
        self._rules = self._load_rules()

    def generate(self, findings: list[Finding]) -> list[Recommendation]:
        """根据发现生成建议"""
        recommendations = []

        for finding in findings:
            # 查找匹配的规则
            matching_rules = [r for r in self._rules if r["match"](finding)]

            for rule in matching_rules:
                rec = rule["generate"](finding)
                if rec:
                    recommendations.append(rec)

        # 去重和排序
        recommendations = self._deduplicate(recommendations)
        recommendations = self._prioritize(recommendations)

        return recommendations

    def _load_rules(self) -> list[dict]:
        """加载推荐规则"""
        return [
            {
                "name": "high_kernel_time",
                "match": lambda f: (
                    f.category == Category.COMPUTE and
                    "ratio" in f.evidence and
                    f.evidence["ratio"] > 0.3
                ),
                "generate": self._suggest_kernel_optimization,
            },
            {
                "name": "memory_leak",
                "match": lambda f: "泄漏" in f.title,
                "generate": self._suggest_memory_leak_fix,
            },
            {
                "name": "load_imbalance",
                "match": lambda f: "不均衡" in f.title,
                "generate": self._suggest_balance_fix,
            },
            {
                "name": "fragmentation",
                "match": lambda f: "碎片化" in f.title,
                "generate": self._suggest_defragmentation,
            },
            {
                "name": "oom_risk",
                "match": lambda f: "OOM" in f.title,
                "generate": self._suggest_oom_prevention,
            },
        ]

    def _suggest_kernel_optimization(self, finding: Finding) -> Recommendation:
        """Kernel优化建议"""
        component = finding.evidence["component"]

        return Recommendation(
            id=f"kernel_opt_{component}",
            title=f"优化 {component} kernel",
            description=f"{component} 占用了 {finding.evidence['ratio']*100:.1f}% 的执行时间，建议进行优化",
            priority=9 if finding.evidence["ratio"] > 0.5 else 7,
            estimated_impact=f"可提升 {10 + finding.evidence['ratio']*30:.0f}%",
            effort="中",
            actions=[
                "检查kernel融合机会",
                "评估是否可以使用FlashAttention等优化库",
                "检查tensor shape是否导致低效的内存访问",
                "考虑使用Triton或CUDA Core手写优化kernel",
                "检查是否有冗余计算",
            ],
            references=[
                "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
                "https://github.com/openai/triton",
            ]
        )

    def _suggest_memory_leak_fix(self, finding: Finding) -> Recommendation:
        """内存泄漏修复建议"""
        return Recommendation(
            id="memory_leak_fix",
            title="修复内存泄漏",
            description=f"检测到内存以 {finding.evidence['leak_rate_mb_per_step']:.2f} MB/step的速度增长",
            priority=10,
            estimated_impact="防止训练崩溃",
            effort="高",
            actions=[
                "使用torch.cuda.memory_summary()分析内存分配",
                "检查是否有未释放的tensor引用",
                "检查gradient checkpointing配置",
                "检查是否有不必要的计算图保留",
                "逐层排查，定位具体泄漏点",
            ],
            references=[
                "https://pytorch.org/docs/stable/generated/torch.cuda.memory_summary.html",
                "https://discuss.pytorch.org/t/memory-leak-debugging/67427",
            ]
        )

    def _suggest_balance_fix(self, finding: Finding) -> Recommendation:
        """负载均衡修复建议"""
        return Recommendation(
            id="load_balance_fix",
            title="改善负载均衡",
            description=f"最慢rank比最快rank慢 {finding.evidence['slowdown_factor']:.2f}x",
            priority=8,
            estimated_impact=f"可减少 {(finding.evidence['slowdown_factor']-1)*50:.0f}% 的等待时间",
            effort="中",
            actions=[
                "检查数据加载是否均衡",
                "评估是否使用动态数据分片",
                "检查是否存在某些rank额外的计算",
                "考虑使用torch.utils.data.DistributedSampler",
                "评估模型并行策略是否合理",
            ],
            references=[
                "https://pytorch.org/tutorials/recipes/recipes/torch_profiler_recipe.html",
            ]
        )

    def _suggest_defragmentation(self, finding: Finding) -> Recommendation:
        """内存碎片化修复建议"""
        return Recommendation(
            id="memory_defrag",
            title="减少内存碎片化",
            description=f"平均碎片化比例为 {finding.evidence['avg_fragmentation_ratio']*100:.1f}%",
            priority=6,
            estimated_impact="可提升内存利用率",
            effort="低",
            actions=[
                "使用torch.cuda.empty_cache()手动释放缓存",
                "考虑使用内存池技术",
                "预分配大块内存，减少频繁分配",
                "使用torch.as_tensor()而非torch.tensor()",
            ],
        )

    def _suggest_oom_prevention(self, finding: Finding) -> Recommendation:
        """OOM预防建议"""
        headroom = finding.evidence.get("headroom_gb", 0)

        return Recommendation(
            id="oom_prevention",
            title="防止OOM",
            description=f"当前峰值内存 {finding.evidence['peak_memory_gb']:.2f}GB，剩余 {headroom:.2f}GB",
            priority=10 if headroom < 1 else 8,
            estimated_impact="防止训练崩溃",
            effort="中",
            actions=[
                "减小batch size或使用gradient accumulation",
                "启用gradient checkpointing",
                "使用mixed precision (FP16/BF16)",
                "使用torch.utils.checkpoint.checkpoint",
                "检查是否有不必要的中间tensor保留",
                "考虑使用CPU offloading技术",
            ],
            references=[
                "https://pytorch.org/docs/stable/checkpoint.html",
                "https://huggingface.co/docs/transformers/main_classes/optimizer_schedules",
            ]
        )

    def _deduplicate(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        """去重"""
        seen = set()
        deduped = []
        for rec in recommendations:
            key = (rec.title, rec.priority)
            if key not in seen:
                seen.add(key)
                deduped.append(rec)
        return deduped

    def _prioritize(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        """按优先级排序"""
        return sorted(recommendations, key=lambda r: r.priority, reverse=True)
```

## 7. 分析管道

```python
class AnalyzerPipeline:
    """分析管道，协调多个分析器"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.analyzers = self._init_analyzers()
        self.recommendation_generator = RecommendationGenerator()

    def _init_analyzers(self) -> list[BaseAnalyzer]:
        """初始化分析器"""
        return [
            TimeAnalyzer(self.config),
            MemoryAnalyzer(self.config),
            CommunicationAnalyzer(self.config),
        ]

    def analyze(self, events: list[MetricEvent]) -> AnalysisReport:
        """执行完整分析"""
        findings = []

        # 运行所有分析器
        for analyzer in self.analyzers:
            try:
                findings.extend(analyzer.analyze(events))
            except Exception as e:
                logger.warning(f"Analyzer {analyzer.__class__.__name__} failed: {e}")

        # 生成建议
        recommendations = self.recommendation_generator.generate(findings)

        # 计算整体得分
        overall_score = self._calculate_score(findings)

        # 生成摘要
        summary = self._generate_summary(findings, recommendations, overall_score)

        return AnalysisReport(
            metadata={
                "event_count": len(events),
                "finding_count": len(findings),
                "recommendation_count": len(recommendations),
                "timestamp": time.time(),
            },
            findings=findings,
            recommendations=recommendations,
            summary=summary,
            overall_score=overall_score,
        )

    def _calculate_score(self, findings: list[Finding]) -> float:
        """计算整体性能得分 (0-100)"""
        if not findings:
            return 100.0

        # 根据严重程度扣分
        severity_weights = {
            Severity.CRITICAL: 30,
            Severity.HIGH: 15,
            Severity.MEDIUM: 5,
            Severity.LOW: 1,
            Severity.INFO: 0,
        }

        total_penalty = sum(
            severity_weights.get(finding.severity, 0) for finding in findings
        )

        score = max(0, 100 - total_penalty)
        return score

    def _generate_summary(
        self,
        findings: list[Finding],
        recommendations: list[Recommendation],
        score: float,
    ) -> str:
        """生成摘要文本"""
        critical_count = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == Severity.HIGH)

        summary_parts = [
            f"性能得分: {score:.0f}/100",
            f"发现 {len(findings)} 个问题",
        ]

        if critical_count > 0:
            summary_parts.append(f"其中 {critical_count} 个严重问题")
        if high_count > 0:
            summary_parts.append(f"{high_count} 个高优先级问题")

        # 主要瓶颈
        bottlenecks = [f for f in findings if "瓶颈" in f.title]
        if bottlenecks:
            top_bottleneck = max(bottlenecks, key=lambda f: f.evidence.get("ratio", 0))
            summary_parts.append(
                f"主要瓶颈: {top_bottleneck.evidence['component']} "
                f"({top_bottleneck.evidence['ratio']*100:.1f}%)"
            )

        return " | ".join(summary_parts)
```

## 8. 使用示例

```python
from my_utils.profiling import (
    AnalyzerPipeline,
    MetricsCollector,
    MyTimerMetricsProvider,
)

# 设置分析器配置
config = {
    "thresholds": {
        "bottleneck_ratio": 0.15,      # 15%
        "imbalance_cv": 0.25,           # 25%
        "memory_leak_slope_mb_per_step": 0.5,
        "fragmentation_ratio": 0.35,
        "oom_usage_ratio": 0.85,
    }
}

# 创建管道
pipeline = AnalyzerPipeline(config)

# 收集指标
collector = MetricsCollector()
timer = MyTimer(use_cuda=True)
collector.register_provider(MyTimerMetricsProvider(timer))

# ... 训练循环 ...

# 执行分析
report = pipeline.analyze(collector._store.read_all_events())

# 输出报告
print(report.summary)
print(f"\n整体得分: {report.overall_score:.0f}/100")

print("\n=== 关键发现 ===")
for finding in report.findings[:5]:
    print(f"[{finding.severity.value.upper()}] {finding.title}")
    print(f"  {finding.description}")

print("\n=== 优化建议 ===")
for rec in report.recommendations[:5]:
    print(f"[优先级: {rec.priority}/10] {rec.title}")
    print(f"  {rec.description}")
    print(f"  预期影响: {rec.estimated_impact}")
    print(f"  工作量: {rec.effort}")
```

## 总结

自动化分析器通过以下方式实现智能分析：

1. **多分析器协同** - Time/Memory/Communication分析器各司其职
2. **规则驱动** - 可配置的阈值和规则
3. **证据导向** - 每个发现都有数据支持
4. **可操作建议** - 不仅指出问题，还给出具体方案

下一步：
1. 实现核心分析器
2. 扩展规则库
3. 添加机器学习预测能力
4. 集成到MetricsCollector
