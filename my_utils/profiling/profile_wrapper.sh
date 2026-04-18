#!/bin/bash
# ---------------------------------------------------------------------------
# Nsight Systems (2026.2) Universal Profiling Wrapper
# 用法: bash profile_wrapper.sh bash examples/light_bagel/run_pretrain.sh
# ---------------------------------------------------------------------------

set -e

# ----------------- 兼容辅助函数 -----------------
_is_true() {
    local v="${1:-}"
    v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
    [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
}

_nsys_detect_version() {
    local hint="${NSYS_VERSION_HINT:-}"
    if [[ "$hint" =~ ([0-9]{4})\.([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
        return 0
    fi
    local raw
    raw="$(nsys --version 2>/dev/null || true)"
    if [[ "$raw" =~ ([0-9]{4})\.([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    else
        echo "0 0"
    fi
}

_nsys_version_gte() {
    local major="${1:-0}"
    local minor="${2:-0}"
    if (( NSYS_VER_MAJOR > major )); then
        return 0
    fi
    if (( NSYS_VER_MAJOR == major && NSYS_VER_MINOR >= minor )); then
        return 0
    fi
    return 1
}

_trace_has_token() {
    local needle="${1:-}"
    shift || true
    local token
    for token in "$@"; do
        if [[ "$token" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

# ==========================================
# ⚙️ 1. 基础配置 (Base Configuration)
# ==========================================
OUTPUT_NAME="light_bagel_profile_%q{OMPI_COMM_WORLD_RANK:-%q{RANK:-0}}" # 自动带上分布式 Rank ID (如果环境变量存在)
FORCE_OVERWRITE="true"   # 覆盖同名报告 (-f)
EXPORT_FORMATS="sqlite"  # 除了 .nsys-rep 外，导出 sqlite 方便程序化分析 (可选 arrow, json, none)
SHOW_OUTPUT="true"       # 打印目标程序的标准输出/错误 (-w)

# ==========================================
# 🎯 2. 追踪目标设置 (Trace APIs)
# ==========================================
# 推荐对于 PyTorch: cuda, nvtx, osrt, cublas, cudnn
TRACE_APIS="cuda,nvtx,osrt,cublas,cudnn" 

# ==========================================
# ⏱️ 3. 抓取触发器配置 (Capture Range)
# ==========================================
# 选项: 
#   1. "none" (全局抓取，从头到尾，报告会很大)
#   2. "nvtx" (配合你代码里的 self._nvtx_range 使用)
#   3. "cudaProfilerApi" (配合代码里的 torch.cuda.profiler.start() 使用)
CAPTURE_MODE="nvtx" 

# [仅当 CAPTURE_MODE="nvtx" 时生效] 指定触发抓取的 NVTX 标记名
NVTX_TRIGGER_NAME="fusion_loop"  
SYSCALL_MODE="${SYSCALL_MODE:-}"   # nsys>=2026 推荐独立 --syscall=

# ==========================================
# 🔍 4. 噪音过滤配置 (Domain Filtering)
# ==========================================
# 开启 NVTX 域过滤，排除框架底层无用的细碎标记。
# 选项: "include" 或 "exclude" 或 "none"
DOMAIN_FILTER_MODE="include"
# 如果是 include，只保留这些 domain (逗号分隔); 如果是 exclude，屏蔽这些 domain
DOMAIN_NAMES="LightBagel,layer_detail" 

# ==========================================
# 📊 5. 硬件与内存指标 (Hardware & Memory Metrics)
# ==========================================
ENABLE_GPU_METRICS=1    # 1: 收集 SM 活跃度、Tensor Core 利用率等 (--gpu-metrics-devices=all)
ENABLE_CUDA_MEMORY=0    # 1: 追踪 CUDA 内存分配生命周期 (极耗性能，排查 OOM 时再开)
ENABLE_CPU_SAMPLING=0   # 1: 开启 CPU 调用栈采样 (排查 Dataloader 或 CPU 瓶颈时再开)
ENABLE_PYTHON_TRACE=0   # 1: 收集 Python 栈帧 (--python-sampling=true)
NIC_METRICS_MODE="${NIC_METRICS_MODE:-}" # 推荐: lf|hf|none（旧版会自动 fallback）
NSYS_VERSION_HINT="${NSYS_VERSION_HINT:-}"

# 版本探测与 trace/syscall 归一化
read -r NSYS_VER_MAJOR NSYS_VER_MINOR < <(_nsys_detect_version)
IFS=',' read -r -a _trace_in <<< "${TRACE_APIS}"
_trace_out=()
_trace_had_syscall=0
for _tok in "${_trace_in[@]}"; do
    _tok="${_tok// /}"
    [[ -z "${_tok}" ]] && continue
    if [[ "${_tok}" == "syscall" ]]; then
        _trace_had_syscall=1
        continue
    fi
    _trace_out+=("${_tok}")
done
if [[ ${#_trace_out[@]} -eq 0 ]]; then
    _trace_out=(cuda nvtx osrt cublas cudnn)
fi
TRACE_APIS_RESOLVED="$(IFS=,; echo "${_trace_out[*]}")"
if _nsys_version_gte 2026 0; then
    if [[ -z "${SYSCALL_MODE}" && "${_trace_had_syscall}" == "1" ]]; then
        SYSCALL_MODE="process-tree"
    fi
elif [[ -n "${SYSCALL_MODE}" && "${SYSCALL_MODE}" != "none" ]]; then
    if ! _trace_has_token "syscall" "${_trace_out[@]}"; then
        TRACE_APIS_RESOLVED="${TRACE_APIS_RESOLVED},syscall"
    fi
fi

# ==========================================
# 🚀 构建 nsys 参数数组
# ==========================================
NSYS_ARGS=(
    "--output=${OUTPUT_NAME}"
    "--force-overwrite=${FORCE_OVERWRITE}"
    "--export=${EXPORT_FORMATS}"
    "--show-output=${SHOW_OUTPUT}"
    "--trace=${TRACE_APIS_RESOLVED}"
    "--stats=true" # 运行结束后生成 CLI 统计摘要
)

# ----------------- 触发器逻辑 -----------------
NSYS_ARGS+=("--capture-range=${CAPTURE_MODE}")
if [[ "${CAPTURE_MODE}" == "nvtx" && -n "${NVTX_TRIGGER_NAME}" ]]; then
    NSYS_ARGS+=("--nvtx-capture=${NVTX_TRIGGER_NAME}")
fi

# ----------------- NVTX 过滤逻辑 -----------------
if [[ "${DOMAIN_FILTER_MODE}" == "include" ]]; then
    NSYS_ARGS+=("--nvtx-domain-include=${DOMAIN_NAMES}")
elif [[ "${DOMAIN_FILTER_MODE}" == "exclude" ]]; then
    NSYS_ARGS+=("--nvtx-domain-exclude=${DOMAIN_NAMES}")
fi

# ----------------- 指标采集逻辑 -----------------
if [[ ${ENABLE_GPU_METRICS} -eq 1 ]]; then
    NSYS_ARGS+=("--gpu-metrics-devices=all" "--gpu-metrics-frequency=10000")
fi

if [[ ${ENABLE_CUDA_MEMORY} -eq 1 ]]; then
    NSYS_ARGS+=("--cuda-memory-usage=true")
fi

if [[ ${ENABLE_CPU_SAMPLING} -eq 1 ]]; then
    NSYS_ARGS+=("--sample=process-tree")
    NSYS_ARGS+=("--osrt-threshold=10000") # 忽略极短的系统调用，防噪音
else
    NSYS_ARGS+=("--sample=none") # 关掉采样降低 Overhead
fi

if [[ ${ENABLE_PYTHON_TRACE} -eq 1 ]]; then
    NSYS_ARGS+=("--python-sampling=true" "--python-sampling-frequency=1000")
fi

# ----------------- NIC 指标兼容逻辑 -----------------
if [[ -n "${NIC_METRICS_MODE}" ]]; then
    _nic_mode="$(printf '%s' "${NIC_METRICS_MODE}" | tr '[:upper:]' '[:lower:]')"
    if _nsys_version_gte 2026 0; then
        if [[ "${_nic_mode}" == "true" || "${_nic_mode}" == "1" ]]; then
            _nic_mode="lf"
        elif [[ "${_nic_mode}" == "false" || "${_nic_mode}" == "0" ]]; then
            _nic_mode="none"
        fi
    else
        if [[ "${_nic_mode}" == "lf" || "${_nic_mode}" == "hf" ]]; then
            _nic_mode="true"
        elif [[ "${_nic_mode}" == "none" ]]; then
            _nic_mode="false"
        fi
    fi
    NSYS_ARGS+=("--nic-metrics=${_nic_mode}")
fi

# ----------------- syscall 新旧版本兼容 -----------------
if _nsys_version_gte 2026 0 && [[ -n "${SYSCALL_MODE}" ]]; then
    NSYS_ARGS+=("--syscall=${SYSCALL_MODE}")
fi

# ==========================================
# 💥 执行目标程序
# ==========================================
echo "========================================================"
echo "🚀 启动 Nsight Systems Profiling..."
echo "⚙️  命令: nsys profile ${NSYS_ARGS[*]} $@"
echo "========================================================"

# 注意：如果是 NVTX 触发，必须确保你代码里的 NVTX 开关打开
export ENABLE_NVTX=1 

# 执行 nsys 并传入目标程序 ($@ 代表所有传入这个脚本的参数)
nsys profile "${NSYS_ARGS[@]}" "$@"
