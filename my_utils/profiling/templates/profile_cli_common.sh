#!/usr/bin/env bash

_profile_is_true() {
    local v="${1:-0}"
    v="${v,,}"
    [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
}

_profile_slug() {
    local s="${1:-}"
    s="${s// /}"
    s="${s//,/-}"
    s="${s//\//-}"
    s="${s//:/-}"
    s="${s//=/-}"
    s="${s//\{/}"
    s="${s//\}/}"
    echo "$s"
}

profile_collect_cli_args() {
    PROFILE_SETTINGS=(
        "${NSYS_SETTINGS[@]}"
        "${NSYS_LAUNCH_SETTINGS[@]}"
        "${TORCH_PROFILER_SETTINGS[@]}"
        "${PROFILING_ENV_SETTINGS[@]}"
    )
}

profile_init_defaults() {
    NSYS_PROFILER=${NSYS_PROFILER:-0}
    NSYS_START_STEP=${NSYS_START_STEP:-1}
    NSYS_STOP_STEP=${NSYS_STOP_STEP:-3}
    NSYS_ENABLE_NVTX_LABELS=${NSYS_ENABLE_NVTX_LABELS:-1}
    NSYS_NVTX_DOMAIN=${NSYS_NVTX_DOMAIN:-default}
    NSYS_OUTPUT=${NSYS_OUTPUT:-}
    NSYS_OUTPUT_DIR=${NSYS_OUTPUT_DIR:-./logs/nsys}
    NSYS_OUTPUT_PREFIX=${NSYS_OUTPUT_PREFIX:-}
    if [[ -z "$NSYS_OUTPUT_PREFIX" ]]; then
        NSYS_OUTPUT_PREFIX='nsys_profile_rank_%q{RANK}'
    fi
    NSYS_OUTPUT_AUTO_NAME=${NSYS_OUTPUT_AUTO_NAME:-1}
    NSYS_OUTPUT_SUFFIX=${NSYS_OUTPUT_SUFFIX:-}
    NSYS_FORCE_OVERWRITE=${NSYS_FORCE_OVERWRITE:-true}
    NSYS_EXPORT_FORMAT=${NSYS_EXPORT_FORMAT:-none}
    NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}
    NSYS_CAPTURE_RANGE=${NSYS_CAPTURE_RANGE:-cudaProfilerApi}
    NSYS_CAPTURE_RANGE_END=${NSYS_CAPTURE_RANGE_END:-stop}
    NSYS_GPU_METRICS_DEVICES=${NSYS_GPU_METRICS_DEVICES:-}
    NSYS_SAMPLE=${NSYS_SAMPLE:-}
    NSYS_CUDABACKTRACE=${NSYS_CUDABACKTRACE:-0}
    NSYS_NIC_METRICS=${NSYS_NIC_METRICS:-0}
    NSYS_NVTX_CAPTURE=${NSYS_NVTX_CAPTURE:-}
    NSYS_NVTX_DOMAIN_INCLUDE=${NSYS_NVTX_DOMAIN_INCLUDE:-}
    NSYS_NVTX_DOMAIN_EXCLUDE=${NSYS_NVTX_DOMAIN_EXCLUDE:-}

    TORCH_PROFILER=${TORCH_PROFILER:-0}
    TORCH_PROFILER_START_STEP=${TORCH_PROFILER_START_STEP:-$NSYS_START_STEP}
    TORCH_PROFILER_STOP_STEP=${TORCH_PROFILER_STOP_STEP:-$NSYS_STOP_STEP}
    TORCH_PROFILER_RECORD_SHAPES=${TORCH_PROFILER_RECORD_SHAPES:-1}
    TORCH_PROFILER_PROFILE_MEMORY=${TORCH_PROFILER_PROFILE_MEMORY:-1}
    TORCH_PROFILER_WITH_STACK=${TORCH_PROFILER_WITH_STACK:-1}
    TORCH_PROFILER_TRACE_DIR=${TORCH_PROFILER_TRACE_DIR:-}

    FSDP_PARAM_DEBUG=${FSDP_PARAM_DEBUG:-0}
    FSDP_PARAM_DEBUG_RANK=${FSDP_PARAM_DEBUG_RANK:-0}
    FSDP_PARAM_DEBUG_MAX_PARAMS=${FSDP_PARAM_DEBUG_MAX_PARAMS:-30}
}

profile_resolve_nsys_output() {
    if ! _profile_is_true "$NSYS_PROFILER"; then
        return 0
    fi
    if [[ -n "${NSYS_OUTPUT:-}" ]]; then
        return 0
    fi

    local output_dir="${NSYS_OUTPUT_DIR:-./logs/nsys}"
    local output_prefix="$NSYS_OUTPUT_PREFIX"
    local auto_name="${NSYS_OUTPUT_AUTO_NAME:-1}"
    local custom_suffix="${NSYS_OUTPUT_SUFFIX:-}"
    local output_suffix=""

    if _profile_is_true "$auto_name"; then
        local parts=()
        parts+=("step_${NSYS_START_STEP}_${NSYS_STOP_STEP}")
        if [[ -n "${NSYS_CAPTURE_RANGE:-}" && "${NSYS_CAPTURE_RANGE}" != "none" ]]; then
            parts+=("cap_$(_profile_slug "$NSYS_CAPTURE_RANGE")")
        fi
        if [[ -n "${NSYS_GPU_METRICS_DEVICES:-}" ]]; then
            parts+=("with_metrics_$(_profile_slug "$NSYS_GPU_METRICS_DEVICES")")
        else
            parts+=("no_metrics")
        fi
        if [[ -n "${NSYS_SAMPLE:-}" ]]; then
            parts+=("sample_$(_profile_slug "$NSYS_SAMPLE")")
        fi
        if [[ -n "$custom_suffix" ]]; then
            parts+=("$(_profile_slug "$custom_suffix")")
        fi
        output_suffix="$(IFS=_; echo "${parts[*]}")"
    elif [[ -n "$custom_suffix" ]]; then
        output_suffix="$(_profile_slug "$custom_suffix")"
    fi

    mkdir -p "$output_dir"
    if [[ -n "$output_suffix" ]]; then
        NSYS_OUTPUT="${output_dir%/}/${output_prefix}_${output_suffix}"
    else
        NSYS_OUTPUT="${output_dir%/}/${output_prefix}"
    fi
}

profile_load_preset() {
    local preset_file="${1:-}"
    if [[ -z "$preset_file" ]]; then
        return 0
    fi
    if [[ ! -f "$preset_file" ]]; then
        echo "[profile] preset file not found: $preset_file" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "$preset_file"
}

profile_build_cli_args() {
    NSYS_SETTINGS=()
    NSYS_LAUNCH_SETTINGS=()
    TORCH_PROFILER_SETTINGS=()
    PROFILING_ENV_SETTINGS=()

    if _profile_is_true "$NSYS_PROFILER"; then
        NSYS_SETTINGS+=(
            --nsys_profiler.enabled
            --nsys_profiler.start_step "$NSYS_START_STEP"
            --nsys_profiler.stop_step "$NSYS_STOP_STEP"
        )
        NSYS_LAUNCH_SETTINGS+=(
            --nsys_launch.enabled
            --nsys_launch.output "$NSYS_OUTPUT"
            --nsys_launch.export_format "$NSYS_EXPORT_FORMAT"
            --nsys_launch.trace "$NSYS_TRACE"
            --nsys_launch.capture_range "$NSYS_CAPTURE_RANGE"
            --nsys_launch.capture_range_end "$NSYS_CAPTURE_RANGE_END"
        )
        if [[ -n "$NSYS_GPU_METRICS_DEVICES" ]]; then
            NSYS_LAUNCH_SETTINGS+=(--nsys_launch.gpu_metrics_devices "$NSYS_GPU_METRICS_DEVICES")
        fi
        if ! _profile_is_true "$NSYS_FORCE_OVERWRITE"; then
            NSYS_LAUNCH_SETTINGS+=(--nsys_launch.no-force-overwrite)
        fi
        if _profile_is_true "$NSYS_CUDABACKTRACE"; then
            NSYS_LAUNCH_SETTINGS+=(--nsys_launch.cudabacktrace)
        fi
        if _profile_is_true "$NSYS_NIC_METRICS"; then
            NSYS_LAUNCH_SETTINGS+=(--nsys_launch.nic_metrics)
        fi
        if [[ -n "$NSYS_SAMPLE" ]]; then
            NSYS_LAUNCH_SETTINGS+=(--nsys_launch.sample "$NSYS_SAMPLE")
        fi
        if [[ -n "$NSYS_NVTX_CAPTURE" ]]; then
            NSYS_LAUNCH_SETTINGS+=(--nsys_launch.nvtx_capture "$NSYS_NVTX_CAPTURE")
        fi
        if [[ -n "$NSYS_NVTX_DOMAIN_INCLUDE" ]]; then
            NSYS_LAUNCH_SETTINGS+=(--nsys_launch.nvtx_domain_include "$NSYS_NVTX_DOMAIN_INCLUDE")
        fi
        if [[ -n "$NSYS_NVTX_DOMAIN_EXCLUDE" ]]; then
            NSYS_LAUNCH_SETTINGS+=(--nsys_launch.nvtx_domain_exclude "$NSYS_NVTX_DOMAIN_EXCLUDE")
        fi
    fi

    if _profile_is_true "$NSYS_ENABLE_NVTX_LABELS"; then
        NSYS_SETTINGS+=(
            --nsys_profiler.enable_nvtx_labels
            --nsys_profiler.nvtx_domain "$NSYS_NVTX_DOMAIN"
        )
    fi

    if _profile_is_true "$TORCH_PROFILER"; then
        TORCH_PROFILER_SETTINGS+=(
            --torch_profiler.enabled
            --torch_profiler.start_step "$TORCH_PROFILER_START_STEP"
            --torch_profiler.stop_step "$TORCH_PROFILER_STOP_STEP"
        )
        if ! _profile_is_true "$TORCH_PROFILER_RECORD_SHAPES"; then
            TORCH_PROFILER_SETTINGS+=(--torch_profiler.no-record-shapes)
        fi
        if ! _profile_is_true "$TORCH_PROFILER_PROFILE_MEMORY"; then
            TORCH_PROFILER_SETTINGS+=(--torch_profiler.no-profile-memory)
        fi
        if _profile_is_true "$TORCH_PROFILER_WITH_STACK"; then
            TORCH_PROFILER_SETTINGS+=(--torch_profiler.with-stack)
        fi
        if [[ -n "$TORCH_PROFILER_TRACE_DIR" ]]; then
            TORCH_PROFILER_SETTINGS+=(--torch_profiler.trace_dir "$TORCH_PROFILER_TRACE_DIR")
        fi
    fi

    if _profile_is_true "$FSDP_PARAM_DEBUG"; then
        PROFILING_ENV_SETTINGS+=(
            --profiling_env.fsdp_param_debug
            --profiling_env.fsdp_param_debug_rank "$FSDP_PARAM_DEBUG_RANK"
            --profiling_env.fsdp_param_debug_max_params "$FSDP_PARAM_DEBUG_MAX_PARAMS"
        )
    fi

    profile_collect_cli_args
}

profile_prepare() {
    local preset_file="${1:-${PROFILE_PRESET:-}}"
    profile_load_preset "$preset_file"
    profile_init_defaults
    profile_resolve_nsys_output
    profile_build_cli_args
}

profile_wrap_exec_with_nsys() {
    local array_name="$1"
    if ! _profile_is_true "$NSYS_PROFILER"; then
        return 0
    fi

    local -n exec_ref="$array_name"
    local nsys_cmd=(
        nsys profile
        --output="$NSYS_OUTPUT"
        --force-overwrite="$NSYS_FORCE_OVERWRITE"
        --export="$NSYS_EXPORT_FORMAT"
        --trace="$NSYS_TRACE"
        --capture-range="$NSYS_CAPTURE_RANGE"
        --capture-range-end="$NSYS_CAPTURE_RANGE_END"
    )
    if [[ -n "$NSYS_GPU_METRICS_DEVICES" ]]; then
        nsys_cmd+=(--gpu-metrics-devices="$NSYS_GPU_METRICS_DEVICES")
    fi
    if [[ "$NSYS_CAPTURE_RANGE" == "nvtx" && -n "$NSYS_NVTX_CAPTURE" ]]; then
        nsys_cmd+=(--nvtx-capture="$NSYS_NVTX_CAPTURE")
    fi
    if [[ -n "$NSYS_NVTX_DOMAIN_INCLUDE" ]]; then
        nsys_cmd+=(--nvtx-domain-include="$NSYS_NVTX_DOMAIN_INCLUDE")
    fi
    if [[ -n "$NSYS_NVTX_DOMAIN_EXCLUDE" ]]; then
        nsys_cmd+=(--nvtx-domain-exclude="$NSYS_NVTX_DOMAIN_EXCLUDE")
    fi
    if _profile_is_true "$NSYS_CUDABACKTRACE"; then
        nsys_cmd+=(--cudabacktrace=true)
    fi
    if [[ -n "$NSYS_SAMPLE" ]]; then
        nsys_cmd+=(--sample="$NSYS_SAMPLE")
    fi
    if _profile_is_true "$NSYS_NIC_METRICS"; then
        nsys_cmd+=(--nic-metrics=true)
    fi

    exec_ref=("${nsys_cmd[@]}" "${exec_ref[@]}")
}
