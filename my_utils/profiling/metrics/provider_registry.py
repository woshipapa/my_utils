from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

from .metrics_provider import MetricsProvider


ProviderFactory = Callable[[str, Mapping[str, Any], Mapping[str, Any]], MetricsProvider]


@dataclass
class ProviderSpec:
    provider_type: str
    provider_id: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderSpec":
        provider_type = str(payload.get("type", "")).strip()
        provider_id = str(payload.get("id", provider_type or "provider")).strip()
        enabled = bool(payload.get("enabled", True))
        params = dict(payload.get("params", {}) or {})
        return cls(
            provider_type=provider_type,
            provider_id=provider_id,
            enabled=enabled,
            params=params,
        )


class MetricsProviderRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, ProviderFactory] = {}

    def register(self, provider_type: str, factory: ProviderFactory) -> None:
        self._factories[str(provider_type).strip()] = factory

    def has(self, provider_type: str) -> bool:
        return str(provider_type).strip() in self._factories

    def list_types(self) -> list[str]:
        return sorted(self._factories.keys())

    def create(
        self,
        spec: ProviderSpec,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> MetricsProvider:
        provider_type = spec.provider_type.strip()
        if provider_type not in self._factories:
            raise KeyError(f"Unknown provider type: {provider_type}")
        factory = self._factories[provider_type]
        provider = factory(spec.provider_id, spec.params, context or {})
        if hasattr(provider, "set_enabled"):
            provider.set_enabled(spec.enabled)
        return provider


def _require_context_obj(context: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in context and context[key] is not None:
            return context[key]
    joined = ", ".join(keys)
    raise KeyError(f"Provider requires context object, expected one of: {joined}")


def register_builtin_providers(registry: MetricsProviderRegistry) -> MetricsProviderRegistry:
    from .metrics_providers import (
        CProfileStatsProvider,
        DcgmCsvMetricsProvider,
        ModuleProfilerMetricsProvider,
        MyTimerMetricsProvider,
        NcclLogMetricsProvider,
        NcuCsvMetricsProvider,
        NsysSqliteGlobMetricsProvider,
        NsysSqliteMetricsProvider,
        PerfStatTextProvider,
        RasJsonMetricsProvider,
        TableCsvMetricsProvider,
        TorchProfilerMetricsProvider,
    )

    def make_my_timer(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        timer = _require_context_obj(context, "my_timer", "timer")
        return MyTimerMetricsProvider(timer=timer, provider_id=provider_id, **dict(params))

    def make_torch_profiler(
        provider_id: str,
        params: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> MetricsProvider:
        profiler = _require_context_obj(context, "torch_profiler", "profiler")
        return TorchProfilerMetricsProvider(profiler=profiler, provider_id=provider_id, **dict(params))

    def make_module_profiler(
        provider_id: str,
        params: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> MetricsProvider:
        module_profiler = _require_context_obj(context, "module_profiler")
        return ModuleProfilerMetricsProvider(module_profiler=module_profiler, provider_id=provider_id, **dict(params))

    def make_table_csv(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        payload = dict(params)
        payload["provider_id"] = provider_id
        return TableCsvMetricsProvider(**payload)

    def make_ncu_csv(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        payload = dict(params)
        payload["provider_id"] = provider_id
        return NcuCsvMetricsProvider(**payload)

    def make_nsys_sqlite(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        payload = dict(params)
        payload["provider_id"] = provider_id
        sqlite_glob = str(payload.get("sqlite_glob", "") or "").strip()
        sqlite_path = str(payload.get("sqlite_path", "") or "").strip()
        has_glob = bool(sqlite_glob) or any(ch in sqlite_path for ch in ("*", "?", "["))
        if has_glob:
            if not sqlite_glob:
                payload["sqlite_glob"] = sqlite_path
                payload.pop("sqlite_path", None)
            return NsysSqliteGlobMetricsProvider(**payload)
        return NsysSqliteMetricsProvider(**payload)

    def make_cprofile(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        payload = dict(params)
        payload["provider_id"] = provider_id
        return CProfileStatsProvider(**payload)

    def make_perf_stat(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        payload = dict(params)
        payload["provider_id"] = provider_id
        return PerfStatTextProvider(**payload)

    def make_dcgm_csv(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        payload = dict(params)
        payload["provider_id"] = provider_id
        return DcgmCsvMetricsProvider(**payload)

    def make_nccl_log(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        payload = dict(params)
        payload["provider_id"] = provider_id
        return NcclLogMetricsProvider(**payload)

    def make_ras_json(provider_id: str, params: Mapping[str, Any], context: Mapping[str, Any]) -> MetricsProvider:
        payload = dict(params)
        payload["provider_id"] = provider_id
        return RasJsonMetricsProvider(**payload)

    registry.register("my_timer", make_my_timer)
    registry.register("torch_profiler", make_torch_profiler)
    registry.register("module_profiler", make_module_profiler)
    registry.register("table_csv", make_table_csv)
    registry.register("ncu_csv", make_ncu_csv)
    registry.register("nsys_sqlite", make_nsys_sqlite)
    registry.register("nsys_sqlite_glob", make_nsys_sqlite)
    registry.register("cprofile", make_cprofile)
    registry.register("perf_stat", make_perf_stat)
    registry.register("dcgm_csv", make_dcgm_csv)
    registry.register("nccl_log", make_nccl_log)
    registry.register("ras_json", make_ras_json)
    return registry


DEFAULT_PROVIDER_REGISTRY = register_builtin_providers(MetricsProviderRegistry())
