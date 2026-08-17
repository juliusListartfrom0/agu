"""Fail-closed CPU and memory guard for local AGU training processes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class TrainingResourceThresholds:
    """Host-level limits that must be breached repeatedly before training stops."""

    max_system_memory_percent: float = 90.0
    min_available_memory_gib: float = 2.0
    min_free_swap_gib: float = 0.0
    max_system_cpu_percent: float = 95.0
    consecutive_breaches: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.max_system_memory_percent < 100:
            raise ValueError("max_system_memory_percent must be in (0,100)")
        if self.min_available_memory_gib < 0:
            raise ValueError("min_available_memory_gib must be non-negative")
        if self.min_free_swap_gib < 0:
            raise ValueError("min_free_swap_gib must be non-negative")
        if not 0 < self.max_system_cpu_percent <= 100:
            raise ValueError("max_system_cpu_percent must be in (0,100]")
        if self.consecutive_breaches < 1:
            raise ValueError("consecutive_breaches must be positive")


@dataclass(frozen=True)
class ResourceSnapshot:
    sampled_at: str
    system_memory_percent: float
    available_memory_bytes: int
    system_cpu_percent: float
    process_tree_rss_bytes: int
    swap_free_bytes: int
    swap_percent: float


@dataclass(frozen=True)
class ResourceDecision:
    event: str
    stage: str
    snapshot: ResourceSnapshot
    reasons: tuple[str, ...]
    consecutive_breaches: int
    should_stop: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class TrainingResourceGuard:
    """Convert resource samples into a debounced stop decision."""

    def __init__(self, thresholds: TrainingResourceThresholds):
        self.thresholds = thresholds
        self._consecutive_breaches = 0

    def observe(self, snapshot: ResourceSnapshot, *, stage: str) -> ResourceDecision:
        reasons = self._breach_reasons(snapshot)
        self._consecutive_breaches = self._consecutive_breaches + 1 if reasons else 0
        return ResourceDecision(
            event="training_resource_sample",
            stage=stage,
            snapshot=snapshot,
            reasons=reasons,
            consecutive_breaches=self._consecutive_breaches,
            should_stop=(
                self._consecutive_breaches >= self.thresholds.consecutive_breaches
            ),
        )

    def _breach_reasons(self, snapshot: ResourceSnapshot) -> tuple[str, ...]:
        reasons: list[str] = []
        if (
            snapshot.system_memory_percent
            > self.thresholds.max_system_memory_percent
        ):
            reasons.append(
                f"system_memory_percent>{self.thresholds.max_system_memory_percent}"
            )
        available_gib = snapshot.available_memory_bytes / 1024**3
        if available_gib < self.thresholds.min_available_memory_gib:
            reasons.append(
                f"available_memory_gib<{self.thresholds.min_available_memory_gib:.3f}"
            )
        free_swap_gib = snapshot.swap_free_bytes / 1024**3
        if free_swap_gib < self.thresholds.min_free_swap_gib:
            reasons.append(
                f"free_swap_gib<{self.thresholds.min_free_swap_gib:.3f}"
            )
        if snapshot.system_cpu_percent > self.thresholds.max_system_cpu_percent:
            reasons.append(
                f"system_cpu_percent>{self.thresholds.max_system_cpu_percent}"
            )
        return tuple(reasons)


def sample_training_resources(process_id: int) -> ResourceSnapshot:
    """Sample host utilization and aggregate RSS for one supervised process tree."""

    import psutil

    memory = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
        swap_free_bytes = int(swap.free)
        swap_percent = float(swap.percent)
    except (OSError, psutil.Error):
        # Some sandboxed macOS processes cannot read host swap counters.
        # Represent unknown swap as exhausted so any positive free-swap
        # threshold remains fail-closed, while a deliberate zero threshold
        # can still supervise CPU and physical-memory pressure.
        swap_free_bytes = 0
        swap_percent = 100.0
    process_tree_rss = 0
    try:
        process = psutil.Process(process_id)
        processes = [process]
        try:
            processes.extend(process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            # macOS sandboxed processes may deny KERN_PROC_ALL while still
            # allowing the directly supervised child's RSS to be sampled.
            pass
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        processes = []
    for process in processes:
        try:
            process_tree_rss += int(process.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            continue
    return ResourceSnapshot(
        sampled_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        system_memory_percent=float(memory.percent),
        available_memory_bytes=int(memory.available),
        system_cpu_percent=float(psutil.cpu_percent(interval=None)),
        process_tree_rss_bytes=process_tree_rss,
        swap_free_bytes=swap_free_bytes,
        swap_percent=swap_percent,
    )
