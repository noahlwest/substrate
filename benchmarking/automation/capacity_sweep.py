#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standing-cluster microVM latency baseline and subscription-limit sweeper.

Deploys Substrate, in-cluster Prometheus + OTLP telemetry meter, microVM
dependencies, and the WorkerPool once on a standing cluster, then drives clean
step-isolated Locust + boomer Jobs across:

  1. Uncontended single-subscription baseline (N = 1), capturing end-to-end
     wake/suspend latencies and per-phase atelet + ateworker waterfalls.
  2. Exponential expansion (N_seed, 2*N_seed, 4*N_seed, ...) until a step
     breaches the latency knee or reliability/OOM bounds (or hits max_users).
  3. Binary bisection between N_lo and N_hi down to bisect_precision to report
     N_max, duty cycle, and host CPU / RAM / Disk / NIC / PSI bottlenecks.

Outputs machine-readable `sweep_summary.json` (consumable by capacity sizing
models) and human-readable `baseline_report.md` artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

# Allow running directly as `python3 benchmarking/automation/capacity_sweep.py`
# from anywhere without requiring PYTHONPATH.
_AUTOMATION_DIR = Path(__file__).resolve().parent
if str(_AUTOMATION_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTOMATION_DIR))

import orchestrator  # noqa: E402
from testtypes import locust as locust_type  # noqa: E402
from util import parse_duration_seconds, run, run_no_check  # noqa: E402

REPO_ROOT = _AUTOMATION_DIR.parents[1]
SCRIPTS_DIR = (
    REPO_ROOT / "internal" / "benchmarking" / "boomer" / "agentsession" / "scripts"
)
MONITORING_MANIFEST = REPO_ROOT / "benchmarking" / "monitoring.yaml"
TELEMETRY_METER_MANIFEST = REPO_ROOT / "benchmarking" / "telemetry" / "meter.yaml"
LOCAL_MANIFESTS_DIR = _AUTOMATION_DIR / "manifests"

TELEMETRY_METER_OTLP_URL = (
    "http://telemetry-meter.benchmarking.svc.cluster.local:4317"
)
SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
QUANTITY_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(m|Ki|Mi|Gi|Ti)?$")
SCRIPT_DURATION_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?$")
GCS_URI_RE = re.compile(r"^gs://([a-z0-9][a-z0-9._-]{1,61}[a-z0-9])(?:/(.*))?$")

BINARY_UNITS = {
    "": 1.0,
    "m": 0.001,
    "Ki": float(1024),
    "Mi": float(1024**2),
    "Gi": float(1024**3),
    "Ti": float(1024**4),
}


@dataclass(frozen=True)
class ScriptInfo:
    """Metadata derived from an embedded agentsession YAML script."""

    name: str
    min_actor_memory: str
    min_actor_memory_bytes: int
    step_count: int
    unscaled_think_s: float
    unscaled_active_s: float


@dataclass
class SweepConfig:
    """Configuration for a single baseline + subscription-limit sweep."""

    name: str
    dest: str
    output_dir: str
    test_file: str = "/app/tests/agentsession.py"
    script: str = "personal-assistant-light"
    think_scale: float = 0.005
    sandbox_class: str = "microvm"
    lifecycle_mode: str = "pause"
    resume_mode: str = "implicit"
    worker_count: int = 1
    actor_memory: str = ""
    actor_cpu: str = "250m"
    worker_memory: str = ""
    worker_wait_timeout: int = 600
    step_duration: str = "6m"
    baseline_duration: str = "4m"
    seed_users: int = 16
    max_users: int = 1024
    bisect_precision: int = 16
    knee_multiplier: float = 2.0
    knee_floor_ms: float = 1000.0
    max_failure_ratio: float = 0.01
    min_user_fraction: float = 0.90
    atelet_lag_s: int = 70
    runner_cpu: str = "4"
    runner_memory: str = "8Gi"
    ate_args: list[str] = field(default_factory=list)
    extra_flags: list[str] = field(default_factory=list)
    skip_deploy: bool = False
    keep_deployed: bool = False
    dry_run: bool = False


@dataclass
class StepMetrics:
    """Normalized client, phase-breakdown, and host telemetry for one N trial."""

    users: int
    trial_name: str
    run_prefix: str = ""
    job_status: str = "complete"
    observed_peak_users: float | None = None
    wake_metric_name: str = "http_WakeFirstTouch"
    wake_p50_ms: float | None = None
    wake_p90_ms: float | None = None
    wake_p95_ms: float | None = None
    wake_p99_ms: float | None = None
    wake_requests: int = 0
    wake_failures: int = 0
    wake_rps: float | None = None
    resume_rpc_p50_ms: float | None = None
    resume_rpc_p90_ms: float | None = None
    resume_rpc_p99_ms: float | None = None
    suspend_metric_name: str = ""
    suspend_p50_ms: float | None = None
    suspend_p90_ms: float | None = None
    suspend_p99_ms: float | None = None
    suspend_rps: float | None = None
    aggregate_failure_ratio: float | None = None
    active_actors_p50: float | None = None
    active_actors_p90: float | None = None
    active_actors_p99: float | None = None
    duty_cycle: float | None = None
    oversubscription_ratio: float | None = None
    empirical_cycle_s: float | None = None
    snapshot_size_p50_mb: float | None = None
    snapshot_size_p95_mb: float | None = None
    checkpoint_mb_s: float | None = None
    working_set_node_p90_gb: float | None = None
    working_set_node_max_gb: float | None = None
    working_set_per_actor_p50_gb: float | None = None
    node_cpu_cores_p90: float | None = None
    node_cpu_cores_max: float | None = None
    node_disk_read_mib_s_p90: float | None = None
    node_disk_read_mib_s_max: float | None = None
    node_disk_write_mib_s_p90: float | None = None
    node_disk_write_mib_s_max: float | None = None
    node_disk_read_iops_p90: float | None = None
    node_disk_write_iops_p90: float | None = None
    node_net_rx_mib_s_p90: float | None = None
    node_net_tx_mib_s_p90: float | None = None
    psi_cpu_stall_p90: float | None = None
    psi_mem_stall_p90: float | None = None
    psi_io_stall_p90: float | None = None
    oom_node: int | None = None
    oom_pod: int | None = None
    worker_restarts: int | None = None
    worker_oomkilled_pods: int | None = None
    evicted: int | None = None
    hardware_facts: dict[str, Any] = field(default_factory=dict)
    phases: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepEvaluation:
    """Pass/fail verdict for a candidate subscription count N."""

    passed: bool
    knee_limit_ms: float
    reasons: list[str] = field(default_factory=list)


def parse_quantity_bytes(raw: str) -> float:
    """Parses a Kubernetes resource quantity (e.g. '256Mi', '1536Mi', '250m')."""
    m = QUANTITY_RE.fullmatch(raw.strip())
    if not m:
        raise ValueError(f"invalid resource quantity {raw!r}")
    val = float(m.group(1))
    unit = m.group(2) or ""
    return val * BINARY_UNITS[unit]


def _parse_go_duration_s(raw: str) -> float:
    text = str(raw).strip()
    if text in ("0", "0s", ""):
        return 0.0
    m = SCRIPT_DURATION_RE.fullmatch(text)
    if not m:
        return float(parse_duration_seconds(text))
    hours = float(m.group(1) or 0.0)
    minutes = float(m.group(2) or 0.0)
    seconds = float(m.group(3) or 0.0)
    return hours * 3600.0 + minutes * 60.0 + seconds


def load_agentsession_script_info(
    script_name: str, scripts_dir: Path = SCRIPTS_DIR
) -> ScriptInfo:
    """Loads step count, think sum, active dwell/burn sum, and min_actor_memory."""
    if not SAFE_NAME_RE.fullmatch(script_name):
        raise ValueError(f"unsafe script name {script_name!r}")
    path = (scripts_dir / f"{script_name}.yaml").resolve()
    if path.parent != scripts_dir.resolve() or not path.is_file():
        available = sorted(p.stem for p in scripts_dir.glob("*.yaml"))
        raise ValueError(
            f"unknown agentsession script {script_name!r} (available: {available})"
        )
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"malformed script YAML at {path}")
    min_mem = str(doc.get("min_actor_memory", "256Mi"))
    min_bytes = int(parse_quantity_bytes(min_mem))
    steps = doc.get("steps") or []
    think_total = 0.0
    active_total = 0.0
    for step in steps:
        think_total += _parse_go_duration_s(step.get("think", "0s"))
        for op in step.get("ops") or []:
            if "burn_cpu" in op:
                active_total += float(op["burn_cpu"].get("millis", 0)) / 1000.0
            elif "dwell" in op:
                active_total += float(op["dwell"].get("millis", 0)) / 1000.0
    return ScriptInfo(
        name=str(doc.get("name", script_name)),
        min_actor_memory=min_mem,
        min_actor_memory_bytes=min_bytes,
        step_count=len(steps),
        unscaled_think_s=round(think_total, 2),
        unscaled_active_s=round(active_total, 2),
    )


def validate_and_finalize_config(
    cfg: SweepConfig, scripts_dir: Path = SCRIPTS_DIR
) -> ScriptInfo | None:
    """Validates all CLI/config inputs before any cluster work runs."""
    if not SAFE_NAME_RE.fullmatch(cfg.name):
        raise ValueError(f"invalid sweep name {cfg.name!r}")
    if cfg.sandbox_class not in orchestrator.SANDBOX_CLASSES:
        raise ValueError(
            f"sandbox_class must be one of {orchestrator.SANDBOX_CLASSES}, "
            f"got {cfg.sandbox_class!r}"
        )
    if cfg.lifecycle_mode not in ("async", "sync", "suspend", "pause", "none"):
        raise ValueError(f"invalid lifecycle_mode {cfg.lifecycle_mode!r}")
    if cfg.resume_mode not in ("explicit", "implicit"):
        raise ValueError(f"invalid resume_mode {cfg.resume_mode!r}")
    if cfg.worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {cfg.worker_count}")
    if cfg.seed_users < 2:
        raise ValueError(f"seed_users must be >= 2, got {cfg.seed_users}")
    if cfg.max_users < cfg.seed_users:
        raise ValueError(
            f"max_users ({cfg.max_users}) must be >= seed_users ({cfg.seed_users})"
        )
    if cfg.bisect_precision < 1:
        raise ValueError(f"bisect_precision must be >= 1, got {cfg.bisect_precision}")
    if cfg.knee_multiplier <= 1.0:
        raise ValueError(f"knee_multiplier must be > 1.0, got {cfg.knee_multiplier}")
    if cfg.knee_floor_ms <= 0:
        raise ValueError(f"knee_floor_ms must be > 0, got {cfg.knee_floor_ms}")
    if not (0.0 <= cfg.max_failure_ratio <= 1.0):
        raise ValueError(f"max_failure_ratio must be in [0, 1], got {cfg.max_failure_ratio}")
    parse_duration_seconds(cfg.step_duration)
    parse_duration_seconds(cfg.baseline_duration)
    for q_name, q_val in (
        ("actor_cpu", cfg.actor_cpu),
        ("runner_cpu", cfg.runner_cpu),
        ("runner_memory", cfg.runner_memory),
    ):
        if q_val:
            parse_quantity_bytes(q_val)
    if cfg.worker_memory:
        parse_quantity_bytes(cfg.worker_memory)

    script_info: ScriptInfo | None = None
    if os.path.basename(cfg.test_file.split(",")[0]) == "agentsession.py" and cfg.script:
        script_info = load_agentsession_script_info(cfg.script, scripts_dir=scripts_dir)
        if not cfg.actor_memory:
            cfg.actor_memory = script_info.min_actor_memory
        else:
            given = int(parse_quantity_bytes(cfg.actor_memory))
            if given < script_info.min_actor_memory_bytes:
                raise ValueError(
                    f"actor_memory {cfg.actor_memory} is below script "
                    f"{script_info.name!r} min_actor_memory "
                    f"{script_info.min_actor_memory}"
                )
    elif cfg.actor_memory:
        parse_quantity_bytes(cfg.actor_memory)
    else:
        cfg.actor_memory = "256Mi"

    return script_info


def _runner_lifecycle_flag(mode: str) -> str:
    """Maps legacy pause/suspend aliases to #3097's async/sync CLI choices."""
    if mode == "pause":
        return "async"
    if mode == "suspend":
        return "sync"
    return mode


def build_runner_flags(cfg: SweepConfig) -> list[str]:
    """Constructs the exact runner.py / boomer CLI flags for a step Job."""
    flags: list[str] = [
        "--lifecycle-mode",
        _runner_lifecycle_flag(cfg.lifecycle_mode),
        "--resume-mode",
        cfg.resume_mode,
        "--atelet-lag-s",
        str(cfg.atelet_lag_s),
    ]
    if os.path.basename(cfg.test_file.split(",")[0]) == "agentsession.py":
        if cfg.script:
            flags += ["--agentsession-script", cfg.script]
        if cfg.think_scale > 0:
            flags += ["--agentsession-think-scale", str(cfg.think_scale)]
    flags += list(cfg.extra_flags)
    return flags


def compute_spawn_rate(users: int, target_ramp_s: float = 90.0) -> int:
    """Computes Locust -r spawn rate so all N users spawn within ~target_ramp_s.

    Locust defaults to 1 user/s when -r is omitted, which caps observed users
    below N for any high-density step (e.g. N >= 512). A floor of 16/s clears
    small/medium steps quickly without saturating initial cold-boot checkpoint
    staging, while math.ceil(users / target_ramp_s) bounds ramp <= 90s up to 4k+.
    """
    if users <= 1:
        return 1
    return max(16, math.ceil(users / target_ramp_s))


def compute_knee_limit_ms(
    baseline_p90_ms: float | None,
    knee_multiplier: float = 2.0,
    knee_floor_ms: float = 1000.0,
) -> float:
    """Computes tau_p90 = max(knee_multiplier * L_base_p90, knee_floor_ms)."""
    scaled = (baseline_p90_ms or 0.0) * knee_multiplier
    return round(max(scaled, knee_floor_ms), 2)


def evaluate_step(
    step: StepMetrics,
    knee_limit_ms: float,
    max_failure_ratio: float = 0.01,
    min_user_fraction: float = 0.90,
) -> StepEvaluation:
    """Determines whether a candidate subscription step N passed all SLO gates."""
    reasons: list[str] = []
    if step.job_status != "complete":
        reasons.append(f"runner job status={step.job_status}")
    if step.wake_requests <= 0 or step.wake_p90_ms is None:
        reasons.append(f"no wake measurements recorded ({step.wake_metric_name})")
    elif step.wake_p90_ms > knee_limit_ms:
        reasons.append(
            f"wake p90 {step.wake_p90_ms:.1f}ms exceeded knee limit {knee_limit_ms:.1f}ms"
        )
    if step.aggregate_failure_ratio is None:
        reasons.append("missing aggregate_failure_ratio")
    elif step.aggregate_failure_ratio > max_failure_ratio:
        reasons.append(
            f"failure ratio {step.aggregate_failure_ratio:.4f} > {max_failure_ratio:.4f}"
        )
    if (
        step.observed_peak_users is not None
        and step.users > 1
        and step.observed_peak_users < step.users * min_user_fraction
    ):
        reasons.append(
            f"observed peak users {step.observed_peak_users:.0f} < "
            f"{step.users * min_user_fraction:.0f} target"
        )
    for oom_field, label in (
        ("oom_node", "node OOM kills"),
        ("oom_pod", "pod OOM kills"),
        ("worker_oomkilled_pods", "OOMKilled worker pods"),
        ("evicted", "evicted worker pods"),
    ):
        count = getattr(step, oom_field)
        if count is not None and count > 0:
            reasons.append(f"{label}={count}")

    return StepEvaluation(
        passed=len(reasons) == 0,
        knee_limit_ms=knee_limit_ms,
        reasons=reasons,
    )


def next_subscription_step(
    tested: dict[int, bool],
    seed_users: int,
    max_users: int,
    bisect_precision: int,
) -> int | None:
    """Pure state-machine planner returning the next N to run, or None when done.

    Sequence:
      1. Run N = 1 baseline first; if N = 1 fails, abort immediately.
      2. Expand exponentially: seed_users, 2*seed_users, ..., clamped at max_users,
         until a candidate fails or max_users itself passes.
      3. Binary-bisect between N_lo (highest pass) and N_hi (lowest fail) until
         (N_hi - N_lo) <= bisect_precision.
    """
    if 1 not in tested:
        return 1
    if not tested[1]:
        return None

    passing = [n for n, ok in tested.items() if ok]
    failing = [n for n, ok in tested.items() if not ok]
    n_lo = max(passing)

    if not failing:
        if n_lo >= max_users:
            return None
        candidate = seed_users if n_lo < seed_users else min(n_lo * 2, max_users)
        return None if candidate in tested else candidate

    n_hi = min(n for n in failing if n > n_lo)
    if n_hi - n_lo <= bisect_precision:
        return None
    mid = (n_lo + n_hi) // 2
    if mid in tested or mid <= n_lo or mid >= n_hi:
        return None
    return mid


def _to_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _to_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return None


def parse_trial_artifacts(
    users: int,
    trial_name: str,
    stats_jsonl_text: str,
    run_prefix: str = "",
    job_status: str = "complete",
    preferred_wake_metric: str = "",
) -> StepMetrics:
    """Parses stats.jsonl (Locust rows + trial_summary + server_summary + phases)."""
    m = StepMetrics(
        users=users,
        trial_name=trial_name,
        run_prefix=run_prefix,
        job_status=job_status,
    )
    by_metric: dict[str, dict[str, Any]] = {}
    phases: list[dict[str, Any]] = []

    for line in stats_jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        metric = str(row.get("metric", ""))
        meas = row.get("measurements") or {}
        if metric.startswith("phase_") and metric != "phase_breakdown_summary":
            phases.append({
                "layer": meas.get("layer", ""),
                "op": meas.get("op", ""),
                "class": meas.get("class", ""),
                "kind": meas.get("kind", ""),
                "scope": meas.get("scope", ""),
                "phase": meas.get("phase", ""),
                "count": _to_int(meas.get("count")) or 0,
                "p50_ms": _to_float(meas.get("p50_ms")),
                "p90_ms": _to_float(meas.get("p90_ms")),
                "p95_ms": _to_float(meas.get("p95_ms")),
                "max_ms": _to_float(meas.get("max_ms")),
            })
        else:
            by_metric[metric] = meas

    # Resolve primary wake latency metric (implicit -> http_WakeFirstTouch,
    # explicit -> http_WakeFirstTouch if present or grpc_ResumeActor, glutton -> grpc_ResumeActor).
    wake_candidates = (
        [preferred_wake_metric] if preferred_wake_metric else []
    ) + ["http_WakeFirstTouch", "grpc_ResumeActor"]
    wake_name = next((c for c in wake_candidates if c and c in by_metric), "")
    if wake_name:
        wm = by_metric[wake_name]
        m.wake_metric_name = wake_name
        m.wake_p50_ms = _to_float(wm.get("p50"))
        m.wake_p90_ms = _to_float(wm.get("p90"))
        m.wake_p95_ms = _to_float(wm.get("p95"))
        m.wake_p99_ms = _to_float(wm.get("p99"))
        m.wake_requests = _to_int(wm.get("request_count")) or 0
        m.wake_failures = _to_int(wm.get("failure_count")) or 0
        m.wake_rps = _to_float(wm.get("requests_per_s"))

    if "grpc_ResumeActor" in by_metric:
        rm = by_metric["grpc_ResumeActor"]
        m.resume_rpc_p50_ms = _to_float(rm.get("p50"))
        m.resume_rpc_p90_ms = _to_float(rm.get("p90"))
        m.resume_rpc_p99_ms = _to_float(rm.get("p99"))

    for susp_name in ("grpc_SuspendActor", "grpc_SuspendActorAsync"):
        if susp_name in by_metric:
            sm = by_metric[susp_name]
            m.suspend_metric_name = susp_name
            m.suspend_p50_ms = _to_float(sm.get("p50"))
            m.suspend_p90_ms = _to_float(sm.get("p90"))
            m.suspend_p99_ms = _to_float(sm.get("p99"))
            m.suspend_rps = _to_float(sm.get("requests_per_s"))
            break

    trial = by_metric.get("trial_summary", {})
    m.aggregate_failure_ratio = _to_float(trial.get("aggregate_failure_ratio"))
    node_count = _to_float(trial.get("node_count"))
    actors_per_node = _to_float(trial.get("actors_per_node"))
    if node_count and actors_per_node is not None:
        m.observed_peak_users = round(node_count * actors_per_node, 2)
    m.hardware_facts = {
        k: trial.get(k)
        for k in (
            "machine_type",
            "node_count",
            "allocatable_cores",
            "allocatable_ram_gb",
            "worker_pod_count",
            "worker_actor_capacity",
            "worker_memory_limit_gb",
            "sandbox_class",
            "kubernetes_version",
        )
        if trial.get(k) is not None
    }

    srv = by_metric.get("server_summary", {})
    m.active_actors_p50 = _to_float(srv.get("active_actors_p50"))
    m.active_actors_p90 = _to_float(srv.get("active_actors_p90"))
    m.active_actors_p99 = _to_float(srv.get("active_actors_p99"))
    m.snapshot_size_p50_mb = _to_float(srv.get("snapshot_size_p50_mb"))
    m.snapshot_size_p95_mb = _to_float(srv.get("snapshot_size_p95_mb"))
    m.checkpoint_mb_s = _to_float(srv.get("checkpoint_mb_s"))
    m.working_set_node_p90_gb = _to_float(srv.get("working_set_node_p90_gb"))
    m.working_set_node_max_gb = _to_float(srv.get("working_set_node_max_gb"))
    m.working_set_per_actor_p50_gb = _to_float(
        srv.get("working_set_per_actor_p50_gb")
    )
    m.node_cpu_cores_p90 = _to_float(srv.get("node_cpu_cores_p90"))
    m.node_cpu_cores_max = _to_float(srv.get("node_cpu_cores_max"))
    m.node_disk_read_mib_s_p90 = _to_float(srv.get("node_disk_read_mib_s_p90"))
    m.node_disk_read_mib_s_max = _to_float(srv.get("node_disk_read_mib_s_max"))
    m.node_disk_write_mib_s_p90 = _to_float(srv.get("node_disk_write_mib_s_p90"))
    m.node_disk_write_mib_s_max = _to_float(srv.get("node_disk_write_mib_s_max"))
    m.node_disk_read_iops_p90 = _to_float(srv.get("node_disk_read_iops_p90"))
    m.node_disk_write_iops_p90 = _to_float(srv.get("node_disk_write_iops_p90"))
    m.node_net_rx_mib_s_p90 = _to_float(srv.get("node_net_rx_mib_s_p90"))
    m.node_net_tx_mib_s_p90 = _to_float(srv.get("node_net_tx_mib_s_p90"))
    m.psi_cpu_stall_p90 = _to_float(srv.get("psi_cpu_stall_p90"))
    m.psi_mem_stall_p90 = _to_float(srv.get("psi_mem_stall_p90"))
    m.psi_io_stall_p90 = _to_float(srv.get("psi_io_stall_p90"))
    m.oom_node = _to_int(srv.get("oom_events_node"))
    m.oom_pod = _to_int(srv.get("oom_events_pod"))
    m.worker_restarts = _to_int(srv.get("oom_events_worker_restarts"))
    m.worker_oomkilled_pods = _to_int(srv.get("oom_events_worker_oomkilled_pods"))
    m.evicted = _to_int(srv.get("oom_events_evicted"))
    m.phases = phases

    # Duty cycle & empirical cycle duration (Little's Law: L_active = lambda * T_active).
    if users > 0 and m.active_actors_p50 is not None:
        m.duty_cycle = round(m.active_actors_p50 / users, 4)
        if m.active_actors_p50 > 0:
            m.oversubscription_ratio = round(users / m.active_actors_p50, 2)
    if users > 0 and m.wake_rps and m.wake_rps > 0:
        m.empirical_cycle_s = round(users / m.wake_rps, 2)

    return m


def diagnose_bottleneck(
    steps: list[StepMetrics], evaluations: dict[int, StepEvaluation]
) -> dict[str, Any]:
    """Classifies the primary saturation bottleneck across the capacity knee."""
    failing_steps = [
        s for s in sorted(steps, key=lambda x: x.users)
        if not evaluations[s.users].passed
    ]
    if not failing_steps:
        best = max(steps, key=lambda x: x.users)
        return {
            "category": "unbounded_at_max_users",
            "summary": (
                f"Passed up to configured max_users={best.users} without "
                "hitting latency knee or resource exhaustion."
            ),
            "evidence": [],
        }

    knee = failing_steps[0]
    evidence: list[str] = list(evaluations[knee.users].reasons)
    alloc_cores = _to_float(knee.hardware_facts.get("allocatable_cores"))
    alloc_ram = _to_float(knee.hardware_facts.get("allocatable_ram_gb"))

    if any(
        (v or 0) > 0
        for v in (knee.oom_node, knee.oom_pod, knee.worker_oomkilled_pods, knee.evicted)
    ) or (
        alloc_ram
        and knee.working_set_node_max_gb
        and knee.working_set_node_max_gb >= 0.90 * alloc_ram
    ) or ((knee.psi_mem_stall_p90 or 0.0) >= 5.0):
        category = "memory_capacity"
        evidence.append(
            f"working_set_max={_fmt(knee.working_set_node_max_gb, ' GiB')}, "
            f"psi_mem_p90={_fmt(knee.psi_mem_stall_p90, '%', digits=2)}"
        )
    elif (knee.psi_io_stall_p90 or 0.0) >= 25.0 or (
        (knee.psi_io_stall_p90 or 0.0) >= 5.0
        and max(
            knee.node_disk_read_mib_s_max or 0.0,
            knee.node_disk_write_mib_s_max or 0.0,
        )
        >= 850.0
    ):
        category = "storage_io_bandwidth"
        evidence.append(
            f"psi_io_p90={_fmt(knee.psi_io_stall_p90, '%', digits=2)}, "
            f"disk_r/w_max={_fmt(knee.node_disk_read_mib_s_max)}/{_fmt(knee.node_disk_write_mib_s_max)} MiB/s"
        )
    elif (knee.aggregate_failure_ratio or 0.0) > 0.01 and (
        knee.suspend_p90_ms or 0.0
    ) >= 30000.0:
        category = "checkpoint_persistence_timeout"
        evidence.append(
            f"suspend_p50/p90={_fmt(knee.suspend_p50_ms)}/{_fmt(knee.suspend_p90_ms)} ms, "
            f"cpu_p90={_fmt(knee.node_cpu_cores_p90)} cores, "
            f"disk_w_max={_fmt(knee.node_disk_write_mib_s_max)} MiB/s"
        )
    elif (knee.psi_io_stall_p90 or 0.0) >= 5.0:
        category = "storage_io_bandwidth"
        evidence.append(
            f"psi_io_p90={_fmt(knee.psi_io_stall_p90, '%', digits=2)}, "
            f"disk_r/w_max={_fmt(knee.node_disk_read_mib_s_max)}/{_fmt(knee.node_disk_write_mib_s_max)} MiB/s"
        )
    elif (
        alloc_cores
        and knee.node_cpu_cores_p90 is not None
        and knee.node_cpu_cores_p90 >= 0.70 * alloc_cores
    ) or (
        (not alloc_cores or knee.node_cpu_cores_p90 is None)
        and (knee.psi_cpu_stall_p90 or 0.0) >= 10.0
    ):
        category = "cpu_contention"
        evidence.append(
            f"cpu_p90={_fmt(knee.node_cpu_cores_p90)} cores, "
            f"psi_cpu_p90={_fmt(knee.psi_cpu_stall_p90, '%', digits=2)}"
        )
    elif (knee.aggregate_failure_ratio or 0.0) > 0.01:
        category = "control_plane_or_slot_admission"
        evidence.append(
            f"cpu_p90={_fmt(knee.node_cpu_cores_p90)} cores, "
            f"psi_io_p90={_fmt(knee.psi_io_stall_p90, '%', digits=2)}, "
            f"disk_r/w_max={_fmt(knee.node_disk_read_mib_s_max)}/{_fmt(knee.node_disk_write_mib_s_max)} MiB/s"
        )
    else:
        category = "snapshot_restore_queueing"
        evidence.append(
            f"cpu_p90={_fmt(knee.node_cpu_cores_p90)} cores, "
            f"psi_io_p90={_fmt(knee.psi_io_stall_p90, '%', digits=2)}, "
            f"disk_r/w_max={_fmt(knee.node_disk_read_mib_s_max)}/{_fmt(knee.node_disk_write_mib_s_max)} MiB/s"
        )

    return {
        "category": category,
        "first_failing_users": knee.users,
        "summary": (
            f"Knee reached at N={knee.users} ({category}): "
            + "; ".join(evidence)
        ),
        "evidence": evidence,
    }


def _fmt(val: float | int | None, suffix: str = "", digits: int = 1) -> str:
    if val is None:
        return "—"
    if isinstance(val, int):
        return f"{val}{suffix}"
    return f"{val:.{digits}f}{suffix}"


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Renders a self-contained Markdown baseline + capacity report."""
    cfg = summary["config"]
    hw = summary.get("hardware_facts") or {}
    base = summary.get("baseline_n1") or {}
    best = summary.get("best_step") or {}
    bottleneck = summary.get("bottleneck") or {}

    lines = [
        f"# MicroVM Baseline & Subscription-Limit Report: `{cfg['name']}`",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Workload / Script** | `{cfg.get('script') or cfg['test_file']}` (`think_scale={cfg.get('think_scale')}`) |",
        f"| **Sandbox / Lifecycle / Resume** | `{cfg['sandbox_class']}` / `{cfg['lifecycle_mode']}` / `{cfg['resume_mode']}` |",
        f"| **Machine Type & Allocatable** | `{hw.get('machine_type', 'unknown')}` ({hw.get('allocatable_cores', '—')} cores, {hw.get('allocatable_ram_gb', '—')} GiB RAM, {hw.get('worker_pod_count', cfg['worker_count'])} worker pods) |",
        f"| **Baseline Wake Latency ($N = 1$)** | p50 **{_fmt(base.get('wake_p50_ms'), ' ms')}** / p90 **{_fmt(base.get('wake_p90_ms'), ' ms')}** / p99 **{_fmt(base.get('wake_p99_ms'), ' ms')}** |",
        f"| **Baseline Suspend Latency ($N = 1$)** | p50 **{_fmt(base.get('suspend_p50_ms'), ' ms')}** / p90 **{_fmt(base.get('suspend_p90_ms'), ' ms')}** / p99 **{_fmt(base.get('suspend_p99_ms'), ' ms')}** |",
        f"| **Knee Latency Limit ($\\tau_{{\\text{{p90}}}}$)** | **{_fmt(summary.get('knee_limit_ms'), ' ms')}** |",
        f"| **Subscription Limit ($N_{{\\max}}$)** | **{summary.get('n_max', '—')} subscriptions** |",
        f"| **Active Concurrency & Duty Cycle at $N_{{\\max}}$** | **{_fmt(best.get('active_actors_p50'))}** active actors (`duty_cycle={_fmt(best.get('duty_cycle'), digits=3)}`, `{_fmt(best.get('oversubscription_ratio'), 'x')}` oversubscription) |",
        f"| **Primary Bottleneck** | `{bottleneck.get('category', 'unknown')}` — {bottleneck.get('summary', '')} |",
        "",
        "## 2. Subscription-Limit Sweep Table",
        "",
        "| $N$ | Verdict | Wake p50 / p90 / p99 (ms) | Suspend p50 / p90 (ms) | Fail % | Active p50 | CPU p90 (cores) | Node RAM p90 (GiB) | Disk R / W p90 (MiB/s) | PSI IO / Mem p90 (%) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    evals = summary.get("evaluations") or {}
    for step in sorted(summary.get("steps", []), key=lambda s: s["users"]):
        ev = evals.get(str(step["users"])) or evals.get(step["users"]) or {}
        verdict = "PASS" if ev.get("passed") else "FAIL"
        fail_pct = (
            f"{step['aggregate_failure_ratio'] * 100:.2f}%"
            if step.get("aggregate_failure_ratio") is not None
            else "—"
        )
        lines.append(
            f"| {step['users']} | **{verdict}** | "
            f"{_fmt(step.get('wake_p50_ms'))} / {_fmt(step.get('wake_p90_ms'))} / {_fmt(step.get('wake_p99_ms'))} | "
            f"{_fmt(step.get('suspend_p50_ms'))} / {_fmt(step.get('suspend_p90_ms'))} | "
            f"{fail_pct} | {_fmt(step.get('active_actors_p50'))} | "
            f"{_fmt(step.get('node_cpu_cores_p90'))} | "
            f"{_fmt(step.get('working_set_node_p90_gb'))} | "
            f"{_fmt(step.get('node_disk_read_mib_s_p90'))} / {_fmt(step.get('node_disk_write_mib_s_p90'))} | "
            f"{_fmt(step.get('psi_io_stall_p90'), digits=2)} / {_fmt(step.get('psi_mem_stall_p90'), digits=2)} |"
        )

    n1_phases = base.get("phases") or []
    if n1_phases:
        lines += [
            "",
            "## 3. Single-Subscription ($N = 1$) Sub-Phase Waterfall",
            "",
            "| Layer | Op | Kind | Phase | Samples | p50 (ms) | p90 (ms) | p95 (ms) | Max (ms) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for ph in n1_phases:
            lines.append(
                f"| `{ph['layer']}` | `{ph['op']}` | `{ph.get('kind') or '—'}` | `{ph['phase']}` | "
                f"{ph['count']} | {_fmt(ph.get('p50_ms'))} | "
                f"{_fmt(ph.get('p90_ms'))} | {_fmt(ph.get('p95_ms'))} | "
                f"{_fmt(ph.get('max_ms'))} |"
            )

    lines.append("")
    return "\n".join(lines)


def read_latest_stats_jsonl(dest: str, trial_name: str, tag: str) -> tuple[str, str]:
    """Reads the latest stats.jsonl written under <dest>/runs/<trial_name>/..."""
    gcs_match = GCS_URI_RE.fullmatch(dest.rstrip("/"))
    if gcs_match:
        bucket_name = gcs_match.group(1)
        base_prefix = gcs_match.group(2) or ""
        search_prefix = "/".join(
            p for p in (base_prefix.strip("/"), "runs", trial_name) if p
        ) + "/"
        suffix = f"/run_tag={tag}/stats.jsonl"
        try:
            from google.cloud import storage
        except ModuleNotFoundError:
            storage = None

        if storage is not None:
            blobs = [
                b
                for b in storage.Client().bucket(bucket_name).list_blobs(
                    prefix=search_prefix
                )
                if b.name.endswith(suffix)
            ]
            if not blobs:
                raise FileNotFoundError(
                    f"no stats.jsonl found under gs://{bucket_name}/{search_prefix}"
                )
            latest = max(blobs, key=lambda b: b.name)
            run_prefix = f"gs://{bucket_name}/{latest.name.rsplit('/', 1)[0]}"
            return latest.download_as_text(encoding="utf-8"), run_prefix

        gcs_root = f"gs://{bucket_name}/{search_prefix}"
        listed = subprocess.run(
            ["gcloud", "storage", "ls", "-r", gcs_root],
            check=False,
            capture_output=True,
            text=True,
        )
        uris = [
            line.strip()
            for line in listed.stdout.splitlines()
            if line.strip().startswith(gcs_root) and line.strip().endswith(suffix)
        ]
        if not uris:
            raise FileNotFoundError(f"no stats.jsonl found under {gcs_root}")
        latest_uri = max(uris)
        content = subprocess.run(
            ["gcloud", "storage", "cat", latest_uri],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return content, latest_uri.rsplit("/", 1)[0]

    root = Path(dest).resolve() / "runs" / trial_name
    candidates = sorted(root.glob(f"run_date=*/run_ts=*/run_tag={tag}/stats.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no stats.jsonl found under {root}")
    latest_path = candidates[-1]
    return latest_path.read_text(encoding="utf-8"), str(latest_path.parent)


def _deploy_standing_control_plane(cfg: SweepConfig) -> None:
    """Deploys monitoring, OTLP meter, Substrate, and microVM runtime deps once."""
    os.chdir(REPO_ROOT)
    if (
        not os.environ.get("KUBECTL_CONTEXT")
        and os.environ.get("CLUSTER_NAME")
        and os.environ.get("CLUSTER_LOCATION")
        and os.environ.get("PROJECT_ID")
    ):
        run([
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            os.environ["CLUSTER_NAME"],
            "--location",
            os.environ["CLUSTER_LOCATION"],
            "--project",
            os.environ["PROJECT_ID"],
        ])
    run(["kubectl", "apply", "-f", str(MONITORING_MANIFEST)])
    run(["kubectl", "apply", "-f", str(TELEMETRY_METER_MANIFEST)])
    run([
        "kubectl",
        "rollout",
        "status",
        "deployment/prometheus",
        "-n",
        "benchmarking",
        "--timeout=180s",
    ])
    run([
        "kubectl",
        "rollout",
        "status",
        "deployment/telemetry-meter",
        "-n",
        "benchmarking",
        "--timeout=180s",
    ])
    ate_args = [
        '--credential-provider={"name":"k8s.io"}',
        f"--otlp-endpoint={TELEMETRY_METER_OTLP_URL}",
        *cfg.ate_args,
    ]
    orchestrator.deploy_substrate(ate_args)
    if cfg.sandbox_class == "microvm":
        orchestrator.install_microvm_deps()


def _deploy_sweep_workloads(cfg: SweepConfig) -> None:
    """Deploys WorkerPool + ActorTemplates sized for this sweep's actor_memory."""
    os.chdir(REPO_ROOT)
    orchestrator.deploy_workloads(
        worker_count=cfg.worker_count,
        sandbox_class=cfg.sandbox_class,
        actor_memory=cfg.actor_memory,
        wait_timeout_secs=cfg.worker_wait_timeout,
        worker_memory=cfg.worker_memory,
        actor_cpu=cfg.actor_cpu,
    )


def _deploy_standing_cluster(cfg: SweepConfig) -> None:
    """Deploys control plane + workloads for a standalone sweep."""
    _deploy_standing_control_plane(cfg)
    _deploy_sweep_workloads(cfg)


_KUBECTL_ATE_CACHE: Path | None = None


def _ensure_kubectl_ate_binary() -> str | None:
    """Builds ./cmd/kubectl-ate once per process for fast straggler cleanup."""
    global _KUBECTL_ATE_CACHE
    if _KUBECTL_ATE_CACHE is not None and _KUBECTL_ATE_CACHE.is_file():
        return str(_KUBECTL_ATE_CACHE)
    bin_path = Path(tempfile.gettempdir()) / f"kubectl-ate-{os.getpid()}"
    res = subprocess.run(
        ["go", "-C", str(REPO_ROOT), "build", "-o", str(bin_path), "./cmd/kubectl-ate"],
        check=False,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not bin_path.is_file():
        return None
    _KUBECTL_ATE_CACHE = bin_path
    return str(bin_path)


def purge_straggler_actors(
    atespace: str = "benchmark-workloads",
    prefix: str = "agent-",
    max_workers: int = 16,
) -> int:
    """Force-deletes any orphaned session actors left after boomer's 90s drain.

    Prevents leaked actors from an overloaded trial step from contaminating
    subsequent binary-bisection steps or blocking ActorTemplate replacement
    between sweeps in baseline-suite. Never touches golden template actors.
    """
    kubectl_ate = _ensure_kubectl_ate_binary()
    if not kubectl_ate:
        return 0
    listed = subprocess.run(
        [kubectl_ate, "get", "actor", "-a", atespace, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0 or not listed.stdout.strip():
        return 0
    try:
        doc = json.loads(listed.stdout)
    except json.JSONDecodeError:
        return 0
    stragglers = [
        str(item.get("metadata", {}).get("name", ""))
        for item in (doc.get("actors") or [])
        if str(item.get("metadata", {}).get("name", "")).startswith(prefix)
    ]
    if not stragglers:
        return 0
    print(
        f"  [cleanup] Purging {len(stragglers)} straggler {prefix}* actor(s) "
        f"in {atespace} before next step...",
        flush=True,
    )

    def _delete_one(actor_name: str) -> bool:
        r = subprocess.run(
            [
                kubectl_ate,
                "delete",
                "actor",
                actor_name,
                "-a",
                atespace,
                "--any-state",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return r.returncode == 0

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(stragglers)))) as pool:
        deleted = sum(1 for ok in pool.map(_delete_one, stragglers) if ok)
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        check = subprocess.run(
            [kubectl_ate, "get", "actor", "-a", atespace, "-o", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0 or not check.stdout.strip():
            break
        try:
            rem = [
                str(item.get("metadata", {}).get("name", ""))
                for item in (json.loads(check.stdout).get("actors") or [])
                if str(item.get("metadata", {}).get("name", "")).startswith(prefix)
            ]
        except json.JSONDecodeError:
            break
        if not rem:
            break
        time.sleep(2)
    print(
        f"  [cleanup] Deleted {deleted}/{len(stragglers)} straggler actor(s).",
        flush=True,
    )
    return deleted


def _teardown_standing_control_plane() -> None:
    os.chdir(REPO_ROOT)
    orchestrator.teardown_microvm_deps()
    orchestrator.teardown_substrate()
    run_no_check([
        "kubectl",
        "delete",
        "--ignore-not-found",
        "-f",
        str(TELEMETRY_METER_MANIFEST),
    ])


def _teardown_standing_cluster() -> None:
    os.chdir(REPO_ROOT)
    purge_straggler_actors()
    orchestrator.teardown_workloads()
    _teardown_standing_control_plane()


def run_capacity_sweep(
    cfg: SweepConfig,
    prebuilt_image: str = "",
    commit: str = "",
    manage_control_plane: bool = True,
) -> dict[str, Any]:
    """Executes (or dry-runs) a complete N=1 baseline + N_max capacity sweep."""
    script_info = validate_and_finalize_config(cfg)
    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = build_runner_flags(cfg)

    if cfg.dry_run:
        sample_job_subs = {
            "JOB_NAME": orchestrator.job_name(f"{cfg.name}-n1", commit or "dryrun0"),
            "IMAGE": prebuilt_image or "gcr.io/example/locust-test:dryrun0",
            "TAG": commit or "dryrun0",
            "NAME": f"{cfg.name}_n1",
            "DEST": cfg.dest,
            **locust_type.job_subs({
                "file": cfg.test_file,
                "duration": cfg.baseline_duration,
                "users": 1,
                "runnerCpu": cfg.runner_cpu,
                "runnerMemory": cfg.runner_memory,
            }),
        }
        rendered_manifest = orchestrator.render_template(
            locust_type.job_tmpl(str(LOCAL_MANIFESTS_DIR)),
            sample_job_subs,
            flags,
        )
        plan = {
            "mode": "dry_run",
            "config": asdict(cfg),
            "script_info": asdict(script_info) if script_info else None,
            "runner_flags": flags,
            "sample_n1_job_manifest": rendered_manifest,
        }
        (out_dir / "dry_run_plan.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        return plan

    os.chdir(REPO_ROOT)
    resolved_commit = commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    image = prebuilt_image or locust_type.build_image(resolved_commit)

    steps: list[StepMetrics] = []
    tested: dict[int, bool] = {}
    evaluations: dict[int, StepEvaluation] = {}
    knee_limit_ms = cfg.knee_floor_ms

    try:
        if not cfg.skip_deploy:
            if manage_control_plane:
                _deploy_standing_cluster(cfg)
            else:
                _deploy_sweep_workloads(cfg)

        while True:
            users = next_subscription_step(
                tested=tested,
                seed_users=cfg.seed_users,
                max_users=cfg.max_users,
                bisect_precision=cfg.bisect_precision,
            )
            if users is None:
                break

            duration = cfg.baseline_duration if users == 1 else cfg.step_duration
            trial_name = f"{cfg.name}_n{users}"
            step_flags = list(flags)
            if "-r" not in step_flags and "--spawn-rate" not in step_flags:
                step_flags += ["-r", str(compute_spawn_rate(users))]
            test_entry = {
                "name": trial_name,
                "type": "locust",
                "file": cfg.test_file,
                "duration": duration,
                "users": users,
                "runnerCpu": cfg.runner_cpu,
                "runnerMemory": cfg.runner_memory,
                "flags": [
                    *step_flags,
                    "--metadata",
                    json.dumps({
                        "sweep_name": cfg.name,
                        "sweep_users": users,
                        "sandbox_class": cfg.sandbox_class,
                        "lifecycle_mode": cfg.lifecycle_mode,
                        "resume_mode": cfg.resume_mode,
                        "script": cfg.script,
                    }),
                ],
            }
            print(
                f"\n=== [{cfg.name}] Running step N={users} (duration={duration}) ===",
                flush=True,
            )
            status = orchestrator.run_test(
                test=test_entry,
                image=image,
                dest=cfg.dest,
                commit=resolved_commit,
                manifests_dir=str(LOCAL_MANIFESTS_DIR),
            )
            try:
                jsonl_text, run_prefix = read_latest_stats_jsonl(
                    cfg.dest, trial_name, resolved_commit
                )
                step_metrics = parse_trial_artifacts(
                    users=users,
                    trial_name=trial_name,
                    stats_jsonl_text=jsonl_text,
                    run_prefix=run_prefix,
                    job_status=status,
                )
            except FileNotFoundError:
                step_metrics = StepMetrics(
                    users=users, trial_name=trial_name, job_status=status
                )

            if users == 1:
                knee_limit_ms = compute_knee_limit_ms(
                    step_metrics.wake_p90_ms,
                    knee_multiplier=cfg.knee_multiplier,
                    knee_floor_ms=cfg.knee_floor_ms,
                )

            verdict = evaluate_step(
                step_metrics,
                knee_limit_ms=knee_limit_ms,
                max_failure_ratio=cfg.max_failure_ratio,
                min_user_fraction=cfg.min_user_fraction,
            )
            steps.append(step_metrics)
            tested[users] = verdict.passed
            evaluations[users] = verdict
            print(
                f"  -> N={users}: {'PASS' if verdict.passed else 'FAIL'} "
                f"(wake_p90={step_metrics.wake_p90_ms}ms, limit={knee_limit_ms}ms, "
                f"reasons={verdict.reasons})",
                flush=True,
            )
            # Incrementally checkpoint partial progress after every completed step.
            _passing_now = [s for s in steps if evaluations[s.users].passed]
            _best_now = max(_passing_now, key=lambda s: s.users) if _passing_now else None
            _n1_now = next((s for s in steps if s.users == 1), None)
            _partial = {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "in_progress": True,
                "commit": resolved_commit,
                "config": asdict(cfg),
                "script_info": asdict(script_info) if script_info else None,
                "hardware_facts": (
                    (_best_now.hardware_facts if _best_now and _best_now.hardware_facts else None)
                    or (_n1_now.hardware_facts if _n1_now else {})
                ),
                "knee_limit_ms": knee_limit_ms,
                "n_max": _best_now.users if _best_now else 0,
                "baseline_n1": asdict(_n1_now) if _n1_now else None,
                "best_step": asdict(_best_now) if _best_now else None,
                "bottleneck": diagnose_bottleneck(steps, evaluations),
                "steps": [asdict(s) for s in sorted(steps, key=lambda x: x.users)],
                "evaluations": {str(u): asdict(ev) for u, ev in sorted(evaluations.items())},
            }
            (out_dir / "sweep_summary.json").write_text(
                json.dumps(_partial, indent=2) + "\n", encoding="utf-8"
            )
            (out_dir / "baseline_report.md").write_text(
                render_markdown_report(_partial), encoding="utf-8"
            )
            # Purge any orphaned agent-* actors left behind if an overloaded
            # step exceeded boomer's 90s SIGTERM drain window, then let async
            # worker/atelet cleanup settle before starting the next step.
            purge_straggler_actors()
            time.sleep(5)
    finally:
        if not cfg.skip_deploy:
            if manage_control_plane:
                if not cfg.keep_deployed:
                    _teardown_standing_cluster()
            else:
                purge_straggler_actors()
                orchestrator.teardown_workloads()

    passing_steps = [s for s in steps if evaluations[s.users].passed]
    best_step = max(passing_steps, key=lambda s: s.users) if passing_steps else None
    n1_step = next((s for s in steps if s.users == 1), None)
    hw_facts = (
        (best_step.hardware_facts if best_step and best_step.hardware_facts else None)
        or (n1_step.hardware_facts if n1_step else {})
    )
    bottleneck = diagnose_bottleneck(steps, evaluations)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": resolved_commit,
        "config": asdict(cfg),
        "script_info": asdict(script_info) if script_info else None,
        "hardware_facts": hw_facts,
        "knee_limit_ms": knee_limit_ms,
        "n_max": best_step.users if best_step else 0,
        "baseline_n1": asdict(n1_step) if n1_step else None,
        "best_step": asdict(best_step) if best_step else None,
        "bottleneck": bottleneck,
        "steps": [asdict(s) for s in sorted(steps, key=lambda x: x.users)],
        "evaluations": {str(u): asdict(ev) for u, ev in sorted(evaluations.items())},
    }

    (out_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "baseline_report.md").write_text(
        render_markdown_report(summary), encoding="utf-8"
    )
    return summary


def render_suite_markdown_report(results: list[dict[str, Any]]) -> str:
    """Renders a cross-configuration comparison table across a multi-sweep suite."""
    hw = next((r.get("hardware_facts") for r in results if r.get("hardware_facts")), {}) or {}
    lines = [
        "# MicroVM Baseline Suite Summary",
        "",
        f"- **Machine Type**: `{hw.get('machine_type', 'unknown')}` "
        f"({hw.get('allocatable_cores', '—')} cores, {hw.get('allocatable_ram_gb', '—')} GiB allocatable RAM)",
        "",
        "| Sweep | Script | Lifecycle | Actor Mem | $N=1$ Wake p50 / p90 (ms) | $N=1$ Suspend p50 / p90 (ms) | $N_{\\max}$ | Active p50 @ $N_{\\max}$ | Duty Cycle | Primary Bottleneck |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for res in results:
        cfg = res.get("config") or {}
        n1 = res.get("baseline_n1") or {}
        best = res.get("best_step") or {}
        bn = res.get("bottleneck") or {}
        lines.append(
            f"| `{cfg.get('name', '')}` | `{cfg.get('script', '')}` | "
            f"`{cfg.get('lifecycle_mode', '')}` | `{cfg.get('actor_memory', '')}` | "
            f"{_fmt(n1.get('wake_p50_ms'))} / {_fmt(n1.get('wake_p90_ms'))} | "
            f"{_fmt(n1.get('suspend_p50_ms'))} / {_fmt(n1.get('suspend_p90_ms'))} | "
            f"**{res.get('n_max', 0)}** | {_fmt(best.get('active_actors_p50'))} | "
            f"{_fmt(best.get('duty_cycle'), digits=3)} | `{bn.get('category', 'unknown')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def build_baseline_suite_configs(
    dest: str,
    output_dir: str,
    max_users_light: int = 1024,
    max_users_full: int = 512,
    step_duration: str = "6m",
    baseline_duration: str = "4m",
    seed_users: int = 16,
    bisect_precision: int = 16,
    think_scale: float = 0.005,
    sandbox_class: str = "microvm",
    worker_count: int = 1,
    worker_memory: str = "",
    actor_cpu: str = "250m",
    knee_multiplier: float = 2.0,
    knee_floor_ms: float = 1000.0,
    max_failure_ratio: float = 0.01,
    atelet_lag_s: int = 70,
    runner_cpu: str = "4",
    runner_memory: str = "8Gi",
    ate_args: list[str] | None = None,
    extra_flags: list[str] | None = None,
    dry_run: bool = False,
    skip_deploy: bool = False,
    keep_deployed: bool = False,
) -> list[SweepConfig]:
    """Builds the canonical 4-sweep microVM baseline matrix:
    {personal-assistant-light, personal-assistant} x {pause, suspend}."""
    base = Path(output_dir).resolve()
    configs: list[SweepConfig] = []
    for script, max_u in (
        ("personal-assistant-light", max_users_light),
        ("personal-assistant", max_users_full),
    ):
        short = "pa_light" if "light" in script else "pa_full"
        for mode in ("pause", "suspend"):
            run_name = f"{sandbox_class}_{short}_{mode}"
            configs.append(
                SweepConfig(
                    name=run_name,
                    dest=dest,
                    output_dir=str(base / run_name),
                    script=script,
                    think_scale=think_scale,
                    sandbox_class=sandbox_class,
                    lifecycle_mode=mode,
                    resume_mode="implicit",
                    worker_count=worker_count,
                    actor_cpu=actor_cpu,
                    worker_memory=worker_memory,
                    max_users=max_u,
                    seed_users=seed_users,
                    bisect_precision=bisect_precision,
                    knee_multiplier=knee_multiplier,
                    knee_floor_ms=knee_floor_ms,
                    max_failure_ratio=max_failure_ratio,
                    atelet_lag_s=atelet_lag_s,
                    runner_cpu=runner_cpu,
                    runner_memory=runner_memory,
                    ate_args=list(ate_args or []),
                    extra_flags=list(extra_flags or []),
                    step_duration=step_duration,
                    baseline_duration=baseline_duration,
                    dry_run=dry_run,
                    skip_deploy=skip_deploy,
                    keep_deployed=keep_deployed,
                )
            )
    return configs


def _load_repo_dev_env_defaults() -> None:
    """Loads .ate-dev-env.sh defaults into os.environ without overriding explicit vars."""
    env_file = REPO_ROOT / ".ate-dev-env.sh"
    if not env_file.is_file() or os.environ.get("NO_DEV_ENV"):
        return
    prior = dict(os.environ)
    orchestrator.source_env(str(env_file))
    os.environ.update(prior)


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("baseline-suite",), default=None,
                   help="Run the full 4-config {pa-light, pa} x {pause, suspend} matrix")
    p.add_argument("--name", default="microvm_pa_light_pause", help="Sweep identifier")
    p.add_argument("--dest", required=True, help="GCS gs://bucket/path for runner artifacts")
    p.add_argument("--output-dir", default="", help="Local directory for JSON + Markdown reports")
    p.add_argument("--prebuilt-image", default="", help="Optional pre-pushed runner image URI")
    p.add_argument("--test-file", default="/app/tests/agentsession.py")
    p.add_argument("--script", default="personal-assistant-light")
    p.add_argument("--think-scale", type=float, default=0.005)
    p.add_argument("--sandbox-class", default="microvm", choices=orchestrator.SANDBOX_CLASSES)
    p.add_argument(
        "--lifecycle-mode",
        default="async",
        choices=("async", "sync", "pause", "suspend", "none"),
    )
    p.add_argument("--resume-mode", default="implicit", choices=("implicit", "explicit"))
    p.add_argument("--worker-count", type=int, default=1)
    p.add_argument("--actor-memory", default="", help="Defaults to script min_actor_memory")
    p.add_argument("--actor-cpu", default="250m")
    p.add_argument("--worker-memory", default="")
    p.add_argument("--step-duration", default="6m")
    p.add_argument("--baseline-duration", default="4m")
    p.add_argument("--seed-users", type=int, default=16)
    p.add_argument("--max-users", type=int, default=1024)
    p.add_argument("--max-users-full", type=int, default=None,
                   help="Max users for full 1.5Gi personal-assistant in baseline-suite (defaults to min(max_users, 512))")
    p.add_argument("--bisect-precision", type=int, default=16)
    p.add_argument("--knee-multiplier", type=float, default=2.0)
    p.add_argument("--knee-floor-ms", type=float, default=1000.0)
    p.add_argument("--max-failure-ratio", type=float, default=0.01)
    p.add_argument("--atelet-lag-s", type=int, default=70)
    p.add_argument("--runner-cpu", default="4")
    p.add_argument("--runner-memory", default="8Gi")
    p.add_argument("--ate-arg", action="append", default=[], dest="ate_args")
    p.add_argument("--extra-flag", action="append", default=[], dest="extra_flags")
    p.add_argument("--skip-deploy", action="store_true")
    p.add_argument("--keep-deployed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _load_repo_dev_env_defaults()
    args = parse_cli_args(argv)
    out_dir = args.output_dir or os.path.join(
        tempfile.gettempdir(), f"substrate-sweep-{args.name}"
    )

    if args.preset == "baseline-suite":
        max_light = args.max_users
        max_full = (
            args.max_users_full
            if args.max_users_full is not None
            else min(args.max_users, 512)
        )
        configs = build_baseline_suite_configs(
            dest=args.dest,
            output_dir=out_dir,
            max_users_light=max_light,
            max_users_full=max_full,
            step_duration=args.step_duration,
            baseline_duration=args.baseline_duration,
            seed_users=args.seed_users,
            bisect_precision=args.bisect_precision,
            think_scale=args.think_scale,
            sandbox_class=args.sandbox_class,
            worker_count=args.worker_count,
            worker_memory=args.worker_memory,
            actor_cpu=args.actor_cpu,
            knee_multiplier=args.knee_multiplier,
            knee_floor_ms=args.knee_floor_ms,
            max_failure_ratio=args.max_failure_ratio,
            atelet_lag_s=args.atelet_lag_s,
            runner_cpu=args.runner_cpu,
            runner_memory=args.runner_memory,
            ate_args=args.ate_args,
            extra_flags=args.ate_args and args.extra_flags or args.extra_flags,
            dry_run=args.dry_run,
            skip_deploy=args.skip_deploy,
            keep_deployed=args.keep_deployed,
        )
        for c in configs:
            validate_and_finalize_config(c)

        commit = ""
        image = args.prebuilt_image
        if not args.dry_run:
            os.chdir(REPO_ROOT)
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            if not image:
                image = locust_type.build_image(commit)

        suite_results: list[dict[str, Any]] = []
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        try:
            if not args.dry_run and not args.skip_deploy:
                _deploy_standing_control_plane(configs[0])
            for c in configs:
                res = run_capacity_sweep(
                    c,
                    prebuilt_image=image,
                    commit=commit,
                    manage_control_plane=False,
                )
                suite_results.append(res)
                (Path(out_dir) / "suite_summary.json").write_text(
                    json.dumps(suite_results, indent=2) + "\n", encoding="utf-8"
                )
                if not args.dry_run:
                    (Path(out_dir) / "suite_report.md").write_text(
                        render_suite_markdown_report(suite_results), encoding="utf-8"
                    )
        finally:
            if not args.dry_run and not args.skip_deploy and not args.keep_deployed:
                _teardown_standing_control_plane()

        return 0

    cfg = SweepConfig(
        name=args.name,
        dest=args.dest,
        output_dir=out_dir,
        test_file=args.test_file,
        script=args.script,
        think_scale=args.think_scale,
        sandbox_class=args.sandbox_class,
        lifecycle_mode=args.lifecycle_mode,
        resume_mode=args.resume_mode,
        worker_count=args.worker_count,
        actor_memory=args.actor_memory,
        actor_cpu=args.actor_cpu,
        worker_memory=args.worker_memory,
        step_duration=args.step_duration,
        baseline_duration=args.baseline_duration,
        seed_users=args.seed_users,
        max_users=args.max_users,
        bisect_precision=args.bisect_precision,
        knee_multiplier=args.knee_multiplier,
        knee_floor_ms=args.knee_floor_ms,
        max_failure_ratio=args.max_failure_ratio,
        atelet_lag_s=args.atelet_lag_s,
        runner_cpu=args.runner_cpu,
        runner_memory=args.runner_memory,
        ate_args=list(args.ate_args),
        extra_flags=list(args.extra_flags),
        skip_deploy=args.skip_deploy,
        keep_deployed=args.keep_deployed,
        dry_run=args.dry_run,
    )
    run_capacity_sweep(cfg, prebuilt_image=args.prebuilt_image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
