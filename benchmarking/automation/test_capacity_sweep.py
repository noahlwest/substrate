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

"""Unit tests for capacity_sweep.py."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import yaml

import capacity_sweep
import orchestrator


def _make_stats_jsonl(
    wake_p50: float = 115.0,
    wake_p90: float = 165.0,
    wake_p99: float = 240.0,
    wake_rps: float = 2.0,
    suspend_p50: float = 180.0,
    suspend_p90: float = 245.0,
    failure_ratio: float = 0.0,
    actors_per_node: float = 64.0,
    active_p50: float = 4.0,
    node_cpu_p90: float = 28.0,
    node_ram_max_gb: float = 84.0,
    disk_write_max: float = 310.0,
    psi_io_p90: float = 0.4,
    psi_mem_p90: float = 0.0,
    oom_node: int = 0,
    use_explicit_resume: bool = False,
) -> str:
    rows = []
    if use_explicit_resume:
        rows.append({
            "metric": "grpc_ResumeActor",
            "measurements": {
                "p50": wake_p50,
                "p90": wake_p90,
                "p95": wake_p90 + 20.0,
                "p99": wake_p99,
                "request_count": 120,
                "failure_count": 0,
                "requests_per_s": wake_rps,
            },
        })
        rows.append({
            "metric": "grpc_SuspendActor",
            "measurements": {
                "p50": suspend_p50,
                "p90": suspend_p90,
                "p99": suspend_p90 + 40.0,
                "request_count": 120,
                "failure_count": 0,
                "requests_per_s": wake_rps,
            },
        })
    else:
        rows.append({
            "metric": "http_WakeFirstTouch",
            "measurements": {
                "p50": wake_p50,
                "p90": wake_p90,
                "p95": wake_p90 + 20.0,
                "p99": wake_p99,
                "request_count": 120,
                "failure_count": 0,
                "requests_per_s": wake_rps,
            },
        })
        rows.append({
            "metric": "grpc_SuspendActorAsync",
            "measurements": {
                "p50": suspend_p50,
                "p90": suspend_p90,
                "p99": suspend_p90 + 40.0,
                "request_count": 120,
                "failure_count": 0,
                "requests_per_s": wake_rps,
            },
        })

    rows.append({
        "metric": "trial_summary",
        "measurements": {
            "aggregate_failure_ratio": failure_ratio,
            "node_count": 1,
            "actors_per_node": actors_per_node,
            "machine_type": "c3-standard-176",
            "allocatable_cores": 175.6,
            "allocatable_ram_gb": 690.5,
            "worker_pod_count": 1,
            "worker_actor_capacity": 700,
            "worker_memory_limit_gb": 680.0,
            "sandbox_class": "microvm",
            "kubernetes_version": "v1.34.1-gke.100",
        },
    })
    rows.append({
        "metric": "server_summary",
        "measurements": {
            "active_actors_p50": active_p50,
            "active_actors_p90": active_p50 + 2.0,
            "active_actors_p99": active_p50 + 4.0,
            "snapshot_size_p50_mb": 124.5,
            "snapshot_size_p95_mb": 131.2,
            "checkpoint_mb_s": 540.0,
            "working_set_node_p90_gb": node_ram_max_gb * 0.9,
            "working_set_node_max_gb": node_ram_max_gb,
            "working_set_per_actor_p50_gb": 0.14,
            "node_cpu_cores_p90": node_cpu_p90,
            "node_cpu_cores_max": node_cpu_p90 + 8.0,
            "node_disk_read_mib_s_p90": 210.0,
            "node_disk_read_mib_s_max": 290.0,
            "node_disk_write_mib_s_p90": disk_write_max * 0.8,
            "node_disk_write_mib_s_max": disk_write_max,
            "node_disk_read_iops_p90": 1850.0,
            "node_disk_write_iops_p90": 2400.0,
            "node_net_rx_mib_s_p90": 42.5,
            "node_net_tx_mib_s_p90": 38.0,
            "psi_cpu_stall_p90": 0.8,
            "psi_mem_stall_p90": psi_mem_p90,
            "psi_io_stall_p90": psi_io_p90,
            "oom_events_node": oom_node,
            "oom_events_pod": 0,
            "oom_events_worker_restarts": 0,
            "oom_events_worker_oomkilled_pods": 0,
            "oom_events_evicted": 0,
        },
    })
    rows.append({
        "metric": "phase_atelet_restore_microvm_pause_actor_restore_vm",
        "measurements": {
            "layer": "atelet",
            "op": "restore",
            "class": "microvm",
            "kind": "pause",
            "scope": "actor",
            "phase": "restore_vm",
            "count": 120,
            "p50_ms": 48.2,
            "p90_ms": 63.5,
            "p95_ms": 71.0,
            "max_ms": 94.0,
        },
    })
    rows.append({
        "metric": "phase_breakdown_summary",
        "measurements": {"phase_count": 1},
    })
    return "\n".join(json.dumps(r) for r in rows) + "\n"


class ScriptAndConfigValidationTest(unittest.TestCase):
    def test_loads_real_checked_in_scripts(self):
        pa_light = capacity_sweep.load_agentsession_script_info("personal-assistant-light")
        self.assertEqual(pa_light.min_actor_memory, "256Mi")
        self.assertEqual(pa_light.min_actor_memory_bytes, 256 * 1024 * 1024)
        self.assertEqual(pa_light.step_count, 61)
        self.assertAlmostEqual(pa_light.unscaled_think_s, 86400.0)
        self.assertGreater(pa_light.unscaled_active_s, 400.0)

        pa_full = capacity_sweep.load_agentsession_script_info("personal-assistant")
        self.assertEqual(pa_full.min_actor_memory, "1536Mi")
        self.assertEqual(pa_full.min_actor_memory_bytes, 1536 * 1024 * 1024)
        self.assertEqual(pa_full.step_count, 61)
        self.assertAlmostEqual(pa_full.unscaled_think_s, 86400.0)

        coding = capacity_sweep.load_agentsession_script_info("coding-session")
        self.assertEqual(coding.min_actor_memory, "1Gi")
        self.assertEqual(coding.step_count, 20)

    def test_rejects_path_traversal_and_missing_script(self):
        with self.assertRaises(ValueError):
            capacity_sweep.load_agentsession_script_info("../personal-assistant")
        with self.assertRaises(ValueError):
            capacity_sweep.load_agentsession_script_info("does-not-exist")

    def test_auto_defaults_and_enforces_min_actor_memory(self):
        cfg = capacity_sweep.SweepConfig(
            name="pa_full_test",
            dest="gs://bucket/prefix",
            output_dir="/tmp/out",
            script="personal-assistant",
        )
        info = capacity_sweep.validate_and_finalize_config(cfg)
        self.assertIsNotNone(info)
        self.assertEqual(cfg.actor_memory, "1536Mi")

        too_small = capacity_sweep.SweepConfig(
            name="pa_full_bad_mem",
            dest="gs://bucket/prefix",
            output_dir="/tmp/out",
            script="personal-assistant",
            actor_memory="1Gi",
        )
        with self.assertRaisesRegex(ValueError, "below script .* min_actor_memory 1536Mi"):
            capacity_sweep.validate_and_finalize_config(too_small)

    def test_tests_yaml_passes_orchestrator_validation(self):
        raw = yaml.safe_load(
            (capacity_sweep.REPO_ROOT / "benchmarking" / "automation" / "tests.yaml").read_text(
                encoding="utf-8"
            )
        )
        tests = raw["tests"]
        orchestrator.validate_and_normalize_tests(tests)
        names = {t["name"]: t for t in tests}
        for expected in (
            "agentsession_pa_light_microvm_pause",
            "agentsession_pa_light_microvm_suspend",
            "agentsession_pa_microvm_pause",
            "agentsession_pa_microvm_suspend",
        ):
            self.assertIn(expected, names)
            self.assertEqual(names[expected]["sandboxClass"], "microvm")


class SubscriptionPlannerTest(unittest.TestCase):
    def test_exponential_then_binary_bisection_convergence(self):
        # True capacity sits between 96 (PASS) and 112 (FAIL).
        tested: dict[int, bool] = {}
        sequence: list[int] = []
        while True:
            nxt = capacity_sweep.next_subscription_step(
                tested, seed_users=16, max_users=1024, bisect_precision=16
            )
            if nxt is None:
                break
            sequence.append(nxt)
            tested[nxt] = nxt <= 100
        self.assertEqual(sequence, [1, 16, 32, 64, 128, 96, 112])
        self.assertEqual(max(n for n, ok in tested.items() if ok), 96)

    def test_aborts_immediately_when_baseline_n1_fails(self):
        self.assertIsNone(
            capacity_sweep.next_subscription_step(
                {1: False}, seed_users=16, max_users=512, bisect_precision=16
            )
        )

    def test_stops_cleanly_at_max_users_ceiling(self):
        tested: dict[int, bool] = {}
        sequence: list[int] = []
        while True:
            nxt = capacity_sweep.next_subscription_step(
                tested, seed_users=16, max_users=100, bisect_precision=16
            )
            if nxt is None:
                break
            sequence.append(nxt)
            tested[nxt] = True
        self.assertEqual(sequence, [1, 16, 32, 64, 100])


class ParsingEvaluationAndBottleneckTest(unittest.TestCase):
    def test_parses_implicit_and_explicit_artifacts_and_duty_cycle(self):
        implicit = capacity_sweep.parse_trial_artifacts(
            users=64,
            trial_name="sweep_n64",
            stats_jsonl_text=_make_stats_jsonl(wake_p90=210.0, active_p50=4.0, wake_rps=2.0),
        )
        self.assertEqual(implicit.wake_metric_name, "http_WakeFirstTouch")
        self.assertEqual(implicit.suspend_metric_name, "grpc_SuspendActorAsync")
        self.assertAlmostEqual(implicit.duty_cycle, 4.0 / 64.0, places=4)
        self.assertAlmostEqual(implicit.oversubscription_ratio, 16.0, places=2)
        self.assertAlmostEqual(implicit.empirical_cycle_s, 32.0, places=2)
        self.assertEqual(len(implicit.phases), 1)
        self.assertEqual(implicit.phases[0]["phase"], "restore_vm")
        self.assertAlmostEqual(implicit.node_disk_write_mib_s_max, 310.0)

        explicit = capacity_sweep.parse_trial_artifacts(
            users=1,
            trial_name="sweep_n1",
            stats_jsonl_text=_make_stats_jsonl(use_explicit_resume=True, actors_per_node=1.0),
        )
        self.assertEqual(explicit.wake_metric_name, "grpc_ResumeActor")
        self.assertEqual(explicit.suspend_metric_name, "grpc_SuspendActor")

    def test_bottleneck_classification_categories(self):
        n1 = capacity_sweep.parse_trial_artifacts(
            1, "n1", _make_stats_jsonl(wake_p90=150.0, actors_per_node=1.0)
        )
        n64_io = capacity_sweep.parse_trial_artifacts(
            64,
            "n64",
            _make_stats_jsonl(wake_p90=1400.0, disk_write_max=980.0, psi_io_p90=12.5),
        )
        evals = {
            1: capacity_sweep.evaluate_step(n1, knee_limit_ms=1000.0),
            64: capacity_sweep.evaluate_step(n64_io, knee_limit_ms=1000.0),
        }
        self.assertTrue(evals[1].passed)
        self.assertFalse(evals[64].passed)
        bn_io = capacity_sweep.diagnose_bottleneck([n1, n64_io], evals)
        self.assertEqual(bn_io["category"], "storage_io_bandwidth")
        self.assertIn("disk_r/w_max=", bn_io["summary"])

        n64_oom = capacity_sweep.parse_trial_artifacts(
            64, "n64_oom", _make_stats_jsonl(wake_p90=300.0, oom_node=2)
        )
        eval_oom = capacity_sweep.evaluate_step(n64_oom, knee_limit_ms=1000.0)
        self.assertFalse(eval_oom.passed)
        bn_mem = capacity_sweep.diagnose_bottleneck([n1, n64_oom], {1: evals[1], 64: eval_oom})
        self.assertEqual(bn_mem["category"], "memory_capacity")

    def test_read_latest_stats_jsonl_gcloud_fallback(self):
        sample_jsonl = _make_stats_jsonl(wake_p90=195.0, actors_per_node=1.0)
        ls_output = "\n".join([
            "gs://example-bench/smoke/runs/smoke_n1/run_date=2026-10-20/run_ts=100/run_tag=abc123/stats.jsonl",
            "gs://example-bench/smoke/runs/smoke_n1/run_date=2026-10-20/run_ts=200/run_tag=abc123/stats.jsonl",
        ])

        def fake_subprocess_run(cmd, **_kwargs):
            if cmd[:3] == ["gcloud", "storage", "ls"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=ls_output, stderr="")
            if cmd[:3] == ["gcloud", "storage", "cat"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=sample_jsonl, stderr="")
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        with (
            mock.patch.dict("sys.modules", {"google": None, "google.cloud": None}),
            mock.patch("capacity_sweep.subprocess.run", side_effect=fake_subprocess_run),
        ):
            text, prefix = capacity_sweep.read_latest_stats_jsonl(
                "gs://example-bench/smoke", "smoke_n1", "abc123"
            )
        self.assertEqual(text, sample_jsonl)
        self.assertEqual(
            prefix,
            "gs://example-bench/smoke/runs/smoke_n1/run_date=2026-10-20/run_ts=200/run_tag=abc123",
        )


class EndToEndSweepAndDryRunTest(unittest.TestCase):
    def test_dry_run_single_and_baseline_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = capacity_sweep.main([
                "--preset",
                "baseline-suite",
                "--dest",
                "gs://example-bench/baseline",
                "--output-dir",
                tmp,
                "--worker-count",
                "2",
                "--worker-memory",
                "340Gi",
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            suite = json.loads((Path(tmp) / "suite_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(suite), 4)
            by_name = {entry["config"]["name"]: entry for entry in suite}
            self.assertEqual(
                by_name["microvm_pa_light_pause"]["config"]["actor_memory"], "256Mi"
            )
            self.assertEqual(
                by_name["microvm_pa_full_suspend"]["config"]["actor_memory"], "1536Mi"
            )
            self.assertEqual(
                by_name["microvm_pa_full_suspend"]["config"]["worker_memory"], "340Gi"
            )
            self.assertIn(
                "--agentsession-script",
                by_name["microvm_pa_light_pause"]["sample_n1_job_manifest"],
            )

    def test_compute_spawn_rate_scales_without_stalling_large_sweeps(self):
        self.assertEqual(capacity_sweep.compute_spawn_rate(1), 1)
        self.assertEqual(capacity_sweep.compute_spawn_rate(64), 16)
        self.assertEqual(capacity_sweep.compute_spawn_rate(1024), 16)
        self.assertEqual(capacity_sweep.compute_spawn_rate(2048), 23)
        self.assertEqual(capacity_sweep.compute_spawn_rate(4096), 46)

    def test_purge_straggler_actors_deletes_only_agent_sessions(self):
        actors_payload = json.dumps({
            "actors": [
                {"metadata": {"name": "golden-glutton-001"}},
                {"metadata": {"name": "agent-1111"}},
                {"metadata": {"name": "agent-2222"}},
            ]
        })
        deleted_cmds: list[list[str]] = []

        def fake_run(cmd, **_kwargs):
            if cmd[1:3] == ["get", "actor"]:
                remaining = [
                    a for a in json.loads(actors_payload)["actors"]
                    if a["metadata"]["name"] not in {c[3] for c in deleted_cmds}
                ]
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=json.dumps({"actors": remaining}), stderr=""
                )
            if cmd[1:3] == ["delete", "actor"]:
                deleted_cmds.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            raise AssertionError(f"unexpected cmd: {cmd}")

        with (
            mock.patch("capacity_sweep._ensure_kubectl_ate_binary", return_value="/tmp/kubectl-ate"),
            mock.patch("capacity_sweep.subprocess.run", side_effect=fake_run),
        ):
            deleted = capacity_sweep.purge_straggler_actors()
        self.assertEqual(deleted, 2)
        self.assertEqual(
            sorted(c[3] for c in deleted_cmds), ["agent-1111", "agent-2222"]
        )
        for c in deleted_cmds:
            self.assertIn("--any-state", c)

    @mock.patch("capacity_sweep.purge_straggler_actors", return_value=0)
    @mock.patch("capacity_sweep.time.sleep")
    @mock.patch("capacity_sweep.orchestrator.teardown_substrate")
    @mock.patch("capacity_sweep.orchestrator.teardown_microvm_deps")
    @mock.patch("capacity_sweep.orchestrator.teardown_workloads")
    @mock.patch("capacity_sweep.orchestrator.deploy_workloads")
    @mock.patch("capacity_sweep.orchestrator.install_microvm_deps")
    @mock.patch("capacity_sweep.orchestrator.deploy_substrate")
    @mock.patch("capacity_sweep.run_no_check")
    @mock.patch("capacity_sweep.run")
    @mock.patch("capacity_sweep.orchestrator.run_test")
    def test_full_mocked_sweep_writes_reports_and_wires_otlp_meter(
        self,
        mock_run_test,
        mock_run,
        mock_run_no_check,
        mock_deploy_sub,
        mock_microvm_deps,
        mock_deploy_wl,
        mock_td_wl,
        mock_td_uvm,
        mock_td_sub,
        _mock_sleep,
        mock_purge,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            dest_dir = Path(tmp) / "dest"
            out_dir = Path(tmp) / "reports"
            commit = "abc1234"

            def fake_run_test(test, image, dest, commit, manifests_dir):
                u = test["users"]
                p90 = 180.0 if u <= 32 else 1500.0
                stats_file = (
                    Path(dest)
                    / "runs"
                    / test["name"]
                    / "run_date=2026-10-19"
                    / "run_ts=2026-10-19T20:00:00Z"
                    / f"run_tag={commit}"
                    / "stats.jsonl"
                )
                stats_file.parent.mkdir(parents=True, exist_ok=True)
                stats_file.write_text(
                    _make_stats_jsonl(wake_p90=p90, actors_per_node=float(u)),
                    encoding="utf-8",
                )
                return "complete"

            mock_run_test.side_effect = fake_run_test

            cfg = capacity_sweep.SweepConfig(
                name="uvm_smoke",
                dest=str(dest_dir),
                output_dir=str(out_dir),
                seed_users=16,
                max_users=64,
                bisect_precision=16,
            )
            summary = capacity_sweep.run_capacity_sweep(
                cfg, prebuilt_image="gcr.io/example/locust:abc1234", commit=commit
            )
            self.assertEqual(summary["n_max"], 32)
            self.assertEqual([s["users"] for s in summary["steps"]], [1, 16, 32, 48, 64])
            self.assertTrue((out_dir / "sweep_summary.json").is_file())
            report_md = (out_dir / "baseline_report.md").read_text(encoding="utf-8")
            self.assertIn("32 subscriptions", report_md)
            self.assertIn("restore_vm", report_md)

            # Verify OTLP meter endpoint & monitoring manifests were wired.
            applied_Cmds = [c.args[0] for c in mock_run.call_args_list]
            self.assertIn(
                ["kubectl", "apply", "-f", str(capacity_sweep.MONITORING_MANIFEST)],
                applied_Cmds,
            )
            self.assertIn(
                ["kubectl", "apply", "-f", str(capacity_sweep.TELEMETRY_METER_MANIFEST)],
                applied_Cmds,
            )
            ate_args = mock_deploy_sub.call_args.args[0]
            self.assertIn(
                f"--otlp-endpoint={capacity_sweep.TELEMETRY_METER_OTLP_URL}",
                ate_args,
            )
            mock_microvm_deps.assert_called_once()
            mock_deploy_wl.assert_called_once()
            mock_td_wl.assert_called_once()
            mock_td_uvm.assert_called_once()
            mock_td_sub.assert_called_once()
            mock_run_no_check.assert_called_once()

    def test_diagnose_checkpoint_persistence_timeout_beats_cgroup_psi_cpu(self) -> None:
        base = capacity_sweep.StepMetrics(
            users=1,
            trial_name="t_n1",
            wake_p90_ms=690.0,
            wake_requests=10,
            aggregate_failure_ratio=0.0,
            hardware_facts={"allocatable_cores": "287.21"},
        )
        fail = capacity_sweep.StepMetrics(
            users=64,
            trial_name="t_n64",
            wake_p90_ms=720.0,
            wake_requests=80,
            suspend_p50_ms=49000.0,
            suspend_p90_ms=60000.0,
            aggregate_failure_ratio=0.2114,
            node_cpu_cores_p90=4.7,
            psi_cpu_stall_p90=20.11,
            psi_io_stall_p90=0.44,
            node_disk_write_mib_s_max=150.0,
            hardware_facts={"allocatable_cores": "287.21"},
        )
        evs = {
            1: capacity_sweep.evaluate_step(base, knee_limit_ms=1380.0),
            64: capacity_sweep.evaluate_step(fail, knee_limit_ms=1380.0),
        }
        bn = capacity_sweep.diagnose_bottleneck([base, fail], evs)
        self.assertEqual(bn["category"], "checkpoint_persistence_timeout")

    def test_diagnose_severe_io_stall_with_suspend_timeout_is_storage_io_bandwidth(self) -> None:
        base = capacity_sweep.StepMetrics(
            users=1,
            trial_name="t_n1",
            wake_p90_ms=2000.0,
            wake_requests=4,
            aggregate_failure_ratio=0.0,
            hardware_facts={"allocatable_cores": "191.45"},
        )
        fail = capacity_sweep.StepMetrics(
            users=32,
            trial_name="t_n32",
            wake_p90_ms=2000.0,
            wake_requests=109,
            suspend_p50_ms=2100.0,
            suspend_p90_ms=60000.0,
            aggregate_failure_ratio=0.1935,
            node_cpu_cores_p90=2.5,
            psi_cpu_stall_p90=21.38,
            psi_io_stall_p90=84.41,
            node_disk_write_mib_s_max=489.5,
            hardware_facts={"allocatable_cores": "191.45"},
        )
        evs = {
            1: capacity_sweep.evaluate_step(base, knee_limit_ms=4000.0),
            32: capacity_sweep.evaluate_step(fail, knee_limit_ms=4000.0),
        }
        bn = capacity_sweep.diagnose_bottleneck([base, fail], evs)
        self.assertEqual(bn["category"], "storage_io_bandwidth")

    def test_diagnose_nvme_high_throughput_low_psi_io_is_not_storage_saturated(self) -> None:
        base = capacity_sweep.StepMetrics(
            users=1,
            trial_name="t_n1",
            wake_p90_ms=650.0,
            wake_requests=6,
            aggregate_failure_ratio=0.0,
            hardware_facts={"allocatable_cores": "287.21"},
        )
        queueing_knee = capacity_sweep.StepMetrics(
            users=256,
            trial_name="t_n256",
            wake_p90_ms=1400.0,
            wake_requests=3570,
            suspend_p50_ms=610.0,
            suspend_p90_ms=1100.0,
            aggregate_failure_ratio=0.0003,
            node_cpu_cores_p90=34.8,
            psi_cpu_stall_p90=28.88,
            psi_io_stall_p90=2.60,
            node_disk_write_mib_s_max=2888.5,
            hardware_facts={"allocatable_cores": "287.21"},
        )
        admission_knee = capacity_sweep.StepMetrics(
            users=64,
            trial_name="t_n64",
            wake_p90_ms=2500.0,
            wake_requests=324,
            suspend_p50_ms=2500.0,
            suspend_p90_ms=2900.0,
            aggregate_failure_ratio=0.0848,
            node_cpu_cores_p90=11.5,
            psi_cpu_stall_p90=49.38,
            psi_io_stall_p90=0.43,
            node_disk_write_mib_s_max=1172.5,
            hardware_facts={"allocatable_cores": "287.21"},
        )
        evs_q = {
            1: capacity_sweep.evaluate_step(base, knee_limit_ms=1300.0),
            256: capacity_sweep.evaluate_step(queueing_knee, knee_limit_ms=1300.0),
        }
        bn_q = capacity_sweep.diagnose_bottleneck([base, queueing_knee], evs_q)
        self.assertEqual(bn_q["category"], "snapshot_restore_queueing")

        evs_a = {
            1: capacity_sweep.evaluate_step(base, knee_limit_ms=4600.0),
            64: capacity_sweep.evaluate_step(admission_knee, knee_limit_ms=4600.0),
        }
        bn_a = capacity_sweep.diagnose_bottleneck([base, admission_knee], evs_a)
        self.assertEqual(bn_a["category"], "control_plane_or_slot_admission")


if __name__ == "__main__":
    unittest.main()
