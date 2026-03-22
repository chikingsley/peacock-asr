import os
import subprocess


def _set_test_env() -> None:
    os.environ.setdefault(
        "HATCHET_CLIENT_TOKEN",
        (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiJib290c3RyYXAiLCJzZXJ2ZXJfdXJsIjoiaHR0cDovLzEyNy4wLjAu"
            "MTo4ODg4IiwiZ3JwY19icm9hZGNhc3RfYWRkcmVzcyI6IjEyNy4wLjAuMTo3MDc3"
            "In0.signature"
        ),
    )
    os.environ.setdefault("HATCHET_CLIENT_TLS_STRATEGY", "none")


def test_default_input_targets_canonical_xlsr53_baseline() -> None:
    _set_test_env()

    from hatchet.workflows.p001_xlsr53 import default_p001_xlsr53_input

    workflow_input = default_p001_xlsr53_input()

    assert workflow_input.project_id == "P001"
    assert workflow_input.backend == "xlsr-espeak"
    assert workflow_input.agent_count == 5
    assert workflow_input.sweep_yaml.endswith("phase1_xlsr_a3_gopt.yaml")


def test_parse_sweep_id_accepts_wandb_cli_output() -> None:
    _set_test_env()

    from hatchet.workflows.p001_xlsr53 import parse_sweep_id

    output = """
    Create sweep with `wandb sweep path/to/file.yaml`
    Created sweep with ID: abc123xy
    Run sweep agent with: wandb agent peacockery/peacock-asr-p001-gop-baselines/abc123xy
    """

    assert parse_sweep_id(output) == "abc123xy"


def test_create_sweep_task_returns_sweep_metadata(monkeypatch) -> None:
    _set_test_env()

    import hatchet.workflows.p001_xlsr53 as workflow

    captured: dict[str, object] = {}

    def fake_run_checked(
        cmd: list[str], *, cwd, ctx
    ) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                "Created sweep with ID: abc123xy\n"
                "Run sweep agent with: "
                "wandb agent peacockery/peacock-asr-p001-gop-baselines/abc123xy\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(workflow, "_run_checked", fake_run_checked)

    result = workflow.create_xlsr53_sweep.mock_run(workflow.default_p001_xlsr53_input())

    assert result["sweep_id"] == "abc123xy"
    assert result["sweep_path"] == "peacockery/peacock-asr-p001-gop-baselines/abc123xy"
    assert str(captured["cwd"]).endswith("/peacock-asr")
    assert captured["cmd"] == [
        "uv",
        "run",
        "--project",
        "projects/P001-gop-baselines",
        "wandb",
        "sweep",
        "projects/P001-gop-baselines/experiments/sweeps/final/phase1_xlsr_a3_gopt.yaml",
    ]


def test_run_agent_task_uses_parent_sweep_output(monkeypatch) -> None:
    _set_test_env()

    import hatchet.workflows.p001_xlsr53 as workflow

    captured: dict[str, object] = {}

    def fake_run_checked(
        cmd: list[str], *, cwd, ctx
    ) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="agent complete\n",
            stderr="",
        )

    monkeypatch.setattr(workflow, "_run_checked", fake_run_checked)

    result = workflow.run_xlsr53_agent.mock_run(
        workflow.default_p001_xlsr53_input(),
        parent_outputs={
            "create_xlsr53_sweep": {
                "sweep_id": "abc123xy",
                "sweep_path": "peacockery/peacock-asr-p001-gop-baselines/abc123xy",
                "sweep_url": (
                    "https://wandb.ai/peacockery/"
                    "peacock-asr-p001-gop-baselines/sweeps/abc123xy"
                ),
            }
        },
    )

    assert result["status"] == "completed"
    assert result["sweep_id"] == "abc123xy"
    assert captured["cmd"] == [
        "uv",
        "run",
        "--project",
        "projects/P001-gop-baselines",
        "wandb",
        "agent",
        "--count",
        "5",
        "peacockery/peacock-asr-p001-gop-baselines/abc123xy",
    ]
