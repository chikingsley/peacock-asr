import os


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


def test_default_input_targets_compact_backbones_compare() -> None:
    _set_test_env()

    from hatchet.workflows.p003 import default_p003_compact_backbones_input

    workflow_input = default_p003_compact_backbones_input()

    assert workflow_input.project_id == "P003 Compact Backbones"
    assert workflow_input.backend == "xlsr-espeak"
    assert workflow_input.seed == 501
    assert workflow_input.no_cache is True
    assert workflow_input.skip_prewarm is True
    assert workflow_input.reuse_existing is True


def test_extract_json_result_reads_marker() -> None:
    _set_test_env()

    from hatchet.workflows.p003 import _extract_json_result

    payload = _extract_json_result(
        "line one\nJSON_RESULT::{\"elapsed_s\": 12.5, \"metrics\": {\"pcc\": 0.5}}\n"
    )

    assert payload["elapsed_s"] == 12.5
    assert payload["metrics"]["pcc"] == 0.5


def test_summary_task_compares_python_vs_k2() -> None:
    _set_test_env()

    import hatchet.workflows.p003 as workflow

    result = workflow.summarize_xlsr53_compare.mock_run(
        workflow.default_p003_compact_backbones_input(),
        parent_outputs={
            "run_xlsr53_python_baseline": {
                "elapsed_s": 20.0,
                "metrics": {"pcc": 0.41, "mse": 0.92},
                "wandb_url": "https://wandb.ai/a/b/runs/python1",
            },
            "run_xlsr53_k2_baseline": {
                "elapsed_s": 10.0,
                "metrics": {"pcc": 0.42, "mse": 0.91},
                "wandb_url": "https://wandb.ai/a/b/runs/k2id",
            },
        },
    )

    assert result["k2_speedup_vs_python"] == 2.0
    assert result["python_pcc"] == 0.41
    assert result["k2_pcc"] == 0.42


def test_prewarm_skip_does_not_use_hatchet_skip_key() -> None:
    _set_test_env()

    import hatchet.workflows.p003 as workflow

    result = workflow.prewarm_xlsr53_k2.mock_run(
        workflow.P003CompactBackbonesInput(skip_prewarm=True),
    )

    assert result["prewarm_skipped"] is True
    assert "skipped" not in result
