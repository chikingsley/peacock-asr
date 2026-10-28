"""Historical export entry point, superseded by the shared split-safe materializer."""

raise SystemExit(
    "Use: uv run --project projects/tajik-asr tajik-parakeet-prepare --all-splits. "
    "The historical SQL export bypassed the frozen evaluation-video policy."
)
