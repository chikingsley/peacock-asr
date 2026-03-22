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


def test_hatchet_modules_import() -> None:
    _set_test_env()
    __import__("hatchet.worker.main")
    __import__("hatchet.workflows.p001_xlsr53")
    __import__("hatchet.workflows.p003")
