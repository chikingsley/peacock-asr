"""Score Russian model cards on an export's test split — delegates to the core eval.

  russian-eval                                    # base on the v0 export's test split
  russian-eval --models ft=<trained card> --device cuda
  russian-eval --only-corpus-prefix youtube-      # conversational held-out alone
"""

from __future__ import annotations

from omni_finetune_core.project import eval_main

from russian_asr.omni.train import PROJECT


def main(argv: list[str] | None = None) -> int:
    return eval_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
