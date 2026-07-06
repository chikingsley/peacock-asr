"""Score Tajik model cards on an export's test split — delegates to the core eval.

  tajik-omni-eval                                  # base/v0/v3 on the v3 export's test split
  tajik-omni-eval --only-corpus-prefix youtube-            # the conversational held-out alone
  tajik-omni-eval --device cuda --models v3=omni_ctc_300m_v2_tajik_v3_step_20000
"""

from __future__ import annotations

from omni_finetune_core.project import eval_main

from tajik_asr.omni.train import PROJECT


def main(argv: list[str] | None = None) -> int:
    return eval_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
