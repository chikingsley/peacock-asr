"""Compatibility wrapper for the promoted Farsi Omni KenLM eval command."""

from farsi_asr.omni.lm_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
