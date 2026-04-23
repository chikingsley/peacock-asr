# 06 — Defensive Programming Audit

Scope: `src/p014/` (~5200 LOC) + `tests/`. P014 is a research / reproduction project, not a long-running service.

## 1. Executive summary

Defensive programming is **mild and mostly well-targeted**. There are no bare `except:`, no `contextlib.suppress`, and no broad `except Exception: pass`. The only error-hiding hotspot is the Trackio integration in `training.py`, which catches `Exception` at four sites and silently downgrades logging.

Verdict:

- 5 of the 9 `try/except` blocks are **justified** (JSON sidecar parse on cache read — corrupt/partial file on disk is a real external failure mode).
- 4 of the 9 `try/except` blocks are in the Trackio log path and catch `Exception` broadly. The **import fallback is dead code** (trackio is a required dep). The runtime `log`/`finish`/`init` catches are defensible for a research logging sidecar but should narrow the exception type and re-raise on programming errors.
- 3 `hasattr(audio_payload, "get_all_samples")` checks are **justified** — HF `datasets.Audio` column legitimately yields either a torchcodec `AudioDecoder` or a pre-decoded dict depending on backend/version.
- `getattr(args, ..., None)` in `cli.py` is mildly redundant but harmless (argparse defaults already guarantee the attribute).
- None of the `is None` guards in this codebase are defensive in the "mask a bug" sense — they are either type-correct (the value is genuinely `Optional`) or raise `RuntimeError`/`ValueError` immediately after.

User rule "Everything logs to W&B" is *not* about Trackio, but the Trackio fallback pattern still swallows real errors in the paper-reproduction run and should be tightened.

## 2. Inventory

| Pattern | Count | Files |
|---|---|---|
| `try` / `except` | 9 | data.py (1), training.py (4), freespeak/asr.py (1), features/ssl_utterance.py (1), features/ctc_gop.py (2) |
| `contextlib.suppress` / `with suppress` | 0 | — |
| `hasattr(...)` | 3 | features/ssl_utterance.py, features/ctc_gop.py, freespeak/asr.py |
| `getattr(x, ..., default)` | 3 | cli.py |
| `if x is None: return` (or similar bail-out) | 0 non-justified | — (all `is None` sites are either legitimate `Optional` dispatch or raise explicit errors) |
| Bare `except:` | 0 | — |
| `except Exception` (broad) | 4 | training.py (all Trackio) |

## 3. Findings — REMOVE

### 3.1 Trackio import fallback is dead code

- **Location**: `src/p014/training.py:88-96`
- **Current**:

  ```python
  def _init_trackio(settings, seed) -> _TrackioHandle:
      try:
          trackio_module = cast(Any, import_module("trackio"))
      except Exception as exc:
          print(f"[trackio] import failed ({exc}); continuing with local-only logging.", file=sys.stderr)
          return _TrackioHandle(active=False)
  ```

- **Why unnecessary**: `trackio>=0.3.0` is listed in `pyproject.toml` *required* `dependencies` (line 11), not `optional-dependencies.train`. If trackio fails to import, the environment is broken and the training run should halt, not silently produce an unreliable local log.
- **Proposed replacement**: `import trackio` at module top-level. Remove `_TrackioHandle.active` flag; the handle is always live. If someone really wants an offline mode, make it an explicit `--no-trackio` flag and branch on that, not on an import error.
- **Confidence**: HIGH

### 3.2 Trackio `log()` swallows exceptions

- **Location**: `src/p014/training.py:69-77`
- **Current**:

  ```python
  if self.active and self.module is not None:
      try:
          self.module.log(metrics, step=step)
      except Exception as exc:  # pragma: no cover - network/runtime issues
          print(f"[trackio] log failed, switching to local-only mode: {exc}", file=sys.stderr)
          self.active = False
  ```

- **Why unnecessary**: This is a broad `except Exception` that hides programming errors (bad metric shape, serialization mismatch) as "network issues." The user rule states experiment tracking is non-negotiable; a swallowed log means the run proceeds with missing training curves, undetected until the user tries to compare runs.
- **Proposed replacement**: Narrow to network/IO exceptions only (`requests.RequestException`, `ConnectionError`, `TimeoutError`), or remove entirely and let it raise. At minimum, re-raise on `TypeError`/`ValueError` (programming errors).
- **Confidence**: HIGH

### 3.3 Trackio `finish()` swallow

- **Location**: `src/p014/training.py:79-85`
- **Current**: `try: self.module.finish(); except Exception as exc: print(...)`
- **Why unnecessary**: Same broad-catch problem. `finish()` failures could indicate a broken run state worth surfacing.
- **Proposed replacement**: Remove the try/except. If trackio's `finish()` raises, the traceback tells the user why; research scripts should fail loudly.
- **Confidence**: MEDIUM (finish-on-exit is borderline — losing a trace at shutdown is arguably tolerable. But log it via `traceback.print_exc()` rather than `{exc}`.)

### 3.4 Trackio `init()` swallow

- **Location**: `src/p014/training.py:107-114`
- **Current**: `try: trackio_module.init(**init_kwargs); except Exception: ... return _TrackioHandle(active=False, module=trackio_module)`
- **Why unnecessary**: This silently converts a misconfigured trackio project / bad Space ID / bad token into a local-only run. The user rule "everything logs to W&B/experiment tracker" is violated without the user noticing.
- **Proposed replacement**: Let it raise. If an offline mode is genuinely needed, add an explicit `settings.disable_tracking` flag.
- **Confidence**: HIGH

### 3.5 Redundant `getattr(args, ..., None)` defaults

- **Location**: `src/p014/cli.py:111, 118, 120`
- **Current**:

  ```python
  value = getattr(args, arg_name, None)
  ...
  if getattr(args, "use_curriculum", None) is None:
  if getattr(args, "use_cono", None) is None:
  ```

- **Why unnecessary**: Every argument listed in `flag_to_field` is declared via `parser.add_argument(...)` in `_add_training_flags`. argparse guarantees the attribute exists on the Namespace (defaulting to `None` for most flags and explicit `default=None` for the boolean pairs at lines 36, 37, 61, 67). The `getattr` default cannot fire.
- **Proposed replacement**: Use direct attribute access: `value = args.__dict__.get(arg_name)` if you want dict-style, or better, iterate over `vars(args).items()` filtered by `flag_to_field`. For the two explicit checks, use `args.use_curriculum` / `args.use_cono`.
- **Confidence**: HIGH (mild — pattern is harmless but dishonest about what shape `args` has)

## 4. Findings — KEEP

### 4.1 JSON sidecar parse guards (5 sites)

- **Locations**: `data.py:431-435`, `freespeak/asr.py:108-111`, `features/ssl_utterance.py:81-84`, `features/ctc_gop.py:275-278`, `features/ctc_gop.py:544-547`.
- **Pattern**: `try: json.loads(sidecar_path.read_text(...)); except (OSError, ValueError): ...`
- **Why justified**: Deserialisation of an *external* file written by a previous run. A half-written or manually corrupted sidecar should invalidate the cache and trigger recomputation, not abort. Exception types are narrowed (`OSError`, `ValueError`) — no broad `Exception`.
- **Verdict**: KEEP.

### 4.2 `hasattr(audio_payload, "get_all_samples")` (3 sites)

- **Locations**: `features/ssl_utterance.py:95`, `features/ctc_gop.py:439`, `freespeak/asr.py:65`.
- **Why justified**: HF `datasets.Audio` column dispatches to either a torchcodec `AudioDecoder` object (has `get_all_samples`) or a pre-decoded `{"array", "sampling_rate", "path"}` dict depending on datasets/torchcodec version. This is runtime polymorphism over an external library's return type, not defensive guessing. An `isinstance` check would couple to torchcodec's internal class.
- **Verdict**: KEEP. (Optional improvement: wrap in a small `_decode_audio(payload)` helper with a docstring explaining the two branches.)

### 4.3 `is None` checks that immediately raise

- **Locations**: `training.py:240, 351, 421, 435`; `features/ctc_gop.py:844, 846`.
- **Pattern**: `if x is None: raise RuntimeError(...)`
- **Why justified**: These are type-narrowing assertions required because the attribute is genuinely `Optional[...]` at the type level but is guaranteed non-None by upstream control flow. Raising `RuntimeError` with a precise message is the opposite of error-hiding.
- **Verdict**: KEEP. (Could be asserted with `assert x is not None` in a non-research project, but the current `raise RuntimeError` is actually stronger because it survives `python -O`.)

### 4.4 Optional dispatch guards in `blocks.py`, `training.py`, `freespeak/annotation.py`

- `blocks.py:77` (`if hidden_dim is None: compute default`)
- `blocks.py:239, 255` (`mask is None` → masked vs unmasked forward path)
- `freespeak/annotation.py:58, 75, 84, 111` (alignment ops where `ref_index`/`hyp_index` is genuinely `None` for insertions/deletions)
- **Verdict**: KEEP. Pure Optional dispatch, not defence.

## 5. Error-hiding antipatterns

Only one cluster: the four Trackio `except Exception` sites listed in §3.1–3.4. No bare excepts, no `except: pass`, no `except Exception as e: logger.error(e); return None` elsewhere. All other exception handlers narrow the type.

## 6. Fallback-imports audit

| Site | Pattern | Required dep? | Verdict |
|---|---|---|---|
| `training.py:89-96` | `try: import_module("trackio"); except Exception` | `trackio>=0.3.0` in required `dependencies` (pyproject.toml line 11) | **Dead code — REMOVE** |

No other `try: import` fallbacks exist in the codebase. `transformers`, `torch`, `torchaudio`, `datasets`, etc. are imported directly (they're in `optional-dependencies.train`, but the project assumes the `[train]` extra is installed for any training path — failing loudly on missing deps is correct).

## Appendix — verification commands

```bash
rg -n 'try:|except ' src/ tests/          # 9 hits
rg -n 'contextlib\.suppress|with suppress' src/ tests/   # 0 hits
rg -n 'hasattr\(|getattr\(' src/          # 6 hits (3 hasattr + 3 getattr)
rg -n 'bare except|except:$' src/         # 0 hits
```
