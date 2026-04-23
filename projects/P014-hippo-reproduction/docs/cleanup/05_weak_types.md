# P014 — Weak Type Audit

## 1. Executive summary

The codebase (~7 000 LOC across `src/p014/`) is mostly well-typed in its
pydantic config layer, model/blocks code, and public dataclasses. Weak typing
is **concentrated almost entirely in a single pattern**: HF `transformers` /
`datasets` boundary code. Six files (`data.py`, `training.py`, `cli.py`,
`features/ctc_gop.py`, `features/ssl_utterance.py`, `freespeak/asr.py`,
`freespeak/g2p.py`) contain the vast majority of `Any` / `cast` / `object`
noise, always in the same shape:

```python
module_any = cast(Any, import_module("transformers"))
model_any: Any = module_any.SomeModel.from_pretrained(...)
```

Essentially every `Any`/`cast` in the repo is a consequence of either (a)
`import_module` being used to keep `transformers` a soft dependency,
(b) HF datasets not shipping precise `__getitem__` overloads, or (c) two
private dataclasses (`_TrackioHandle`, audio payloads) that could be a
`Protocol`. There are **no `# type: ignore` comments** anywhere, which is
excellent. `ty check` reports only **2** real diagnostics — both in
`features/ctc_gop.py` (`int(payload["..."])` where payload is
`dict[str, object]`). Ruff `ANN401` flags exactly **3** sites.

## 2. Counts

| Category | Count | Notes |
|---|---|---|
| `Any` occurrences | 69 | 8 files; all at HF/import boundaries |
| `cast(...)` calls | 70 | ~55 are library-boundary narrowing; ~15 are legitimate payload unpacks |
| `object` annotations | 5 | 3 function params (`tokenizer/model/processor`, `dataset`) |
| `# type: ignore` | 0 | Clean |
| Missing annotations (ruff ANN) | 3 | All `ANN401` on `_load_waveform(audio_payload: Any)` |
| `ty` errors | 2 | `int(object)` at `ctc_gop.py:354,809` |
| Bare generics (`dict`, `list`, `tuple` without params) | 0 | Good — all parameterized |
| `Callable` without params | 0 | No `Callable` used |
| `TYPE_CHECKING`-only imports | 0 | None present |
| `Optional[X]` (legacy form) | 0 | All use `X \| None` |

## 3. Findings

### 3.1 HF `transformers` module boundary (very high frequency, same pattern)

**Locations** (all near-identical):

- `data.py:453-457` — `AutoTokenizer`, `AutoModel`
- `features/ctc_gop.py:635,642-648` — `Wav2Vec2CTCTokenizer`, `Wav2Vec2Processor`, `Wav2Vec2ForCTC`
- `features/ssl_utterance.py:128-132` — `AutoFeatureExtractor`, `AutoModel`
- `freespeak/asr.py:176-178` — `AutoProcessor`, `AutoModelForSpeechSeq2Seq`

**Current**:

```python
transformers_module = cast(Any, import_module("transformers"))
model_any: Any = transformers_module.Wav2Vec2ForCTC.from_pretrained(model_source)
```

**Proposed strong type**: Import the classes directly (not through
`import_module`) at module top level. `transformers` is already a **hard**
dependency in `pyproject.toml` (used everywhere), so the lazy-import dance
adds no value.

```python
from transformers import (
    AutoFeatureExtractor, AutoModel, AutoProcessor, AutoTokenizer,
    AutoModelForSpeechSeq2Seq, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
model: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(model_source)
```

`transformers` ships `py.typed`; `from_pretrained` classmethods return
`Self`. Downstream `.to(device)` returns `Self`. That removes ~40 `Any`s and
the `model_any`/`processor_any` shadow variables.

**Confidence: HIGH** — verified `transformers>=4.x` has `py.typed` and the
`PreTrainedModel.from_pretrained` overloads; lazy import is genuinely
unnecessary because the import already happens unconditionally inside the
function.

---

### 3.2 `object` param annotations for HF models (5 sites)

**Location**: `data.py:482-483`, `features/ctc_gop.py:467-468`,
`features/ssl_utterance.py:121`.

**Current**:

```python
def encode_word_embeddings(..., tokenizer: object, model: object, ...) -> list[Tensor]:
    typed_tokenizer = cast(Any, tokenizer)
    typed_model = cast(Any, model)
```

**Proposed**:

- `tokenizer: PreTrainedTokenizerBase`
- `model: PreTrainedModel` (or the specific subclass, e.g. `Wav2Vec2ForCTC`)
- `processor: Wav2Vec2Processor`
- `dataset: Dataset` (from `datasets`)

This lets us delete the `cast(Any, ...)` on the very next line. For the
`encode_word_embeddings` call site, the tokenizer actually has the
`.word_ids(batch_index=...)` method which is on `BatchEncoding` — so the
return of `tokenizer(...)` is `transformers.BatchEncoding`, not `Any`.

**Confidence: HIGH** for class choices; **MEDIUM** for `Wav2Vec2Processor`
vs `ProcessorMixin` — the latter is the common base but the concrete class
offers tighter return types on `.batch_decode`.

---

### 3.3 `raw_any: Any = load_dataset(...)` (5 sites)

**Locations**: `data.py:320`, `features/ctc_gop.py:525,659`,
`features/ssl_utterance.py:187`, `freespeak/asr.py:148`.

**Current**: `raw_any: Any = load_dataset(dataset_id, split=split)`

**Proposed**: `raw: Dataset = load_dataset(dataset_id, split=split)` —
import `from datasets import Dataset`. `load_dataset` is overloaded and
returns `DatasetDict | Dataset | IterableDataset | IterableDatasetDict`;
with `split=` passed it returns `Dataset`. A runtime `isinstance` narrow
(or a targeted `cast`) is reasonable because the overloads in
`datasets>=2` don't always resolve.

**Confidence: HIGH** — this is exactly the documented behaviour.
`raw_any.cast_column` returns `Dataset`, confirming the runtime type.

---

### 3.4 `example = cast(dict[str, Any], raw_any[int(index)])` (6 sites)

**Locations**: `data.py:324,368`, `features/ctc_gop.py:690`,
`features/ssl_utterance.py:145,210`, `freespeak/asr.py:196`.

**Current**: `example = cast(dict[str, Any], raw_any[int(index)])`

**Proposed**: Introduce a `TypedDict` per dataset schema. For SpeechOcean762:

```python
class SpeechOceanWord(TypedDict):
    text: str
    phones: list[str]
    phones_accuracy: list[float]  # note: real key is "phones-accuracy"
    accuracy: float
    stress: float
    total: float

class SpeechOceanExample(TypedDict):
    text: str
    words: list[SpeechOceanWord]
    accuracy: float
    fluency: float
    completeness: float
    prosodic: float
    total: float
    audio: AudioPayload  # see 3.5
    speaker: NotRequired[str]
    id: NotRequired[str]
```

The `-` in `phones-accuracy` blocks a real TypedDict field — in practice
this means **one** `cast` is still required when reading that one key, but
everything else becomes strongly typed.

**Confidence: HIGH** for the schema; **MEDIUM** for TypedDict vs pydantic
BaseModel — TypedDict is cheaper but can't validate. Given the code already
calls `str(...)`, `float(...)` defensively, TypedDict is appropriate.

---

### 3.5 `_load_waveform(audio_payload: Any)` (3 sites, ruff-flagged)

**Locations**: `features/ctc_gop.py:436`, `features/ssl_utterance.py:94`,
`freespeak/asr.py:64` — identical body, all flagged `ANN401`.

**Current**:

```python
def _load_waveform(audio_payload: Any) -> Tensor:
    if hasattr(audio_payload, "get_all_samples"):
        samples = audio_payload.get_all_samples()
        waveform = cast(Tensor, samples.data).to(dtype=torch.float32)
```

**Proposed**: Union of two shapes plus a Protocol:

```python
class _DecodedAudio(TypedDict):
    array: NDArray[np.float32]
    sampling_rate: int
    path: NotRequired[str | None]

class _AudioDecoder(Protocol):
    def get_all_samples(self) -> _AudioSamples: ...

AudioPayload = _AudioDecoder | _DecodedAudio
```

The `get_all_samples` branch is `torchcodec.AudioDecoder` (HF datasets'
new audio backend); `.data` is a `Tensor` and `.sample_rate` is `int` —
verifiable from `datasets.features.Audio`. The duplicated function body
should also be hoisted into a single utility (see cleanup-document 03,
refactoring pass).

**Confidence: MEDIUM** — exact torchcodec class name depends on `datasets`
version. The `Protocol` form is safer than naming the concrete class.

---

### 3.6 `_TrackioHandle.module: Any`, `.log(metrics: dict[str, Any])`

**Location**: `training.py:61-64`.

**Current**:

```python
@dataclass
class _TrackioHandle:
    active: bool
    module: Any = None
    local_log: list[dict[str, Any]] = field(default_factory=list)
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
```

**Proposed**:

- `module: TrackioProtocol | None = None` using a `Protocol`:

  ```python
  class TrackioProtocol(Protocol):
      def init(self, **kwargs: object) -> object: ...
      def log(self, metrics: Mapping[str, float | int | str], step: int | None = None) -> None: ...
      def finish(self) -> None: ...
  ```

- `metrics: Mapping[str, float | int | str]` — every recorded value in the
  codebase (`history`, `epoch_record`, `step_record`) is a scalar numeric
  or string. No tensors, dicts, or lists are logged.

**Confidence: HIGH** for the metrics union (verified at every `.log()`
call site: `training.py:329-343,488-496`, `cli.py` emits only asdict'd
dataclass fields).

---

### 3.7 `load_or_build_word_embeddings` return type

**Location**: `data.py:425`.

**Current**: `-> dict[str, list[Tensor]]` with keys `"train"` and
`"test"` — callers index these by literal string.

**Proposed**:

```python
class WordEmbeddingsCache(TypedDict):
    train: list[Tensor]
    test: list[Tensor]
```

Or return a frozen dataclass. This removes a class of silent
`KeyError` risk.

**Confidence: HIGH**.

---

### 3.8 `torch.load(...)` return narrowing

**Locations**: `data.py:445,449,537,546`, `features/ctc_gop.py:352,802`.

**Current**: `cast(dict[str, Any], torch.load(...))` or `cast(dict[str, object], ...)`

**Proposed**: Replace with a `TypedDict` per cached file format, e.g.
`_GopCachePayload` (already a frozen dataclass!) mirrored as
`_GopCachePayloadDict` for the on-disk form:

```python
class _GopCachePayloadDict(TypedDict):
    dataset_indices: list[int]
    utterance_ids: list[str]
    features: list[Tensor]
    gop_dim: int
    model_id: str
```

This is also what would fix the two **real** `ty` errors at
`ctc_gop.py:354` and `:809`: replacing `dict[str, object]` with this
TypedDict makes `payload["gop_dim"]` resolve to `int`, and the `int(...)`
wrapper can be removed entirely.

**Confidence: HIGH** — fixes both current type-checker errors.

---

### 3.9 `whisper_kwargs: dict[str, Any]` (CLI/training plumbing)

**Location**: `data.py:197`.

**Current**: loose kwargs dict passed to `transcribe_split`.

**Proposed**: just call `transcribe_split(...)` with named arguments
directly — Python handles the `None` default for `whisper_model` via the
function's own signature. No dict needed.

**Confidence: HIGH** — pure refactor, no type surgery required.

---

### 3.10 `overrides: dict[str, Any]` in CLI arg plumbing

**Location**: `cli.py:83`.

**Current**: assembled as a `dict[str, Any]` then splatted into
`TrainingRunSettings(**overrides)`.

**Proposed**: `TrainingRunSettings` is a pydantic `BaseSettings`; construct
via `TrainingRunSettings.model_validate(overrides)` where `overrides: dict[str, object]`
— pydantic already coerces. But the values genuinely are heterogeneous
(Path, int, float, str, bool, list[int], None), so `dict[str, object]`
is the tightest honest type, not a strong typedict. **Leave as `dict[str, object]`** — pydantic validates downstream.

**Confidence: MEDIUM** — `object` is the honest answer here.

---

### 3.11 `g2p_en` module

**Location**: `freespeak/g2p.py:36,46`.

**Current**: `g2p_module = cast(Any, import_module("g2p_en"))`; return cast
`list[str]`.

**Proposed**: `g2p_en` has no type stubs, so introduce a local `.pyi`
shim under `src/p014/_stubs/g2p_en.pyi` (add to `ty.sources`), OR wrap
behind a small `Protocol`:

```python
class G2pProtocol(Protocol):
    def __call__(self, text: str) -> list[str]: ...
```

**Confidence: MEDIUM** — need to verify the actual call signature of
`g2p_en.G2p()`; the fallback `cast(list[str], ...)` for `phones_any` is
acceptable.

## 4. Legitimate `Any`s (do not touch)

1. **`_deep_merge(base: dict[str, Any], override: Mapping[str, Any])` — `config.py:155`.**
   Deep-merge over arbitrary YAML. The top-level pydantic validator
   narrows on the boundary. `dict[str, object]` would be tighter and
   equally honest; `Any` here is merely pragmatic. Low priority.
2. **`_read_yaml(path) -> dict[str, Any]` — `config.py:166`.** Same — YAML
   is genuinely heterogeneous; pydantic is the downstream validator.
3. **`tokenized: Any = typed_tokenizer(...)`, `processed: Any = ...`**
   — these are `BatchEncoding` / `BatchFeature` returns. Replaceable (see
   3.2) but the downstream `.to(device)`, `.input_values`, `.word_ids`
   attribute accesses are all typed on those classes.
4. **`g2p_en` untyped module** (3.11) — third-party without stubs.
5. **CLI `overrides: dict[str, Any]`** (3.10) — genuinely heterogeneous;
   pydantic validates.

## 5. `# type: ignore` audit

**Zero `# type: ignore` comments in the codebase.** There is nothing to
audit. This is unusually clean and should be preserved — every weak type
above is at least honest about being weak, rather than silenced. The two
live `ty` errors (`int(payload[...])` on `dict[str, object]` at
`ctc_gop.py:354,809`) are both fixable by adopting the TypedDict from
3.8, not by suppression.

---

### Implementation-effort ranking (highest-payoff first)

1. **3.1 + 3.2 + 3.3** — direct `transformers`/`datasets` imports replace
   ~55 Any/cast sites with one mechanical refactor.
2. **3.8** — fixes both live `ty` errors.
3. **3.4** — SpeechOcean762 TypedDict cleans up 6 per-example casts.
4. **3.6** — `_TrackioHandle` Protocol, small isolated change.
5. **3.5** — deduplicate `_load_waveform` + Protocol.
6. **3.7, 3.9, 3.10, 3.11** — minor cleanups.
