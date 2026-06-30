# Russian (rus_Cyrl) ASR datasets — availability survey

Researched 2026-06-15. Audio + Russian transcripts for fine-tuning. Russian is high-resource —
this is the acquisition map. Canonical living index to re-check for new releases:
<https://github.com/alphacep/awesome-russian-speech> (Vosk team).

## Already in hand (local, under `data` → `/mnt/workerssd-2t/peacock-asr/russian-asr`)

- **ru_open_stt (OpenSTT)** — the ~20,000 h giant. **License: CC-BY-NC** (non-commercial — poisons a
  commercial release unless you negotiate with snakers4). Heterogeneous; filter hard by alignment.
- **SOVA** — umbrella, **CC-BY-4.0** (commercial OK, attribute Virtual Assistant LLC). Confirm parts
  (RuDevices 101 h manual / RuYouTube 17,451 h auto / RuAudiobooks 298 h manual). Prioritize **manual**.
- ~~TIMIT-Russian~~ — **REMOVED. It is the original ENGLISH LDC TIMIT mislabeled as Russian** (verified
  from its README/PROMPTS/SPKRINFO): wrong language AND LDC-restricted (no redistribute / no commercial
  / no publishing models on it). Quarantined; do not use. Plus FLEURS `ru_ru` + Common Voice `ru`.

## Licenses (verified 2026-06-15) — commercial-clean vs NC

- **Commercial-safe + publishable:** SOVA (CC-BY-4.0), RuLS (Public Domain), ToneBooks (Apache-2.0),
  M-AILABS ru (BSD-3), Golos (CC-BY-SA — ShareAlike attaches to a *published dataset export*, a trained
  model is lower-risk; attribute Sber), FLEURS (CC-BY-4.0), Common Voice ru (CC0).
- **Non-commercial (keep OUT of a commercial mix):** ru_open_stt (CC-BY-NC), RUSLAN (CC-BY-NC-SA),
  ESpeech-podcasts (CC-BY-NC).
- **Unusable:** the local TIMIT (LDC English, restricted).

## Tier 1 — get these first (big, transcribed, permissive)

| Dataset | Where | Hours | Domain | License | Verdict |
|---|---|---|---|---|---|
| **Golos** (Sber) | OpenSLR SLR114 · `bond005/sberdevices_golos_*` | ~1,240 | crowd read + farfield | permissive (commercial OK) | **The best clean open Russian corpus. Get it.** |
| **ESpeech-podcasts** | HF `ESpeech/ESpeech-podcasts` | **3,200** | podcasts (spontaneous) | CC-BY-NC-4.0 | Huge spontaneous; verify transcript quality (likely pseudo-label). NC only. |
| **ESpeech-webinars2** | HF `ESpeech/ESpeech-webinars2` | ~800 | webinars/lectures | MIT (confirm on card) | Big longform, permissive if MIT. |

## Tier 2 — solid, fully transcribed

| Dataset | Where | Hours | Domain | License | Note |
|---|---|---|---|---|---|
| **RuLS** (Russian LibriSpeech) | OpenSLR SLR96 · `bond005/rulibrispeech` | ~98 | audiobooks (clean read) | Public Domain | Easy open win. (SLR96 ≠ OpenSTT — different corpus.) |
| **SOVA RuDevices** | `bond005/sova_rudevices` | ~101 | live mic, **manual** | free | Convenient parquet form of the manual SOVA slice. |
| **Common Voice ru** | Mozilla Data Collective (MDC-only since Oct 2025) | ~252 validated + new **Spontaneous** track | read + spontaneous | CC0 | Own pipeline (user's CV project). It grew — refresh + add the spontaneous subset. |
| **Podlodka Speech** | `bond005/podlodka_speech` | tiny | IT podcast | — | Too small to train — use as the standard Russian podcast **eval benchmark**. |

## Tier 3 — TTS corpora as clean read-ASR pairs

RUSLAN (~31 h, 1 spk, ruslan-corpus.github.io) · M-AILABS Russian (~47 h, BSD-ish, commercial OK) ·
ToneBooks (`Vikhrmodels/ToneBooks`, ~179 h audiobook). Good clean-read augmentation.

## Tier 4 — massive but weak/pseudo-labeled (pretraining / pseudo-label mining, not clean FT)

SOVA RuYouTube (~17,451 h auto) · **YODAS ru** (`espnet/yodas2`; prefer the cleaned **`espnet/yodas-granary`** / `yodas_owsmv4` variants) · Sinoosoida/SpeechRu (7.38 TB, **no transcripts** — SSL only).

## Paid / LDC (flagged)

CALLFRIEND Russian (LDC2023S08/T09, ~48 h telephone — the only notable RU telephone set, paywalled) ·
Russian Call Center Speech (~832 h, vendor-gated) · UniData/AxonData (samples free, full paid).

## Negative findings (save time)

- **MLS has NO Russian** (8 EU langs only). **NVIDIA Granary has NO Russian** (use `yodas-granary` instead).
- No separate CALLHOME-Russian; CALLFRIEND is the LDC equivalent.

## Recommended order

1. **Golos** (SLR114) → 2. **ESpeech podcasts+webinars** (if NC/MIT fit) → 3. **RuLS + M-AILABS + RUSLAN**
(clean read) → 4. **Common Voice ru** refresh → 5. **SOVA RuDevices** (manual) → 6. YODAS-granary +
filtered OpenSTT (noisy pools) → 7. **Podlodka** as eval benchmark.
