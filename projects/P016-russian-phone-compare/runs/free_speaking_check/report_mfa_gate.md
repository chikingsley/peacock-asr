# P016 Free-Speaking Eval Report

Scoring path: audio -> Qwen ASR -> ASR text -> lane-specific G2P -> ZIPA/XLSR phones -> PER/PFER.

Known dataset text is used only for audit/reporting. It is not fed to the scorer.

## Lane Summary

| language | lane | n | avg PER | avg PFER |
| --- | --- | ---: | ---: | ---: |
| en_us | xlsr-espeak | 2 | 0.1695 | 0.0909 |
| en_us | zipa | 2 | 0.1139 | 0.0358 |
| ru | xlsr-espeak | 2 | 0.4733 | 0.1573 |
| ru | xlsr-mfa | 2 | 0.5251 | 0.1565 |
| ru | zipa | 2 | 0.2693 | 0.0864 |
| ru | zipa-mfa | 2 | 0.2158 | 0.0582 |

## ASR Vs Known Text

- `fleurs_en_us_1548` (en_us): match
  - known: `when you call someone who is thousands of miles away you are using a satellite`
  - asr: `When you call someone who is thousands of miles away, you are using a satellite.`
- `fleurs_en_us_1510` (en_us): changed
  - known: `the u.n. also hopes to finalize a fund to help countries affected by global warming to cope with the impacts`
  - asr: `The UN also hopes to finalize a fund to help countries affected by global warming to cope with the impacts.`
- `fleurs_ru_1554` (ru): changed
  - known: `он сказал что создал дверной звонок работающий от wifi`
  - asr: `Он сказал, что создал дверной звонок, работающий от Wi-Fi.`
- `fleurs_ru_1634` (ru): changed
  - known: `в японии приблизительно 7000 островов самый большой из которых — хонсю что делает японию 7-м по величине островом в мире!`
  - asr: `В Японии приблизительно 7 тысяч островов, самый большой из которых Хонсю, что делает Японию седьмым по величине островом в мире.`

## Target Backends

- `fleurs_en_us_1548`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1510`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_ru_1554`
  - zipa: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-mfa: `mfa:russian_mfa`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1634`
  - zipa: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-mfa: `mfa:russian_mfa`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none

## Worst 20 Word Rows

| sample | language | lane | word | PER | PFER | target | heard | details |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| fleurs_en_us_1510 | en_us | xlsr-espeak | the | 1.5000 | 1.5000 | `ð ə` | `ð ə j u w` | insertions: `j, u, w` |
| fleurs_ru_1634 | ru | xlsr-espeak | в | 1.0000 | 1.0000 | `v` | `` | deletions: `v` |
| fleurs_ru_1634 | ru | xlsr-mfa | в | 1.0000 | 1.0000 | `f` | `` | deletions: `f` |
| fleurs_ru_1634 | ru | zipa | в | 1.0000 | 0.6458 | `vʲ e` | `v` | substitutions: `e->v`<br>deletions: `vʲ` |
| fleurs_en_us_1548 | en_us | xlsr-espeak | are | 1.0000 | 0.6250 | `ɑ ɹ` | `ə˞` | substitutions: `ɹ->ə˞`<br>deletions: `ɑ` |
| fleurs_en_us_1510 | en_us | xlsr-espeak | a | 1.0000 | 0.5417 | `e ɪ` | `ɐ` | substitutions: `ɪ->ɐ`<br>deletions: `e` |
| fleurs_ru_1634 | ru | zipa | из | 1.0000 | 0.5208 | `i s` | `z` | substitutions: `s->z`<br>deletions: `i` |
| fleurs_ru_1634 | ru | xlsr-mfa | из | 1.0000 | 0.5208 | `ɪ z` | `oɪ s` | substitutions: `ɪ->oɪ, z->s` |
| fleurs_ru_1634 | ru | xlsr-mfa | большой | 0.8333 | 0.5208 | `b ɐ lʲ ʂ o j` | `b a l ʃ` | substitutions: `ʂ->a, o->l, j->ʃ`<br>deletions: `ɐ, lʲ` |
| fleurs_en_us_1510 | en_us | zipa | the | 0.5000 | 0.5000 | `ð ə` | `ð ə w` | insertions: `w` |
| fleurs_ru_1634 | ru | xlsr-espeak | из | 0.5000 | 0.5000 | `ɪ s` | `oɪ s` | substitutions: `ɪ->oɪ` |
| fleurs_ru_1634 | ru | zipa-mfa | из | 0.5000 | 0.5000 | `ɪ z` | `z` | deletions: `ɪ` |
| fleurs_ru_1634 | ru | xlsr-mfa | что | 1.0000 | 0.4861 | `ʂ t o` | `ʃ t a d` | substitutions: `ʂ->ʃ, o->d`<br>insertions: `a` |
| fleurs_ru_1634 | ru | xlsr-mfa | делает | 0.7143 | 0.4821 | `dʲ e l ə j ɪ t` | `j e l ð` | substitutions: `dʲ->j, t->ð`<br>deletions: `ə, j, ɪ` |
| fleurs_ru_1634 | ru | xlsr-mfa | тысяч | 0.8333 | 0.3819 | `t ɨ sʲ ɪ t ɕ` | `t i s i ʃ` | substitutions: `sʲ->i, ɪ->s, t->i, ɕ->ʃ`<br>deletions: `ɨ` |
| fleurs_ru_1634 | ru | xlsr-espeak | тысяч | 0.8333 | 0.3750 | `t y sʲ ʌ t ʃʲ` | `t i s i ʃ` | substitutions: `sʲ->i, ʌ->s, t->i, ʃʲ->ʃ`<br>deletions: `y` |
| fleurs_ru_1554 | ru | xlsr-mfa | работающий | 0.6364 | 0.3712 | `r ɐ b o t ə j ʉ ɕː ɪ j` | `r a b o t i ʃ i` | substitutions: `ɐ->a, ɕː->i, ɪ->ʃ, j->i`<br>deletions: `ə, j, ʉ` |
| fleurs_ru_1634 | ru | xlsr-espeak | большой | 0.6667 | 0.3681 | `b ʌ ɭ ʃ o j` | `b a l ʃ` | substitutions: `ʌ->a, ɭ->l`<br>deletions: `o, j` |
| fleurs_ru_1634 | ru | zipa | в | 1.0000 | 0.3542 | `vʲ e` | `o v` | substitutions: `vʲ->o, e->v` |
| fleurs_en_us_1548 | en_us | xlsr-espeak | call | 0.6667 | 0.3472 | `k ɔ l` | `k a ʊ l` | substitutions: `ɔ->ʊ`<br>insertions: `a` |

## Read

- The end-to-end free-speaking path runs on this sample.
- English is bounded but still has false-positive surface area.
- Russian remains noisy enough that the current score should be treated as diagnostic, not learner feedback.
- en_us/xlsr-espeak: avg PER 0.1695, avg PFER 0.0909.
- en_us/zipa: avg PER 0.1139, avg PFER 0.0358.
- ru/xlsr-espeak: avg PER 0.4733, avg PFER 0.1573.
- ru/xlsr-mfa: avg PER 0.5251, avg PFER 0.1565.
- ru/zipa: avg PER 0.2693, avg PFER 0.0864.
- ru/zipa-mfa: avg PER 0.2158, avg PFER 0.0582.
- High-error rows are concentrated in short function words, abbreviations/numbers, and Russian target/recognizer inventory mismatches.
