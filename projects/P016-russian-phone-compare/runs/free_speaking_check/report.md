# P016 Free-Speaking Eval Report

Scoring path: audio -> Qwen ASR -> ASR text -> lane-specific G2P -> ZIPA/XLSR phones -> PER/PFER.

Known dataset text is used only for audit/reporting. It is not fed to the scorer.

## Lane Summary

| language | lane | n | avg PER | avg PFER |
| --- | --- | ---: | ---: | ---: |
| en_us | xlsr-espeak | 2 | 0.1695 | 0.0909 |
| en_us | zipa | 2 | 0.1139 | 0.0358 |
| ru | xlsr-espeak | 2 | 0.4759 | 0.2199 |
| ru | zipa | 2 | 0.3083 | 0.1150 |

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
- `fleurs_ru_1634`
  - zipa: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none

## Worst 20 Word Rows

| sample | language | lane | word | PER | PFER | target | heard | details |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| fleurs_en_us_1510 | en_us | xlsr-espeak | the | 1.5000 | 1.5000 | `ð ə` | `ð ə j u w` | insertions: `j, u, w` |
| fleurs_ru_1634 | ru | xlsr-espeak | в | 1.0000 | 1.0000 | `v` | `` | deletions: `v` |
| fleurs_ru_1634 | ru | zipa | в | 1.0000 | 0.6458 | `vʲ e` | `v` | substitutions: `e->v`<br>deletions: `vʲ` |
| fleurs_en_us_1548 | en_us | xlsr-espeak | are | 1.0000 | 0.6250 | `ɑ ɹ` | `ə˞` | substitutions: `ɹ->ə˞`<br>deletions: `ɑ` |
| fleurs_ru_1554 | ru | xlsr-espeak | wi-fi | 0.6429 | 0.5923 | `( e n ) w a ɪ f a ɪ ( r u )` | `b a ɪ f a ɪ` | substitutions: `w->b`<br>deletions: `(, e, n, ), (, r, u, )` |
| fleurs_en_us_1510 | en_us | xlsr-espeak | a | 1.0000 | 0.5417 | `e ɪ` | `ɐ` | substitutions: `ɪ->ɐ`<br>deletions: `e` |
| fleurs_ru_1634 | ru | zipa | из | 1.0000 | 0.5208 | `i s` | `z` | substitutions: `s->z`<br>deletions: `i` |
| fleurs_en_us_1510 | en_us | zipa | the | 0.5000 | 0.5000 | `ð ə` | `ð ə w` | insertions: `w` |
| fleurs_ru_1634 | ru | xlsr-espeak | из | 0.5000 | 0.5000 | `ɪ s` | `oɪ s` | substitutions: `ɪ->oɪ` |
| fleurs_ru_1554 | ru | zipa | wi-fi | 1.0000 | 0.3958 | `w i f i` | `ɑ j f ɑ j` | substitutions: `w->ɑ, i->j, i->j`<br>insertions: `ɑ` |
| fleurs_ru_1634 | ru | xlsr-espeak | тысяч | 0.8333 | 0.3750 | `t y sʲ ʌ t ʃʲ` | `t i s i ʃ` | substitutions: `sʲ->i, ʌ->s, t->i, ʃʲ->ʃ`<br>deletions: `y` |
| fleurs_ru_1634 | ru | xlsr-espeak | большой | 0.6667 | 0.3681 | `b ʌ ɭ ʃ o j` | `b a l ʃ` | substitutions: `ʌ->a, ɭ->l`<br>deletions: `o, j` |
| fleurs_ru_1634 | ru | zipa | в | 1.0000 | 0.3542 | `vʲ e` | `o v` | substitutions: `vʲ->o, e->v` |
| fleurs_en_us_1548 | en_us | xlsr-espeak | call | 0.6667 | 0.3472 | `k ɔ l` | `k a ʊ l` | substitutions: `ɔ->ʊ`<br>insertions: `a` |
| fleurs_ru_1634 | ru | xlsr-espeak | хонсю | 0.5000 | 0.3403 | `x ʌ n sʲ u "` | `x o n ts u` | substitutions: `ʌ->o, sʲ->ts`<br>deletions: `"` |
| fleurs_en_us_1548 | en_us | zipa | when | 0.3333 | 0.3333 | `w ɛ n` | `h w ɛ n` | insertions: `h` |
| fleurs_ru_1634 | ru | xlsr-espeak | что | 0.3333 | 0.3333 | `ʃ t o` | `ʃ t` | deletions: `o` |
| fleurs_ru_1634 | ru | xlsr-espeak | делает | 1.0000 | 0.3155 | `dʲ e ɭ ʌ j i t` | `a d j e l ð` | substitutions: `e->a, ɭ->d, ʌ->j, j->e, i->l, t->ð`<br>deletions: `dʲ` |
| fleurs_ru_1634 | ru | zipa | приблизительно | 0.4615 | 0.3141 | `p rʲ ɪ b lʲ ɪ zʲ i tʲ ɪ lʲ n ə` | `p r ɪ b ɪ lʲ ɪ zʲ i tʲ ɪ l n ə sʲ e m` | substitutions: `rʲ->r, lʲ->l`<br>insertions: `ɪ, sʲ, e, m` |
| fleurs_ru_1554 | ru | zipa | от | 1.0000 | 0.3125 | `o t` | `t b` | substitutions: `o->t, t->b` |

## Read

- The end-to-end free-speaking path runs on this sample.
- English is bounded but still has false-positive surface area.
- Russian remains noisy enough that the current score should be treated as diagnostic, not learner feedback.
- en_us/xlsr-espeak: avg PER 0.1695, avg PFER 0.0909.
- en_us/zipa: avg PER 0.1139, avg PFER 0.0358.
- ru/xlsr-espeak: avg PER 0.4759, avg PFER 0.2199.
- ru/zipa: avg PER 0.3083, avg PFER 0.1150.
- High-error rows are concentrated in short function words, abbreviations/numbers, and Russian target/recognizer inventory mismatches.
