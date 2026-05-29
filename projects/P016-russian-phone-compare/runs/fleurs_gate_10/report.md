# P016 Free-Speaking Eval Report

Scoring path: audio -> Qwen ASR -> ASR text -> lane-specific G2P -> ZIPA/XLSR phones -> PER/PFER.

Known dataset text is used only for audit/reporting. It is not fed to the scorer.

## Lane Summary

| language | lane | n | avg PER | avg PFER |
| --- | --- | ---: | ---: | ---: |
| en_us | xlsr-espeak | 10 | 0.1718 | 0.0776 |
| en_us | zipa | 10 | 0.1428 | 0.0496 |
| ru | xlsr-espeak | 10 | 0.4593 | 0.1397 |
| ru | xlsr-mfa | 10 | 0.5663 | 0.1402 |
| ru | zipa | 10 | 0.2080 | 0.0549 |
| ru | zipa-charsiu | 10 | 0.2508 | 0.0786 |

## ASR Vs Known Text

- `fleurs_en_us_1548` (en_us): match
  - known: `when you call someone who is thousands of miles away you are using a satellite`
  - asr: `When you call someone who is thousands of miles away, you are using a satellite.`
- `fleurs_en_us_1620` (en_us): changed
  - known: `now widely available throughout the archipelago javanese cuisine features an array of simply seasoned dishes the predominant flavorings the javanese favor being peanuts chillies sugar especially javanese coconut sugar and various aromatic spices`
  - asr: `Now widely available throughout the archipelago, Japanese cuisine features an array of simply seasoned dishes. The predominant flavorings the Japanese favor being peanuts, chilies, sugar, especially Japanese coconut sugar, and various aromatic spices.`
- `fleurs_en_us_1510` (en_us): changed
  - known: `the u.n. also hopes to finalize a fund to help countries affected by global warming to cope with the impacts`
  - asr: `The UN also hopes to finalize a fund to help countries affected by global warming to cope with the impacts.`
- `fleurs_en_us_1578` (en_us): changed
  - known: `then lakkha singh took the lead in singing the bhajans`
  - asr: `Then Lakasang took the lead in singing the pasongs.`
- `fleurs_en_us_1652` (en_us): match
  - known: `the major religion in moldova is orthodox christian`
  - asr: `The major religion in Moldova is Orthodox Christian.`
- `fleurs_en_us_1531` (en_us): match
  - known: `the east african islands are in the indian ocean off the eastern coast of africa`
  - asr: `The East African Islands are in the Indian Ocean off the eastern coast of Africa.`
- `fleurs_en_us_1528` (en_us): match
  - known: `some festivals have special camping areas for families with young children`
  - asr: `Some festivals have special camping areas for families with young children.`
- `fleurs_en_us_1595` (en_us): changed
  - known: `the tibetan buddhism is based on the teachings of buddha but were extended by the mahayana path of love and by a lot of techniques from indian yoga`
  - asr: `The Tibetan Buddhism is based on the teachings of Buddha, but were extended by the Mayana Path of Love and by a lot of techniques from Indian Yoga.`
- `fleurs_en_us_1559` (en_us): match
  - known: `pronunciation is relatively easy in italian since most words are pronounced exactly how they are written`
  - asr: `Pronunciation is relatively easy in Italian since most words are pronounced exactly how they are written.`
- `fleurs_en_us_1633` (en_us): changed
  - known: `in this dynamic transport shuttle everyone is somehow connected with and supporting a transport system based on private cars`
  - asr: `In this dynamic transport shuttle, everyone is connected with and supporting a transport system based on private cars.`
- `fleurs_ru_1614` (ru): changed
  - known: `они умеют отлично видеть в темноте при помощи ночного видения и почти незаметно передвигаться оцелоты выслеживают добычу сливаясь с окружающей обстановкой а затем набрасываются на добычу`
  - asr: `Они умеют отлично видеть в темноте при помощи ночного видения и почти незаметно передвигаться. Оцилоты выслеживают добычу, сливаясь с окружающей обстановкой, а затем набрасываются на добычу.`
- `fleurs_ru_1554` (ru): changed
  - known: `он сказал что создал дверной звонок работающий от wifi`
  - asr: `Он сказал, что создал дверной звонок, работающий от Wi-Fi.`
- `fleurs_ru_1634` (ru): changed
  - known: `в японии приблизительно 7000 островов самый большой из которых — хонсю что делает японию 7-м по величине островом в мире!`
  - asr: `В Японии приблизительно 7 тысяч островов, самый большой из которых Хонсю, что делает Японию седьмым по величине островом в мире.`
- `fleurs_ru_1615` (ru): changed
  - known: `тысячи лет назад человек по имени аристарх сказал что солнечная система вращается вокруг солнца`
  - asr: `Тысячи лет назад человек по имени Ристарх сказал, что Солнечная система вращается вокруг Солнца.`
- `fleurs_ru_1609` (ru): changed
  - known: `робин утхаппа набрал рекордное количество очков в иннинге 70 ранов за всего 41 подачу отбил 11 четвёрок и 2 шестёрки`
  - asr: `Робин Утхаппа набрал рекордное количество очков в вынинге — 70 ранов за всего 41 подачу, отбил 11 четверок и две шестерки.`
- `fleurs_ru_1549` (ru): changed
  - known: `эти теории предполагают что у людей есть определённые потребности и/или желания которые накапливаются внутри в процессе взросления`
  - asr: `Эти теории предполагают, что у людей есть определенная потребности и/или желания, которые накапливаются внутри в процессе взросления.`
- `fleurs_ru_1587` (ru): match
  - known: `о первых случаях заболевания в этом сезоне было сообщено в июле`
  - asr: `О первых случаях заболевания в этом сезоне было сообщено в июле.`
- `fleurs_ru_1618` (ru): changed
  - known: `в северной части то есть на хребте сентинел находятся самые высокие горы антарктиды массив винсон самая высшая точка которого достигает 4892 м и называется пик винсон`
  - asr: `В северной части, то есть на хребте Сантинел, находятся самые высокие горы Антарктиды – массив Винсон. Самая высшая точка которого достигает 4892 метра и называется пик Винсон.`
- `fleurs_ru_1651` (ru): changed
  - known: `вариант становящийся всё более популярным для тех кто планирует взять академический год это путешествовать и учиться`
  - asr: `Вариант, становящийся все более популярным для тех, кто планирует взять академический год, это путешествовать и учиться.`
- `fleurs_ru_1647` (ru): match
  - known: `изложенные мнения часто поверхностны расплывчаты и чрезмерно упрощены по сравнению с повсеместно доступной более подробной информацией`
  - asr: `Изложенные мнения часто поверхностны, расплывчаты и чрезмерно упрощены по сравнению с повсеместно доступной более подробной информацией.`

## Target Backends

- `fleurs_en_us_1548`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1620`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1510`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1578`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1652`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1531`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1528`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1595`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1559`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_en_us_1633`
  - zipa: `espeak-ng:en-us`, warnings: none
  - xlsr-espeak: `espeak-ng:en-us`, warnings: none
- `fleurs_ru_1614`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1554`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1634`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1615`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1609`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1549`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1587`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1618`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1651`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none
- `fleurs_ru_1647`
  - zipa: `mfa:russian_mfa`, warnings: none
  - xlsr-espeak: `espeak-ng:ru`, warnings: none
  - zipa-charsiu: `charsiu:charsiu/g2p_multilingual_byT5_tiny_16_layers_100`, warnings: none
  - xlsr-mfa: `mfa:russian_mfa`, warnings: none

## Worst 20 Word Rows

| sample | language | lane | word | PER | PFER | target | heard | details |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| fleurs_en_us_1510 | en_us | xlsr-espeak | the | 1.5000 | 1.5000 | `ð ə` | `ð ə j u w` | insertions: `j, u, w` |
| fleurs_en_us_1633 | en_us | zipa | is | 1.5000 | 1.5000 | `ɪ z` | `ɪ z ʃ o ʊ` | insertions: `ʃ, o, ʊ` |
| fleurs_en_us_1633 | en_us | xlsr-espeak | is | 1.5000 | 1.2292 | `ɪ z` | `ɪ ʃ o ʊ` | substitutions: `z->ʊ`<br>insertions: `ʃ, o` |
| fleurs_ru_1587 | ru | xlsr-espeak | о | 2.0000 | 1.1667 | `o` | `a ɪ` | substitutions: `o->ɪ`<br>insertions: `a` |
| fleurs_ru_1587 | ru | xlsr-mfa | о | 2.0000 | 1.1667 | `o` | `a ɪ` | substitutions: `o->ɪ`<br>insertions: `a` |
| fleurs_ru_1614 | ru | xlsr-espeak | в | 1.0000 | 1.0000 | `f` | `` | deletions: `f` |
| fleurs_ru_1634 | ru | xlsr-espeak | в | 1.0000 | 1.0000 | `v` | `` | deletions: `v` |
| fleurs_ru_1634 | ru | xlsr-mfa | в | 1.0000 | 1.0000 | `f` | `` | deletions: `f` |
| fleurs_ru_1549 | ru | xlsr-mfa | что | 1.0000 | 0.8194 | `ʂ t o` | `o g` | substitutions: `t->o, o->g`<br>deletions: `ʂ` |
| fleurs_ru_1615 | ru | zipa-charsiu | тысячи | 0.7500 | 0.7500 | `t ɨ ɕ ɪ` | `t ɨ sʲ ɪ t ɕ ɪ` | insertions: `sʲ, ɪ, t` |
| fleurs_ru_1614 | ru | zipa-charsiu | в | 1.0000 | 0.6875 | `vʲ e` | `t` | substitutions: `e->t`<br>deletions: `vʲ` |
| fleurs_ru_1634 | ru | zipa-charsiu | в | 1.0000 | 0.6458 | `vʲ e` | `v` | substitutions: `e->v`<br>deletions: `vʲ` |
| fleurs_ru_1609 | ru | zipa-charsiu | в | 1.0000 | 0.6458 | `vʲ e` | `v` | substitutions: `e->v`<br>deletions: `vʲ` |
| fleurs_ru_1549 | ru | zipa-charsiu | в | 1.0000 | 0.6458 | `vʲ e` | `v` | substitutions: `e->v`<br>deletions: `vʲ` |
| fleurs_ru_1587 | ru | zipa-charsiu | в | 1.0000 | 0.6458 | `vʲ e` | `v` | substitutions: `e->v`<br>deletions: `vʲ` |
| fleurs_ru_1618 | ru | zipa-charsiu | в | 1.0000 | 0.6458 | `vʲ e` | `v` | substitutions: `e->v`<br>deletions: `vʲ` |
| fleurs_en_us_1548 | en_us | xlsr-espeak | are | 1.0000 | 0.6250 | `ɑ ɹ` | `ə˞` | substitutions: `ɹ->ə˞`<br>deletions: `ɑ` |
| fleurs_ru_1618 | ru | xlsr-mfa | есть | 1.0000 | 0.6250 | `j e sʲ tʲ` | `ɪ s` | substitutions: `sʲ->ɪ, tʲ->s`<br>deletions: `j, e` |
| fleurs_en_us_1510 | en_us | xlsr-espeak | a | 1.0000 | 0.5417 | `e ɪ` | `ɐ` | substitutions: `ɪ->ɐ`<br>deletions: `e` |
| fleurs_en_us_1578 | en_us | zipa | in | 1.0000 | 0.5417 | `ɪ n` | `ə n d` | substitutions: `ɪ->ə`<br>insertions: `d` |

## Read

- The end-to-end free-speaking path runs on this sample.
- English is bounded but still has false-positive surface area.
- Russian remains noisy enough that the current score should be treated as diagnostic, not learner feedback.
- en_us/xlsr-espeak: avg PER 0.1718, avg PFER 0.0776.
- en_us/zipa: avg PER 0.1428, avg PFER 0.0496.
- ru/xlsr-espeak: avg PER 0.4593, avg PFER 0.1397.
- ru/xlsr-mfa: avg PER 0.5663, avg PFER 0.1402.
- ru/zipa: avg PER 0.2080, avg PFER 0.0549.
- ru/zipa-charsiu: avg PER 0.2508, avg PFER 0.0786.
- High-error rows are concentrated in short function words, abbreviations/numbers, and Russian target/recognizer inventory mismatches.
