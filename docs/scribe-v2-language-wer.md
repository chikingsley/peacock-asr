# Scribe v2 Language WER Snapshot

Date recorded: 2026-06-29

Source: user-provided Scribe v2 language WER bucket list. Treat this as an external headline snapshot, not a Peacock benchmark run.

## Buckets

### Excellent: \<= 5% WER

Belarusian (`bel`), Bosnian (`bos`), Bulgarian (`bul`), Catalan (`cat`), Croatian (`hrv`), Czech (`ces`), Danish (`dan`), Dutch (`nld`), English (`eng`), Estonian (`est`), Finnish (`fin`), French (`fra`), Galician (`glg`), German (`deu`), Greek (`ell`), Hungarian (`hun`), Icelandic (`isl`), Indonesian (`ind`), Italian (`ita`), Japanese (`jpn`), Kannada (`kan`), Latvian (`lav`), Macedonian (`mkd`), Malay (`msa`), Malayalam (`mal`), Norwegian (`nor`), Polish (`pol`), Portuguese (`por`), Romanian (`ron`), Russian (`rus`), Slovak (`slk`), Spanish (`spa`), Swedish (`swe`), Turkish (`tur`), Ukrainian (`ukr`), and Vietnamese (`vie`).

### High Accuracy: > 5% to \<= 10% WER

Armenian (`hye`), Azerbaijani (`aze`), Bengali (`ben`), Cantonese (`yue`), Filipino (`fil`), Georgian (`kat`), Gujarati (`guj`), Hindi (`hin`), Kazakh (`kaz`), Lithuanian (`lit`), Maltese (`mlt`), Mandarin (`cmn`), Marathi (`mar`), Nepali (`nep`), Odia (`ori`), Persian (`fas`), Serbian (`srp`), Slovenian (`slv`), Swahili (`swa`), Tamil (`tam`), and Telugu (`tel`).

### Good: > 10% to \<= 20% WER

Afrikaans (`afr`), Arabic (`ara`), Assamese (`asm`), Asturian (`ast`), Burmese (`mya`), Hausa (`hau`), Hebrew (`heb`), Javanese (`jav`), Korean (`kor`), Kyrgyz (`kir`), Luxembourgish (`ltz`), Maori (`mri`), Occitan (`oci`), Punjabi (`pan`), Tajik (`tgk`), Thai (`tha`), Uzbek (`uzb`), and Welsh (`cym`).

### Moderate: > 25% to \<= 50% WER

Amharic (`amh`), Ganda (`lug`), Igbo (`ibo`), Irish (`gle`), Khmer (`khm`), Kurdish (`kur`), Lao (`lao`), Mongolian (`mon`), Northern Sotho (`nso`), Pashto (`pus`), Shona (`sna`), Sindhi (`snd`), Somali (`som`), Urdu (`urd`), Wolof (`wol`), Xhosa (`xho`), Yoruba (`yor`), and Zulu (`zul`).

Note: the supplied list has no bucket for > 20% to \<= 25% WER.

## Peacock Comparison

This broadly lines up for Tajik and only partially lines up for Persian.

- **Persian (`fas`)** is listed as High Accuracy, > 5% to \<= 10% WER. Our clean/read Persian evals can hit that band: the current Omni Scribe-v4 run recorded FLEURS at 8.69% WER and Neyshekar at 8.49% WER. Broader or noisier Peacock splits are higher: Common Voice 19.37%, YouTube 20.34%, WorldSpeech 27.45%, and six-split greedy macro WER around 20%. So the Scribe v2 headline is plausible for clean Persian, but it should not be read as a promise for noisy conversational Iranian Persian.
- **Tajik (`tgk`)** is listed as Good, > 10% to \<= 20% WER. That matches our early Scribe-v2 baseline well: corpus-level split WER was train 11.7%, dev 13.1%, and test 14.7%, with macro WER 16.74%. It also matches the read-speech model/eval band: Tajik v2 FLEURS test WER was 17.17%. It does **not** describe the harder conversational YouTube condition, where our honest held-out benchmark for the shipping v3 model was 37.65% WER before KenLM and 31.66% after KenLM.

Operational read: these buckets are useful for deciding whether Scribe v2 is strong enough to bootstrap pseudo-labeling. For Persian and Tajik, the answer is yes, but Peacock still needs domain-specific gates: language/script checks, descriptor-junk filters, WER/self-consistency scoring, and held-out conversational tests.
