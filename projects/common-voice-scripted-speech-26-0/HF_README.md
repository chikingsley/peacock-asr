---
license: cc0-1.0
language:
  - abq
  - bsh
  - dru
  - fa
  - gjk
  - gya
  - hno
  - hux
  - jgo
  - kab
  - kbd
  - kjh
  - kln
  - kls
  - km
  - kvx
  - lo
  - mau
  - nla
  - nlv
  - nnh
  - nso
  - nyu
  - oc
  - odk
  - om
  - or
  - oru
  - os
  - pcm
  - phl
  - phr
  - pl
  - ps
  - pt
  - pwn
  - qur
  - qus
  - qux
  - quy
  - qva
  - qvl
  - qwa
  - qws
  - qxa
  - qxp
  - qxt
  - qxu
  - qxw
  - ro
  - ru
  - udl
  - xka
  - yav
language_bcp47:
  - nb-NO
  - ne-NP
  - nn-NO
  - pa-IN
  - rm-sursilv
  - rm-vallader
task_categories:
  - automatic-speech-recognition
tags:
  - common-voice
  - mozilla-data-collective
  - scripted-speech
  - multilingual-asr
  - asr
pretty_name: Common Voice Scripted Speech
---

# Common Voice Scripted Speech

A row-normalized multilingual ASR dataset built from Mozilla Data Collective Common Voice Scripted Speech. Each upstream archive is converted to appendable parquet shards under `data/<upstream_split>/`, one shard per source archive and split, with audio bytes embedded in an `audio` struct column.

## Status

- Manifest languages: 60
- Languages uploaded: 31

## Columns

- `audio` (`bytes`, `path`)
- `sentence`, `locale`, `language`, `upstream_split`
- `source_dataset_id`, `source_archive`, `collection`
- upstream Common Voice metadata: `client_id`, `sentence_id`, `sentence_domain`, `up_votes`, `down_votes`, `age`, `gender`, `accents`, `variant`, `segment`
- `duration_ms` from `clip_durations.tsv` when available
- `license`, `license_url`

## License

Mozilla Data Collective records these archives as Creative Commons Zero v1.0 Universal (`CC0-1.0`). See `LICENSE` and <https://spdx.org/licenses/CC0-1.0.html>.

## Languages

| Language                        | Locale        | MDC dataset ID              | Status   |
| ------------------------------- | ------------- | --------------------------- | -------- |
| Abaza                           | `abq`         | `cmqilxwjm00r8nr07acb9f5i9` | complete |
| Afaan Oromo                     | `om`          | `cmqiodaxn00ycnq07slso6and` | complete |
| Huautla Mazatec                 | `mau`         | `cmqi6gj5v00dwmf07mu79ql0u` | complete |
| Kabardian                       | `kbd`         | `cmqinjn4o00w6nq07pyqsmwg7` | complete |
| Kabyle                          | `kab`         | `cmqim4fux00tynq07ljtyhzfh` | pending  |
| Kachhi                          | `gjk`         | `cmqi97zao002anr07098afaeh` | complete |
| Kalasha                         | `kls`         | `cmqi6frkb00eyo507j3c6xegh` | complete |
| Kalenjin                        | `kln`         | `cmqinqp7r00x2nq071z5e06oa` | pending  |
| Kalkoti                         | `xka`         | `cmqi6hp9d00egmf07zv75th14` | complete |
| Kateviri                        | `bsh`         | `cmqi96rv7002tnq079eha9t4b` | complete |
| Khakas                          | `kjh`         | `cmqiam2dh007jnr07i8m96tra` | complete |
| Khmer                           | `km`          | `cmqie95uy00evnq07dc9tqagz` | complete |
| Lao                             | `lo`          | `cmqiaq1nn007znr07qss7tcb5` | complete |
| Nepali                          | `ne-NP`       | `cmqi732rk00himf07x3hy8twt` | complete |
| Ngiembon                        | `nnh`         | `cmqioh7eo00zcnq07acxohqzn` | complete |
| Ngomba                          | `jgo`         | `cmqi97py20025nr07in93znww` | complete |
| Ngombale                        | `nla`         | `cmqi6pf7o00g0o507a566s2db` | complete |
| Nigerian Pidgin English         | `pcm`         | `cmqigs4fs00jtnq07o0txdcvm` | complete |
| Northern Hindko                 | `hno`         | `cmqi92xdx0021nq07etytyjx8` | complete |
| Northern Sotho                  | `nso`         | `cmqiapqo1007vnr07uogmwf44` | complete |
| Northwest Gbaya                 | `gya`         | `cmqi92sm70017nr073k30zych` | complete |
| Norwegian Bokmål                | `nb-NO`       | `cmqi72cq900h6mf07dr1ev961` | complete |
| Norwegian Nynorsk               | `nn-NO`       | `cmqi72xiw00hemf07um5n1jk7` | complete |
| Nuasue                          | `yav`         | `cmqi9783o001znr072zwxveon` | complete |
| Nyungwe                         | `nyu`         | `cmqi6pagy00f0mf07z9us86wy` | complete |
| Nüpode Huitoto                  | `hux`         | `cmqiglrbo00hbnr07ck8co7cp` | complete |
| Oadki                           | `odk`         | `cmqiglh6400grnr07x3fdx6ol` | complete |
| Occitan                         | `oc`          | `cmqi9c2sh003rnr07kpbmcrdx` | complete |
| Odia                            | `or`          | `cmqiny4d900xqnq0779zijp37` | pending  |
| Orizaba Nahuatl                 | `nlv`         | `cmqigourq00jdnq07m5oqpktn` | complete |
| Ormuri                          | `oru`         | `cmqio1smz00ysnr0789x2tj38` | pending  |
| Ossetian                        | `os`          | `cmqie9asa00cfnr07ym5l5aw8` | complete |
| Ouldémé                         | `udl`         | `cmqi6h8mp00e4mf07ohctwg4y` | complete |
| Pahari-Pothwari                 | `phr`         | `cmqi9csv20061nq07a7pzrd1w` | pending  |
| Paiwan                          | `pwn`         | `cmqigs15l00i7nr07425s2r5d` | pending  |
| Palula                          | `phl`         | `cmqio2bjn00y6nq07kxuf99dj` | pending  |
| Parkari Koli                    | `kvx`         | `cmqi988xh002gnr07ld9i4fcx` | pending  |
| Pashto                          | `ps`          | `cmqim2e2c00t8nq07jywr0vbi` | pending  |
| Persian                         | `fa`          | `cmqinhw5100v8nr07gyg5gi4v` | pending  |
| Polish                          | `pl`          | `cmqinmu2a00winq07gyrtri0q` | pending  |
| Portuguese                      | `pt`          | `cmqinmnkf00w8nr07hkbbxgw7` | pending  |
| Punjabi                         | `pa-IN`       | `cmqi6zuw000gamf07owiuz3wl` | complete |
| Puno Quechua                    | `qxp`         | `cmqio1wd400xynq079ifamac6` | complete |
| Quechua Ambo-Pasco              | `qva`         | `cmqi6l6rd00fmo507hl4oiplq` | pending  |
| Quechua Arequipa-La Unión       | `qxu`         | `cmqi9x7yx005dnr07mo68a08v` | pending  |
| Quechua Cajatambo               | `qvl`         | `cmqi6hk0d00ecmf07m557157t` | pending  |
| Quechua Chanka                  | `quy`         | `cmqie9c3700cjnr078l71gt9y` | pending  |
| Quechua Chiquián                | `qxa`         | `cmqi6g7z200dsmf07gkncskx6` | pending  |
| Quechua Corongo Ancash          | `qwa`         | `cmqi914oa001hnq07ps0z1iqg` | pending  |
| Quechua Jauja Wanka             | `qxw`         | `cmqi9c7f30058nq07kcwk3tay` | pending  |
| Quechua Pasco Santa Ana de Tusi | `qxt`         | `cmqi9byn00051nq07sepglfct` | pending  |
| Quechua Santiago del Estero     | `qus`         | `cmqi92ctd0013nr07rsn2lvrt` | pending  |
| Quechua Sihuas Ancash           | `qws`         | `cmqi6bsix00eqo5078x6n4v1l` | pending  |
| Quechua Yanahuanca              | `qur`         | `cmqi6kw6m00esmf07kbllsiix` | pending  |
| Quechua Yauyos                  | `qux`         | `cmqi9bsm9003lnr07f3r7za6j` | pending  |
| Romanian                        | `ro`          | `cmqinu08d00xenq07ha33e29y` | pending  |
| Romansh Sursilvan               | `rm-sursilv`  | `cmqilxycc00rcnr071b156jdl` | pending  |
| Romansh Vallader                | `rm-vallader` | `cmqi6zbmg00g2mf07uqhc228t` | pending  |
| Rukai                           | `dru`         | `cmqi8uhp2000jnq072niarffi` | pending  |
| Russian                         | `ru`          | `cmqinj9g500vsnr07qf4hmr3j` | pending  |
