---
license: cc0-1.0
language:
- ab
- abb
- ady
- af
- ajg
- am
- an
- ar
- as
- ast
- az
- ba
- bas
- bax
- bba
- bbj
- bce
- bci
- be
- beb
- bfd
- bft
- bg
- bgp
- bkh
- bkm
- bn
- bnm
- bnn
- br
- brh
- bri
- bsk
- btv
- bum
- byv
- ca
- cjk
- ckb
- cnh
- cpy
- cs
- cv
- da
- dag
- dar
- dav
- de
- dmk
- dml
- dua
- dv
- dyu
- ebr
- eko
- el
- en
- eo
- esu
- et
- eto
- eu
- ewo
- fan
- fi
- fmp
- fr
- fub
- fue
- gej
- ggg
- gid
- gig
- giz
- gju
- gl
- gn
- gsw
- gv
- gwc
- gwt
- ha
- haz
- he
- hem
- hi
- hr
- ht
- hu
- ia
- ibb
- id
- ig
- ipk
- is
- it
- ja
- jqr
- ka
- kdh
- kmr
- ko
- ksf
- kw
- ky
- lg
- lij
- lrk
- lss
- lt
- ltg
- luo
- lv
- lzz
- mau
- mbo
- mcf
- mcn
- mcx
- mdd
- mdf
- mgg
- mhk
- mhr
- mk
- mki
- ml
- mn
- mr
- mrj
- ms
- mse
- mt
- mua
- mug
- mve
- mvy
- mxu
- myv
- ncx
- nl
- nmg
- nmz
- nr
- plk
- prq
- qvj
- rof
- rup
- rw
- sq
- tar
- tay
- var
- wes
- xmf
- yue
- zoc
language_bcp47:
- fy-NL
- ga-IE
- hy-AM
- zh-CN
- zh-HK
- zh-TW
task_categories:
- automatic-speech-recognition
tags:
- common-voice
- mozilla-data-collective
- scripted-speech
- multilingual-asr
- asr
pretty_name: Common Voice Scripted Speech 25.0
---

# Common Voice Scripted Speech 25.0

This repository is being prepared as a row-normalized multilingual
ASR dataset built from Mozilla Data Collective Common Voice
Scripted Speech 25.0. The normalized rows are the main deliverable:
`audio`, `sentence`, `locale`, `language`, `split`,
`source_dataset_id`, `source_archive`, and upstream Common Voice
metadata where present.

During staging, the original MDC `.tar.gz` archives are preserved
with checksums as source material. They provide provenance and make
the normalization reproducible; the final user-facing dataset should
load like other Hugging Face ASR datasets, with language and split
fields available for filtering.

## Status

- Manifest entries: 164
- Archives staged: 79
- Archives still downloading or missing locally: 85
- Current staging layer: original MDC `.tar.gz` archives plus
  manifest metadata.
- Target dataset layer: normalized rows across all downloaded
  languages, preserving upstream `train`, `dev`, `test`,
  `validated`, `invalidated`, `other`, and reporting files where
  present.

## Why this exists

Mozilla moved recent Common Voice downloads to Mozilla Data Collective.
This mirror does the collection and bookkeeping work once, with
checksums and source dataset IDs, giving ASR users one documented
place to retrieve the combined collection.

## Prior Art

- `mozilla-foundation/common_voice_17_0` points users to Mozilla Data
  Collective for
  post-October-2025 Common Voice access.
- `legacy-datasets/common_voice` is the older Hugging Face packaged
  Common Voice dataset and uses `cc0-1.0` metadata.
- Many community repos publish per-language or processed Common
  Voice variants. This repo is focused on one combined, traceable,
  row-normalized Common Voice Scripted Speech 25.0 dataset.

## Dataset Shape

The normalized table should use one row per audio clip and keep the
upstream split name in a `split` or `original_split` field. Expected
columns are:

- `audio`
- `sentence`
- `locale`
- `language`
- `split` / `original_split`
- `source_dataset_id`
- `source_archive`
- upstream Common Voice metadata such as `client_id`, `sentence_id`,
  `sentence_domain`, votes, age, gender, accents, variant, and
  segment when present
- `duration_ms` from `clip_durations.tsv` when available

## Quality Views

Quality filtering should be published as optional views or manifests
layered on top of the canonical normalized rows. The local
`omni-curator` notes treat scripted Common Voice like read or
broadcast speech: Scribe WER tiers of excellent <= 5%, good <= 15%,
and acceptable <= 25%, plus duration and characters-per-second
backstops for obvious audio/text mismatch. Those filters are useful
for training recipes, while the base dataset should keep upstream
split and validation state visible.

## Archives

| Language | Locale | MDC dataset ID | Archive | Size | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| Abkhaz | `ab` | `cmn29f1jp014to1077s6w9o6g` | `common-voice-scripted-speech-25-0-abkhaz-10f6100b.tar.gz` | 5113659167 | `3a88afb4bbd050f78e1b24d76d8feef91f2c4cd3b52aa689cfcd64a35dbb598a` |
| Adamawa Fulfulde | `fub` | `cmn2cijyg01gamm07434y1o8l` | `common-voice-scripted-speech-25-0-adamaw-f74cee41.tar.gz` | 262798335 | `ae44594dc694a4df2567b7629e88b52709ca836853fa5e20f392d7bd46579d29` |
| Adja | `ajg` | `cmn2cidwc01g6mm07bv38rtqx` | `common-voice-scripted-speech-25-0-adja-8f935683.tar.gz` | 264920483 | `c0f6db24d06708d121a38ebf039fc176249304a68ac666de5b265c67cedb18e4` |
| Adyghe | `ady` | `cmn2e80ea01kmmm07lzsoe5z9` | `common-voice-scripted-speech-25-0-adyghe-c34f1d66.tar.gz` | 1805000755 | `66f874bef7d426717af5a2338961f56c0451907009d4528ec24170b70f3e37d3` |
| Afrikaans | `af` | `cmn29hngy0188mm07kzspi4d4` | `common-voice-scripted-speech-25-0-afrika-f9ba253e.tar.gz` | 36039823 | `7976ee567d9e26aea4cadecad6f7c3cabf8efd598dea18869621ddb4585d299a` |
| Albanian | `sq` | `cmn29zkso01aimm07wb1ar40j` | `common-voice-scripted-speech-25-0-albani-c77d0397.tar.gz` | 198575142 | `7b4575fb73be1f2f5d0bc8afbb0a8a70cbc3e1c88c395a8728d1b895bc5ad3b1` |
| Alsatian | `gsw` | `cmn29hk3a0184mm072ledt03r` | `common-voice-scripted-speech-25-0-alsati-85ef00d4.tar.gz` | 33852437 | `df415a73dfc3af13c07a58b9ed2616b66a58149a24667ef292eee0cadc6260af` |
| Amharic | `am` | `cmn29lq6f0164o10748yd3o7w` | `common-voice-scripted-speech-25-0-amhari-81b86f27.tar.gz` | 61719846 | `4ae6148ece4641e7eaeb0231a7356c734332a4e22a6ed25fa8294661f17ddf91` |
| Arabic | `ar` | `cmn2g7uu701fqo1072r5na25l` | `common-voice-scripted-speech-25-0-arabic-3f852591.tar.gz` | 3524109316 | `f47fc303117170fa8a8d356650f39c54c9731ef79f9a8ae17c2671a724315eeb` |
| Aragonese | `an` | `cmn2cpd2m01himm07yj9w1lxn` | `common-voice-scripted-speech-25-0-aragon-f45d0697.tar.gz` | 375485423 | `96e642705c63c35f75b4fd432d34a9e22d7fc85305a16ec2d4d7384ee94ac2eb` |
| Armenian | `hy-AM` | `cmn2e8k9z01kymm07yqqy4bk1` | `common-voice-scripted-speech-25-0-armeni-25882ac4.tar.gz` | 1222233164 | `127f0254953d8cf6c083f169a04c4d7176df8cd383ba9e84b96c7cdb0c4f67fe` |
| Aromanian | `rup` | `cmn1q11oz00v7o107a25yjpsp` | `common-voice-scripted-speech-25-0-aroman-da4c959d.tar.gz` | 7266270 | `a2db6e000cc222d7665bf32333fbe32b381eb957aa179a352fa0c072eb36483d` |
| Asheninka Perene | `prq` | `cmn29z01q01aamm07oyzrkbnb` | `common-voice-scripted-speech-25-0-asheni-6cedf608.tar.gz` | 196537639 | `1a07a265660445cb24acc6d0cdf1980efce9987a4e062f73a18525f2a4a3eb1a` |
| Asheninka South Ucayali | `cpy` | `cmn2aoh0g01d3mm07de71ivsg` | `common-voice-scripted-speech-25-0-asheni-bdbc65fb.tar.gz` | 210669797 | `90ec3fc32dd489980eb2f4d1b3bf6959968164449e5a7cae2e61120858e2a56b` |
| Assamese | `as` | `cmn29vpx50196mm07mm08x8xm` | `common-voice-scripted-speech-25-0-assame-61628770.tar.gz` | 166888627 | `19c21426dd862743910774898e0c2a945693f715cd22110b6c0e0c986dd6ccdd` |
| Asturian | `ast` | `cmn1q4g7i00y4mm07eb1yeh11` | `common-voice-scripted-speech-25-0-asturi-24d15c2a.tar.gz` | 43146073 | `9bb5a42462b1e29f82f12b8b6080d605774fb5fde5be13847b802efc10c051c6` |
| Atayal | `tay` | `cmn2bjxwf01eymm071zgnlf93` | `common-voice-scripted-speech-25-0-atayal-f20f0664.tar.gz` | 258255513 | `749e47e3272981c6e56589e60dc9af409fb44e677fa1fd2aa24167778e705842` |
| Azerbaijani | `az` | `cmn29hqvk015ko107fblsr5ay` | `common-voice-scripted-speech-25-0-azerba-ddcce9c6.tar.gz` | 38470282 | `a091621f2509f68507fa902258c10bcefc2b04e3d8b8395e874482eac60364e1` |
| Bafut | `bfd` | `cmn2awjf501ehmm07zqozqv3q` | `common-voice-scripted-speech-25-0-bafut-cb2ff3ee.tar.gz` | 230058925 | `c60864c6b2a16acb4882a8b8b1870cdf026738c2ffb7542f44b8d090f5fd0a83` |
| Baatonum | `bba` | `cmn2chcee01c7o10782n6ozvp` | `common-voice-scripted-speech-25-0-baaton-ff56add5.tar.gz` | 284227558 | `38c4a7e78f1b2d1a8023e2e5d8922541023315b51e14bca3865e0d39e1e27b4e` |
| Bafia | `ksf` | `cmn1qeyjg00xvo107ica6bh2f` | `common-voice-scripted-speech-25-0-bafia-dce8b73c.tar.gz` | 359207958 | `92e56c8dffb1f859e2edeedb82c720015dc31b3b202c5dda1e9334953fdc8aaf` |
| Bakoko | `bkh` | `cmn2bjje101eumm07avcbr9fk` | `common-voice-scripted-speech-25-0-bakoko-88e99776.tar.gz` | 261694804 | `e39b4a4b6b90b1b741031f7291f23110f32dad64ac9c80c6844f5ff8f2d7f564` |
| Balti | `bft` | `cmn2cpjc301hqmm072p9pm8c1` | `common-voice-scripted-speech-25-0-balti-3d2bbf36.tar.gz` | 363878525 | `3b57141ebadd3065fa0128e83a67a7fa1da82312edee6a370a75b53837515101` |
| Bamun | `bax` | `cmn2c99q101b6o1079w6l6r5l` | `common-voice-scripted-speech-25-0-bamun-44b5fcbf.tar.gz` | 243212469 | `863a5809c6b3c862b89cb3daad4d0c82240b8d696713eb8f9e304b6e514c1d06` |
| Bamvele | `beb` | `cmn2awma9019vo1077myulutp` | `common-voice-scripted-speech-25-0-bamvel-66c08b99.tar.gz` | 230589104 | `e4cd86b224170bef551d0911839aef0d76ed60fc47d55fb95e1487791c35946e` |
| Bankon | `abb` | `cmn2c9uwa01fgmm073lt5l0i0` | `common-voice-scripted-speech-25-0-bankon-bb562f9c.tar.gz` | 238216372 | `fa9e488fbc13a177c84d1354e400538884f37a2ca19e0bbb21de5ea284dfa3b1` |
| Bunun | `bnn` | `cmn2c96vl01b2o107ys2d3rim` | `common-voice-scripted-speech-25-0-bunun-785947f9.tar.gz` | 246986349 | `d469fdf29db55236b5f25d1013999d7c3613acd1317c9a92309f076e32947221` |
| Bulu | `bum` | `cmn2aok00018vo107c7he8w00` | `common-voice-scripted-speech-25-0-bulu-aa423bd3.tar.gz` | 210579351 | `0eae56badd31300352fc7da7a7b3bc08e56cdb7c040b450b65581fe496b1fc16` |
| Bulgarian | `bg` | `cmn2coplj01gqmm07dbibr68y` | `common-voice-scripted-speech-25-0-bulgar-2af845dc.tar.gz` | 450725864 | `5e79f6982b42dfeece019c8e4522385a3152788bf7103e89ec653743500d59ee` |
| Brushaski | `bsk` | `cmn2avimf019no107bb37vfx8` | `common-voice-scripted-speech-25-0-brusha-43bd4424.tar.gz` | 229279161 | `2ab4aa98b375e7368995a8b2342973e962410a5e0de78fa8946d296c17dfe8b5` |
| Brahui | `brh` | `cmn2aefkq01b8mm07dblhdy6q` | `common-voice-scripted-speech-25-0-brahui-7f308fad.tar.gz` | 207687899 | `77445add06f31b8cc368061316c8638f5a4f5641e77cbd3fc58e5dd19a0ee846` |
| Breton | `br` | `cmn2cx1ko01djo1076muxlj24` | `common-voice-scripted-speech-25-0-breton-0b478ca3.tar.gz` | 806464425 | `a8c510f0062f499a9772ab4fbc31725f85b0166057f749e33369447605f757d2` |
| Borgu Fulfulde | `fue` | `cmn2auup5019bo107zvq9bitg` | `common-voice-scripted-speech-25-0-borgu-8ddbd16a.tar.gz` | 221111740 | `54d64c6cf1e7d1db9fd5efbcfe80647a3db3d71204c049c28776708ff2e69a49` |
| Bengali | `bn` | `cmn3ipo8b00ejmi079e8upl2k` | `common-voice-scripted-speech-25-0-bengal-6b0bf35a.tar.gz` | 26671663045 | `de9efc688c172ef81c9f47771dd8dd673111ae3c43ac362999d0d517603d055d` |
| Belarusian | `be` | `cmn4xg3a900d3nu075gnh4jpt` | `common-voice-scripted-speech-25-0-belaru-87cdbfb0.tar.gz` | 38875606311 | `4539a19634b191011e376da84b5bfd43060989eb514e49e91774084177be8f84` |
| Bateri | `btv` | `cmn2an7at018ro1076m25xok8` | `common-voice-scripted-speech-25-0-bateri-074a2ad7.tar.gz` | 215828535 | `491b29a0b49df7b67a9f71faa5a89c1961d4f458e67227f1d5971c60fe4f7568` |
| Batanga | `bnm` | `cmn2cpwp901i2mm07hpqh5mre` | `common-voice-scripted-speech-25-0-batang-5a28ff15.tar.gz` | 336822349 | `72bfeaf12e2528d3a03c92ecc43c42ac364c8befc4d1441c62de72ed0b96671e` |
| Basque | `eu` | `cmn2hwe0d01n8mm07wug9r5he` | `common-voice-scripted-speech-25-0-basque-2008364c.tar.gz` | 15552486520 | `4cea1dc7b49e5f14d0ba1e24bb46177fcf90e9fe565569dea02ab055324929c2` |
| Basaa | `bas` | `cmn2bk0jn01f2mm07dv6v1jbt` | `common-voice-scripted-speech-25-0-basaa-8ac0e053.tar.gz` | 254569391 | `bbe542459f2eeb732759bd96efce4669084f754563672a0edc0b02828359ee57` |
| Bashkir | `ba` | `cmn29exhf014po107d6mpl5ec` | `common-voice-scripted-speech-25-0-bashki-df84974a.tar.gz` | 5471729194 | `b6ec4fc2573b9538422994ea420920041e408af92f0e15bd8d2bff5fbb535561` |
| Baoule | `bci` | `cmn2cqa8r01immm074kdpi5v8` | `common-voice-scripted-speech-25-0-baoule-1f090222.tar.gz` | 298992812 | `3c0f5fcd4a84e4cb4d67f6646effa8bc45840d21eae5cb7a504db1b28cb1a0b9` |
| Czech | `cs` | `cmn2h5zd801h3o1075tita1ap` | `common-voice-scripted-speech-25-0-czech-a54711a9.tar.gz` | 5970163541 | `99366f369ee223d9727ccef4545c2ea3438cb8ff6ebcfcf8e6970db3d28968e9` |
| Croatian | `hr` | `cmn29gs96017jmm0705ypjod7` | `common-voice-scripted-speech-25-0-croati-0ec9b2a5.tar.gz` | 793494 | `64f2a7f1a9b7ec600d4f18c59139741dc12ed1adf1d3f534bc6bee2ff53c2263` |
| Cornish | `kw` | `cmn2chs8a01cno107aydgie03` | `common-voice-scripted-speech-25-0-cornis-21e4215a.tar.gz` | 272873692 | `25ff0f54a7e973eedf7e3375eb2d50989468eac2b16b88466d750c26f700dd7e` |
| Copainalá Zoque | `zoc` | `cmn2ansfl01cfmm07ivk4w1rf` | `common-voice-scripted-speech-25-0-copain-9f62c425.tar.gz` | 213980458 | `63cb37757d4210b08e53493df74c96ea79cf18fadb3898220363c6d26f2e8189` |
| Chuvash | `cv` | `cmn2cxjsd01dzo107sraxorft` | `common-voice-scripted-speech-25-0-chuvas-479c454c.tar.gz` | 700495298 | `28d2fbd488799a6bb7ee6faa6ef001925fb85358096cefe3442b7069c70138dc` |
| Chokwe | `cjk` | `cmn1qeu5500xro107r3og0kg7` | `common-voice-scripted-speech-25-0-chokwe-38a72767.tar.gz` | 275785495 | `a355fb55d25a4af2485825d091f6a222b7fa621f2e74940e328f76ed26373b7a` |
| Chinese (Taiwan) | `zh-TW` | `cmn2g7eaj01fio10769r1m96n` | `common-voice-scripted-speech-25-0-chines-e84858c5.tar.gz` | 3165560607 | `9f88c8978b145ad13d502c2df5010f942830d58a21f05854eee41b20126b1028` |
| Chinese (Hong Kong) | `zh-HK` | `cmn2g8zqd01m2mm07prcmehku` | `common-voice-scripted-speech-25-0-chines-3debabbc.tar.gz` | 3674912896 | `c34935c4a75bb1a13f4fb9a4ff570291a55c53095a347c4e7b9b0a43ac7cb0d9` |
| Central Alaskan Yup’ik | `esu` | `cmn1q4qsg00vyo107sjh2vufw` | `common-voice-scripted-speech-25-0-centra-a34567da.tar.gz` | 142933775 | `1aa45ce99e45d2172a5c1ef0347d486a43546788dd5d0b5d5b18938245b9ef1e` |
| Catalan | `ca` | `cmnd4la5a02fwmh074t1fx5y9` | `common-voice-scripted-speech-25-0-catala-4707e7c7.tar.gz` | 49060446208 | `df37f1ef9caae376dc1671aa5f81d350b283333b9a52bd5ebbfde9a33950a3e5` |
| Chinese (China) | `zh-CN` | `cmn3iaztg00e4mb070uvufz7q` | `common-voice-scripted-speech-25-0-chines-89c8536a.tar.gz` | 22953106130 | `59ecd91ef0f2a23b5635c12c3dfab833721e4bb76518995811d5690893203d79` |
| Cantonese | `yue` | `cmn29rqn9016to107eniyak65` | `common-voice-scripted-speech-25-0-canton-bea0224a.tar.gz` | 6431854197 | `e3924bd720dee56c2849013320c62b73b35338d92c787a4eee3b074fc2e4638b` |
| Central Kurdish | `ckb` | `cmn2g9npx01g2o107hentahj9` | `common-voice-scripted-speech-25-0-centra-3aff9382.tar.gz` | 3859763014 | `76432036cb19f61072c4d6e72a0ba1f68c9694a42d12c5b3ec21ea25cad01d82` |
| Central Puebla Nahuatl | `ncx` | `cmn2c9fhc01beo1074z081p8t` | `common-voice-scripted-speech-25-0-centra-e21f3063.tar.gz` | 243165666 | `b2c35410070ba3b983926612806a85193a2b481499593acfc2360b1ee968d64d` |
| Cameroon Pidgin | `wes` | `cmn1qa0u300z4mm07egfgo1k4` | `common-voice-scripted-speech-25-0-camero-a53295d7.tar.gz` | 209729941 | `46b49ef2899bf3d179f1b23f82e077849c1680054b04b989eca887d12e3c6bbd` |
| Central Tarahumara | `tar` | `cmn2aodxp01czmm078t6zlp2s` | `common-voice-scripted-speech-25-0-centra-be49e846.tar.gz` | 211018236 | `9975f56d0c9b39a1e33c704451b77121d0cba9d5b8f0c9f4f43c154b893293e0` |
| Dagbani | `dag` | `cmn2cy2su01iymm07xfr6ul2b` | `common-voice-scripted-speech-25-0-dagban-ec0b9801.tar.gz` | 552191590 | `f0b067e954219ffe56d339d063bbd0ceb74674231f6240e4a334cec6f84831c5` |
| Dameli | `dml` | `cmn2awoti01elmm07gfwlnqs8` | `common-voice-scripted-speech-25-0-dameli-821f04fa.tar.gz` | 230760481 | `eab336e5b6210b475e31c07bc8eef5f060d046131527d6e1d0c0ca876257bb5f` |
| Danish | `da` | `cmn2cptsh01hymm07mulngxv0` | `common-voice-scripted-speech-25-0-danish-05bd9c29.tar.gz` | 339266849 | `d51753540bbbb215aa449b162bf7f07abd0c8bb0d36cbae9262119df4c4169e3` |
| Dargwa | `dar` | `cmn2cosov01gumm070n74b5i6` | `common-voice-scripted-speech-25-0-dargwa-9bbbd0f4.tar.gz` | 399955795 | `230dc639d6dfbc369891082cd0270bbd20f725949d4c73b2759925cd67c11173` |
| Dawoodi | `dmk` | `cmn2al6jd01bvmm07tfcmt9k0` | `common-voice-scripted-speech-25-0-dawood-4b521d50.tar.gz` | 216969975 | `5203e03d5848906083df76cd6c954df29d662e2a47ed1718f98b5002ac171c9a` |
| Dhatki | `mki` | `cmn29z2pb01aemm072es1wj06` | `common-voice-scripted-speech-25-0-dhatki-cac8afce.tar.gz` | 196867286 | `dbd6a2939aed327f141970fe8b329813c143011f7df288ab516a7a9b035c761e` |
| Dhivehi | `dv` | `cmn2e8dkd01ero107fgwmo0qz` | `common-voice-scripted-speech-25-0-dhiveh-714e9de2.tar.gz` | 1415477900 | `5a284602ed26250450b761b1ebfebf590697a220a75a3226746e2d610601b5d5` |
| Dholuo | `luo` | `cmn2e7tbt01kemm07drju0bcf` | `common-voice-scripted-speech-25-0-dholuo-0d1a6b6d.tar.gz` | 2229793887 | `a210be0b5c9f319f37874e7b24dd9dfa15001e37b49211fd6e7866d0bbfb1866` |
| Dioula | `dyu` | `cmn1q3sgr00xwmm07t7te56k4` | `common-voice-scripted-speech-25-0-dioula-67c804b3.tar.gz` | 10797797 | `c8eb8616660253cdffaac95bb0802919d63ff4246308b999f59ba667553899f6` |
| Duala | `dua` | `cmn2ch6p301bzo107o6gib3e7` | `common-voice-scripted-speech-25-0-duala-8b2a4573.tar.gz` | 292401751 | `d261c0e2bb7ad3b5185050e551943bd240f6070e1ef4409b7486c1093a1abc42` |
| Dutch | `nl` | `cmn2g7nu901fmo107a1ydn0n5` | `common-voice-scripted-speech-25-0-dutch-47b12a16.tar.gz` | 3395193380 | `d0d40534eb023bd2845f4dcd5f5586896d6fa9c8aee19016bad4ce2da1e507f2` |
| Ewondo | `ewo` | `cmn2ch06r01bro107450c5m72` | `common-voice-scripted-speech-25-0-ewondo-bd6511b4.tar.gz` | 295260851 | `6286bf0075fd841a8328cba31d52812b5f76bc90e0dafdd6d8b2da893b66907c` |
| Eton | `eto` | `cmn2ae4ke01awmm075jsuvk3j` | `common-voice-scripted-speech-25-0-eton-bab6f2b4.tar.gz` | 202113285 | `fa476adddbd8d01ae910f6f71872cda0c5517ba4f16f6c641e9baf6645fc3286` |
| Estonian | `et` | `cmn2e880l01kumm07i9upoz99` | `common-voice-scripted-speech-25-0-estoni-287171ef.tar.gz` | 1603835772 | `d1f4a97ba736f7bc8cec6084b43d587af392a6c0b6a5a9680ad5afb159dad65a` |
| Esperanto | `eo` | `cmn4o8691005pnu07fxmq06px` | `common-voice-scripted-speech-25-0-espera-e42e8ff1.tar.gz` | 41880993802 | `1fdcc230fdb5ac6c624b3a6896976df3bd81a4c198043df673841ec7e3b92d9d` |
| Erzya | `myv` | `cmn29m1vc016co107za0i4zp0` | `common-voice-scripted-speech-25-0-erzya-4a0a0aa7.tar.gz` | 68338555 | `65450d92435898b2cb0943cfc2d45a35f2b501efb3cbb5174ee77e50be0d881b` |
| Ekoti | `eko` | `cmn29v9kx017do107ce0r76ku` | `common-voice-scripted-speech-25-0-ekoti-b45fc663.tar.gz` | 176504080 | `2b93e69837ca6751336284784ef2ba09bc0f14fb8ea22eaf67c76609535d843f` |
| Iñupiaq | `ipk` | `cmn1q5h2g00ycmm074ucll31q` | `common-voice-scripted-speech-25-0-i-upia-a697a7c6.tar.gz` | 153468347 | `ce774bec16071d4741f6bfc3943bdc473157dbc00cd7e1bf46cc21d3f22e35f3` |
| Italian | `it` | `cmn2h0yei01msmm07u8z5vu87` | `common-voice-scripted-speech-25-0-italia-ef57e630.tar.gz` | 10424613036 | `486958feed03d52d2cccc7aa35807c8bf7b118011c1c672b762107c4bc59a80f` |
| IsiNdebele (South) | `nr` | `cmn1pyaq600uzo1079yby8vs2` | `common-voice-scripted-speech-25-0-isinde-075b5110.tar.gz` | 890459 | `4c155fd1a4f3d712b1d2a2b7000cbbb0187be9bfa82a7c5d9b541b93c12fdb90` |
| Irish | `ga-IE` | `cmn2cp9uy01hemm07jfogi1zf` | `common-voice-scripted-speech-25-0-irish-34a05cec.tar.gz` | 379570015 | `d49efb897f66ea61d53cdc5d7f7da3766584cc5a8b0126d9d0100f594d34eb15` |
| Interlingua | `ia` | `cmn1qf7ns00y3o107oa3cxsut` | `common-voice-scripted-speech-25-0-interl-281626ae.tar.gz` | 421849127 | `37769614c273586290dda0fd8617b22de8820f7dcd1d579e8234695cbe6e96c6` |

## License

The Mozilla Data Collective records for these archives report
Creative Commons Zero v1.0 Universal (`CC0-1.0`). See `LICENSE`
and https://spdx.org/licenses/CC0-1.0.html.
