# Tajik YouTube Candidate Channels — Round 2 (conversational diversity)

Discovery date: 2026-06-10. Goal: conversational Tajik (talk shows, podcasts, interviews, call-in/radio, vlogs) + regional accents (Khujand/Sughd, Kulob/Khatlon, Badakhshan), since the v2 corpus is strong on news anchors and audiobooks but weak on spontaneous speech.

Method: yt-dlp `ytsearch` with Tajik-script queries, then per-channel verification — sampled recent titles for language and enumerated the videos tab (counts capped at 1000, so "1000+" means at least that). Every channel below was confirmed live with majority-Tajik titles; bilingual (RU/TJ) channels are noted and rely on the tgk_Cyrl language gate.

Registry observations (no edits made):

- `tvt_tojikiston` (@TVTOJIKISTON) resolves to **the same UC id** as `tv_tajikistan` (UC0NDr4nXx-c4cto_RLz9UPA) — a duplicate pair in sources.py. The actual state-TV channel «Телевизиони Тоҷикистон (TVT)» is UCl24XHk8XfLXPCJjpLgR9Ng (news; not proposed — saturated).
- `radio_ozodi` (@Radio-Ozodi) resolves to Ozodivideo (UCyTDJHSzuKZ2JZO41c5Do1A), and `ozodagon` (@ozodagontv) to Хабаргузории Озодагон (UCJeof7I5c_8L_R2JHV7jKcg) — both prior-doc "candidates" are therefore already covered.
- `asiaplus` (@asiaplustj) resolves to Аsia-Plus TV (UCxaPn-OJIG6FY2oM1TB0ZSw) — also already covered despite the different id in the round-1 notes.

## New channels (30)

| #   | Name                            | URL                                            | Videos | Content type                                                        | Tier  | Why it adds diversity                                                                                 |
| --- | ------------------------------- | ---------------------------------------------- | ------ | ------------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------- |
| 1   | PRESSA TJ                       | youtube.com/channel/UCww_Gm1xbEzCglzlNybgidg   | 1000+  | «Рӯ ба Рӯ» long-form celebrity/culture interviews                   | noisy | Flagship sit-down interview series; spontaneous two-speaker speech                                    |
| 2   | ПАЁМ tv                         | youtube.com/channel/UCOqafrVhGQVpv8SsIu_wcxw   | 1000+  | Opposition/diaspora interviews (Шавкати Муҳаммад)                   | noisy | Unscripted political interviews, emotional registers                                                  |
| 3   | Peshsaf TV                      | youtube.com/channel/UCYtCNoGsNKl\_-iSXY0u88TQ  | 88     | «Суҳбати озод» open interviews, women's craft show                  | noisy | Free-flowing conversation incl. female guests                                                         |
| 4   | Радиои Нав 95.5 FM              | youtube.com/channel/UCBDdW8ZjdwNfXkW6K2aK-nQ   | 502    | FM radio studio talk (Salomaleykum talk show, live guests)          | noisy | Closest thing found to call-in radio; banter + overlap                                                |
| 5   | 90 Дақиқа                       | youtube.com/channel/UCct-HZ1xW51szliQsuRApdg   | 461    | Football talk show / debate                                         | noisy | Multi-speaker sports argument — a register absent from corpus                                         |
| 6   | Varzish TV (official)           | youtube.com/channel/UCQNfOjRJg2QdrKAMvZ7GF7A   | 1000+  | Sports TV: live commentary, locker-room interviews                  | noisy | Excited commentary + post-match interviews                                                            |
| 7   | ФАРАҲ онлайн                    | youtube.com/channel/UCwKOMPNY7mlasbNXa_BK9jw   | 492    | Interviews + social-issue street reports                            | noisy | Mixes studio interviews with on-location voices                                                       |
| 8   | AHRORI CHANNEL                  | youtube.com/channel/UC4uWE-0UyMjM4dxECi4D9UA   | 206    | Khujand-area stories, artist interviews                             | noisy | Northern (Sughd) speakers, informal interviews                                                        |
| 9   | Телевизиони Хатлон              | youtube.com/channel/UCzKIJe_BQYbSw9qQhv-8T9w   | 1000+  | Khatlon regional state TV: reports, field interviews                | noisy | Kulob/Khatlon accent at broadcast scale                                                               |
| 10  | Телевизиони Кулоб               | youtube.com/channel/UCfyx1E\_-nKrivjUX_KJ9nhg  | 859    | Kulob city TV: programs, weather, portraits                         | noisy | Kulob accent; mixes clean bulletins and field speech                                                  |
| 11  | Телевизиони вилояти Суғд        | youtube.com/channel/UCKYGQHdKytuD9fEeSqrB-Mg   | 1000+  | Sughd regional state TV incl. youth talk programs                   | noisy | Khujand/Sughd accent; studio talk («Дунё пур аз ҷавонист»)                                            |
| 12  | Пайки Сугд                      | youtube.com/channel/UCOpejZWtEKAjElK8CC0IrDg   | 1000+  | Sughd MIA «Милитсия хабар медиҳад» programs                         | noisy | Police field reports: detainee/witness speech, Sughd region                                           |
| 13  | Сежд Pamir Screen               | youtube.com/channel/UCPZ4JBQIloUfU_6heBc-Q9w   | 1000+  | Badakhshan news/commentary («Ахбор аз Бадахшон»)                    | noisy | GBAO voices — only Badakhshan-focused source found; some Pamiri-language and music items, gate needed |
| 14  | CM-1 (Худжанд)                  | youtube.com/channel/UC3rDAd8HP6tv4Sf8fIjsUyQ   | 1000+  | Khujand street reports, bazaar vox-pop, health programs             | noisy | Street interviews in Khujand; bilingual TJ/RU                                                         |
| 15  | СУБҲИ ТОҶИКИСТОН                | youtube.com/channel/UClh30sBA2zMftjGyLtZfDMw   | 975    | Morning-show segments («Субҳ»)                                      | noisy | Studio chit-chat format, multiple hosts                                                               |
| 16  | «Телевизиони Сафина» (official) | youtube.com/channel/UCIjEdRKckv3MXN9_9wqeHog   | 1000+  | Safina's own channel: «Дар деҳа» village visits, talk               | noisy | Distinct from registry `safina` id; villagers interviewed on location, rural accents                  |
| 17  | Хочи Мирзо 2020                 | youtube.com/channel/UCa_zXmCs2QruQrECforM4fw   | 512    | Hoji Mirzo sermons + саволу ҷавоб Q&A                               | noisy | Spontaneous audience Q&A from the best-known Tajik preacher                                           |
| 18  | РОХИ ФАРДО                      | youtube.com/channel/UCRmveynAO64yqvC1jlmMbMw   | 541    | Domullo Akbarjon sermons/stories to live audience                   | noisy | Long unscripted oratory, mosque acoustics                                                             |
| 19  | Суҳроб Одилиён                  | youtube.com/channel/UCzB5GL5-UX3H5YgdMR0ao8A   | 623    | Religious lectures + «Сӯҳбати кушод бо мардум» marathon lives (3h+) | noisy | Open live conversations with callers/audience                                                         |
| 20  | Salimov Mirzo                   | youtube.com/channel/UCDWRQfGBP6e3Ofh2F-IhDmQ   | 1000+  | Daily news/incident commentary                                      | noisy | High-volume colloquial monologue, current vocabulary                                                  |
| 21  | Ҷомеъа Live                     | youtube.com/channel/UCdfM4J2vHsRkjJTndYudMVg   | 169    | Diaspora live commentary                                            | noisy | Unscripted emotional live speech                                                                      |
| 22  | Sulaymoni FORS                  | youtube.com/channel/UCrjeb9IgUggXGqP68_sb3ig   | 48     | Emotional opposition monologues                                     | noisy | Shouted/agitated register, colloquial southern speech                                                 |
| 23  | РУШДИ МИЛЛАТ                    | youtube.com/channel/UCl2ddsg2vGcQSNYKbRTGPxg   | 135    | News commentary, migrant issues                                     | noisy | Migrant-focused colloquial commentary                                                                 |
| 24  | GAYUROV                         | youtube.com/channel/UCB07-cMoeNiSpYnVSf5LTGA   | 35     | Work-abroad explainers + travel vlogs                               | noisy | Diaspora topics (USA/EU jobs) in conversational Tajik; some RU titles                                 |
| 25  | Talabsho Mukimov                | youtube.com/channel/UCK6xYrrzDKSdpQoJe7F25vQ   | 84     | Tajiks-in-USA events, lawyer interviews, vlogs                      | noisy | US diaspora voices; occasional song items, gate needed                                                |
| 26  | SM Sharipov                     | youtube.com/channel/UC-EvNte5h0RtqVlsrJqJD1Q   | 1000+  | True-crime/story voiceover in colloquial Tajik                      | noisy | Huge volume of informal narration («кияй?» register)                                                  |
| 27  | Hamnavo                         | youtube.com/channel/UCSz_lk2CpAedtc4Wnf\_\_TBw | 814    | Dushanbe street vlog / city news walks                              | noisy | Street-level spontaneous speech, passersby                                                            |
| 28  | ГАП НАДОРМ                      | youtube.com/channel/UCWYmF-OgBfGFMdxlKPMoh8w   | 196    | Cars/daily-life vlogs                                               | noisy | Heavily colloquial young-male speech; some RU mixing                                                  |
| 29  | Taj mama                        | youtube.com/channel/UCoE97TS7fekqgoxgyDFSP7g   | 1000+  | Family vlogs (weddings, trips, Khujand)                             | noisy | Female colloquial speech, family crosstalk, northern dialect                                          |
| 30  | M AVZUN                         | youtube.com/channel/UCwcsw18CCxoT0MVZRqR6sLg   | 393    | Women's shopping/family vlogs                                       | noisy | Northern-dialect female vlogger («дидум» forms); some Uzbek mixing, gate needed                       |

## Ready-to-paste `_ch` entries

```python
    # --- round 2: conversational / regional (docs/tajik_youtube_channels_round2.md) ----------
    _ch("pressa_tj", "UCww_Gm1xbEzCglzlNybgidg", "noisy", "PRESSA TJ: Ру ба Ру long interviews."),
    _ch("payom_tv", "UCOqafrVhGQVpv8SsIu_wcxw", "noisy", "ПАЁМ tv: opposition/diaspora interviews."),
    _ch("peshsaf_tv", "UCYtCNoGsNKl_-iSXY0u88TQ", "noisy", "Peshsaf TV: суҳбати озод open interviews."),
    _ch("radioi_nav", "UCBDdW8ZjdwNfXkW6K2aK-nQ", "noisy", "Радиои Нав 95.5 FM: studio radio talk shows."),
    _ch("navad_daqiqa", "UCct-HZ1xW51szliQsuRApdg", "noisy", "90 Дақиқа: football talk/debate show."),
    _ch("varzish_tv", "UCQNfOjRJg2QdrKAMvZ7GF7A", "noisy", "Varzish TV: live sports commentary + interviews."),
    _ch("farah_online", "UCwKOMPNY7mlasbNXa_BK9jw", "noisy", "ФАРАҲ онлайн: interviews + street reports."),
    _ch("ahrori_channel", "UC4uWE-0UyMjM4dxECi4D9UA", "noisy", "AHRORI: Khujand stories/interviews (Sughd accent)."),
    _ch("tv_khatlon", "UCzKIJe_BQYbSw9qQhv-8T9w", "noisy", "Телевизиони Хатлон: regional TV, Kulob accent."),
    _ch("tv_kulob", "UCfyx1E_-nKrivjUX_KJ9nhg", "noisy", "Телевизиони Кулоб: Kulob city TV programs."),
    _ch("tv_sughd", "UCKYGQHdKytuD9fEeSqrB-Mg", "noisy", "ТВ вилояти Суғд: regional TV + youth talk, Khujand accent."),
    _ch("payki_sughd", "UCOpejZWtEKAjElK8CC0IrDg", "noisy", "Пайки Сугд: MIA police programs, field interviews."),
    _ch("sezhd_pamir", "UCPZ4JBQIloUfU_6heBc-Q9w", "noisy", "Сежд Pamir Screen: Badakhshan news/commentary (some Pamiri/music)."),
    _ch("cm1_khujand", "UC3rDAd8HP6tv4Sf8fIjsUyQ", "noisy", "CM-1: Khujand street reports/vox-pop (TJ/RU)."),
    _ch("subhi_tojikiston", "UClh30sBA2zMftjGyLtZfDMw", "noisy", "СУБҲИ ТОҶИКИСТОН: morning-show studio chat."),
    _ch("safina_official", "UCIjEdRKckv3MXN9_9wqeHog", "noisy", "Сафина official: Дар деҳа village talks (distinct from safina)."),
    _ch("hoji_mirzo", "UCa_zXmCs2QruQrECforM4fw", "noisy", "Hoji Mirzo: sermons + audience Q&A."),
    _ch("rohi_fardo", "UCRmveynAO64yqvC1jlmMbMw", "noisy", "РОХИ ФАРДО: Domullo Akbarjon live-audience sermons."),
    _ch("suhrob_odilien", "UCzB5GL5-UX3H5YgdMR0ao8A", "noisy", "Суҳроб Одилиён: lectures + open live talks with callers."),
    _ch("salimov_mirzo", "UCDWRQfGBP6e3Ofh2F-IhDmQ", "noisy", "Salimov Mirzo: daily colloquial news commentary."),
    _ch("jomea_live", "UCdfM4J2vHsRkjJTndYudMVg", "noisy", "Ҷомеъа Live: diaspora live commentary."),
    _ch("sulaymoni_fors", "UCrjeb9IgUggXGqP68_sb3ig", "noisy", "Sulaymoni FORS: emotional opposition monologues."),
    _ch("rushdi_millat", "UCl2ddsg2vGcQSNYKbRTGPxg", "noisy", "РУШДИ МИЛЛАТ: migrant-focused commentary."),
    _ch("gayurov", "UCB07-cMoeNiSpYnVSf5LTGA", "noisy", "GAYUROV: work-abroad explainers + travel (some RU)."),
    _ch("talabsho_usa", "UCK6xYrrzDKSdpQoJe7F25vQ", "noisy", "Talabsho Mukimov: Tajiks-in-USA events/interviews."),
    _ch("sm_sharipov", "UC-EvNte5h0RtqVlsrJqJD1Q", "noisy", "SM Sharipov: colloquial true-crime narration, high volume."),
    _ch("hamnavo", "UCSz_lk2CpAedtc4Wnf__TBw", "noisy", "Hamnavo: Dushanbe street vlogs/news walks."),
    _ch("gap_nadorm", "UCWYmF-OgBfGFMdxlKPMoh8w", "noisy", "ГАП НАДОРМ: colloquial cars/life vlogs (some RU)."),
    _ch("taj_mama", "UCoE97TS7fekqgoxgyDFSP7g", "noisy", "Taj mama: family vlogs, northern colloquial female speech."),
    _ch("m_avzun", "UCwcsw18CCxoT0MVZRqR6sLg", "noisy", "M AVZUN: women's shopping/family vlogs, northern dialect (some UZ)."),
```

Secondary (verified Tajik, smaller or higher music/compilation risk — paste if more hours needed):

```python
    _ch("istgoh", "UCyxGtin6cDwiAcwxEyJ0cKA", "noisy", "Istgoh media: street/culture conversations (small)."),
    _ch("dukoni_khanda", "UCNslha8NiUhzLZ7iBsMpU0Q", "noisy", "Дукони ханда: Хандинкамон stage comedy, Khatlon troupe."),
    _ch("hazl_show", "UCvp9X9CIub6svHoEoIT2zJg", "noisy", "ҲазлShow: vine/skit compilations, street speech (music risk)."),
    _ch("mama_life", "UCJaudGh0l46CFhp19CcAW2Q", "noisy", "Mama life: cooking/shopping vlogs, female colloquial."),
    _ch("nurali_karim", "UCna9i61C18YgSgfKY1P6hCw", "noisy", "Nurali Karim: NYC travel vlogs in Tajik (small)."),
```

## What I could NOT find (saturated or empty niches)

- **Dedicated street-interview (vox-pop) channels.** Vox-pop exists only as segments inside Ozodi/Ozodagon (already in registry) and CM-1/ФАРАҲ (proposed). No standalone Tajik «пурсиш дар кӯча» channel surfaced; queries like «назарпурсӣ мардуми тоҷик» returned a single Ozodagon video.
- **Call-in shows.** No true phone-in format found; Радиои Нав studio shows and Суҳроб Одилиён's open lives are the nearest equivalents.
- **Women's advice/talk programs.** «барномаи занона маслиҳат тоҷикӣ» returned nothing; female speech is best sourced from the vlog channels above (Taj mama, M AVZUN, Mama life).
- **Badakhshan beyond one outlet.** Pamir-tagged channels are dominated by music/dance or Shughni-language speech (not Tajik); only Сежд Pamir Screen carries regular Tajik-language Badakhshan content. Pamiri-language speech is a real contamination risk there — the script gate won't catch it from audio alone, only from transcripts.
- **News/audiobooks/kids/lessons** — deliberately skipped: well covered by round 1.
- **Beauty, gaming, cars, travel** — re-confirmed Russian/Uzbek-dominated (round-1 finding holds); GAYUROV/ГАП НАДОРМ are the usable exceptions.
- Channels checked and rejected: Sooroosh Media (music concerts), Nodir (mostly song programs), ES TV HD (Zarafshon wedding singers), KOINOT LIVE + EmpowerU (Russian/English podcasts), DOCTORCOVID19TJK (Hoji Mirzo re-uploads), ПОРЧАИ СУХАН (religious clips with heavy сурud share), ҶУМҲУРИЯТ (government reportage — news niche already saturated).
