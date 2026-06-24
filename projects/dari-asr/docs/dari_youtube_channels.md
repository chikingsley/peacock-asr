# Dari (Afghan Persian, prs) YouTube channels — seeds a separate Dari ASR project

Agent-researched 2026-06-13. **Dari / Afghan Persian only** — Iranian Persian, Tajik, and
Pashto-dominant channels excluded. Kept SEPARATE from Iranian Persian (different project) because
mixing varieties dilutes the model. 28 channels. Many Afghan outlets are bilingual Dari+Pashto —
**language-filter at ingest** (the curator's language gate handles this; flag the bilingual ones).

## Paste block (for a future projects/dari-asr sources.py)

```python
# Category 1 — national news & current affairs (TV, Dari)
_ch("tolonews", "@tolonews", "clean", "Afghanistan's first 24h news network; Dari anchor bulletins (Pashto interleave — filter)."),
_ch("tolonews_talkshows", "@tolonewstalkshows", "noisy", "TOLOnews talk-show feed (Farakhabar/Mehwar); multi-speaker Dari debate."),
_ch("ariana_news", "@ArianaNewsTV", "clean", "Largest Afghan news network; studio Dari bulletins (segment-filter Pashto)."),
_ch("ariana_television", "@Ariana.Television", "noisy", "ATN flagship; magazine/social-report shows, mostly Dari."),
_ch("one_tv_kabul", "@1TVKabul", "clean", "1TV Afghanistan, Kabul; predominantly Dari news + Cactus+ interviews."),
_ch("afghanistan_intl", "@afintltv", "clean", "London/DC; launched Persian-Dari only, news + analysis (filter post-2023 Pashto)."),
_ch("amu_tv", "@AmuTelevision", "noisy", "Virginia-based independent; Dari news + Mawj interviews (some Pashto/diaspora)."),
_ch("khurshid_tv", "@KhurshidTV", "noisy", "Kabul Dari-language general entertainment; talk + magazine."),
_ch("tamadon_tv", "@TamadonTV", "noisy", "Kabul Dari channel, Hazara-leaning; religious + current affairs."),
_ch("negaah_tv", "@NegaahTV", "noisy", "Kabul Dari-language channel; talk + entertainment."),
_ch("rah_e_farda", "@RaheFardaTV", "noisy", "Kabul Dari-Persian radio/TV; talk + call-in, Hazara-Dari register."),
# Category 2 — independent / diaspora exile media (Dari)
_ch("hasht_e_subh", "@HashteSubhDaily", "clean", "8AM Media, leading independent daily in exile; Dari reports + analysis."),
_ch("etilaatroz", "@Etilaatroz", "clean", "Investigative outlet (Zaki Daryabi); strongly Dari news + explainers."),
_ch("voa_dari", "@voaafghanistandari", "clean", "VOA Dari service; news, interviews, documentary reports (Dari-dedicated)."),
_ch("rta_dari", "@RTADari", "clean", "State broadcaster's Dari channel; clean anchor read (split from RTA Pashto)."),
_ch("azadi_radio", "@Azadiradio", "clean", "RFE/RL Radio Azadi; Dari + Pashto mixed — segment-filter for Dari."),
_ch("begum_tv", "@BegumTelevision", "noisy", "Paris women's channel; educational + talk, primarily Dari (diaspora accent)."),
_ch("rukhshana_media", "@RukhshanaMedia", "noisy", "Women-focused (Zahra Joya); Dari interviews/testimony (+English — filter)."),
# Category 3 — talk / podcast / interview (conversational Dari)
_ch("afghan_voice_radio", "@afghanvoiceradio", "noisy", "London Afghan community radio; Dari interviews/podcasts (diaspora register)."),
# Category 4 — vlogs, travel, documentary, daily-life (conversational Dari)
_ch("ebrahim_danish", "@EbrahimDanish", "noisy", "Village life/travel/food across Afghanistan; Dari narration + street talk (filter Pashto regions)."),
_ch("daricha_watan", "@DarichaWatan", "noisy", "Dareecha Watan; Afghan documentary/vlog, predominantly Dari field audio."),
_ch("ahmad_shahram_wafaee", "@ahmadshahramwafaee", "noisy", "Lecturer/researcher; Islamic + documentary series, Dari narration."),
_ch("sunrise_afghanistan", "@SunriseAfghanistan", "noisy", "Culture + travel community channel; Dari-dominant vlog/field audio."),
_ch("afghan_vlogs", "@afghanvlogs", "noisy", "Daily-life vlogging in Dari (some diaspora-accent creators)."),
_ch("afghan_daily_life", "@AfghanDailyLife", "noisy", "Urban/rural daily-life reports in Dari."),
# Category 5 — music, culture & personalities (spoken segments only)
_ch("farhad_darya", "@FarhadDaryaOfficial", "noisy", "Afghan artist; Dari interviews/spoken intros — keep talk, drop sung audio."),
_ch("afghan_comedy", "@AfghanComedyOfficial", "noisy", "Dari comedy skits; scripted dialogue (verify each series isn't Pashto)."),
```

## Deny-list (do NOT add — Pashto-dominant)

`@lemartelevision` (LemarTV), Shamshad TV (~85% Pashto), Zhwandoon.

## Risks

- **Pashto interleave (biggest):** TOLOnews, Ariana, Amu, Afghanistan International (post-2023),
  Azadi, and field vlogs carry Pashto on the same channel — language-filter, don't ingest blind.
  Prefer the language-split feeds (`@RTADari`, `@voaafghanistandari`, `@tolonewstalkshows`).
- **Handle spelling unverified** for several (resolve by `/channel/UC…` ID at ingest).
- **Diaspora accent drift** (Amu, Afghanistan Intl, Begum, Rukhshana) — tag in-country vs diaspora.
- **Dialect spread** (Kabuli, Hazaragi, Herati, diaspora) — a feature; tag for balanced sampling.
