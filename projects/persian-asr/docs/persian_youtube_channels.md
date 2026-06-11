# Persian YouTube channels — vetted registry candidates (2026-06)

Research pass for the create pipeline (VAD -> Scribe ensemble -> verify). Priority was
CONVERSATIONAL speech — the production model's weak benchmark splits — so talk shows, interview
podcasts, call-in shows, street interviews, and vlogs dominate. Every channel below was verified
live with yt-dlp (`--flat-playlist`, titles + channel id + subscriber count; video counts capped
at 1200). Variant is tagged per channel: the model targets **Iranian** Persian; **Dari** is a
bonus tier. Pure-music channels and mostly-English channels were excluded during vetting.

Sizing notes: "videos" is the uploads-tab count (capped at 1200 — "1200+" means the cap was hit).
Hour estimates are rough (count x typical runtime from the first page of uploads).

## 1. Channel table

### Iranian — conversational (the priority)

| Name | Canonical URL | Variant | Content type | Est. size | Tier | Diversity rationale |
|---|---|---|---|---|---|---|
| Dr. Holakouee (رازها و نیازها) | youtube.com/channel/UCEUO9scRXBptMsPycZygmbg | Iranian | Call-in psychology show (Radio Hamrah, LA) | 1200+ vids, 50k+ callers over the years; ~10-60 min each | noisy | The single best spontaneous-speech source found: thousands of distinct unscripted callers (all ages/regions/phone audio) + expert monologue |
| Poletik — Kambiz Hosseini | youtube.com/channel/UCEt1wJTpIl4oKMEEuT1V3vw | Iranian | Weekly political interview show | 95 vids x ~1h | noisy | Long-form two-speaker spontaneous debate; diaspora + in-country guests |
| Tabaghe 16 (طبقه ۱۶) | youtube.com/@Tabaghe16 | Iranian | Joe-Rogan-style interview podcast (startups/tech/culture) | 202 vids x 1-2h, 60k subs | noisy | Modern Tehrani colloquial register, overlapping casual dialogue |
| King Raam — Masty o Rasty (مستی و راستی) | youtube.com/@KingRaam | Iranian | Uncensored interview podcast + vlogs | 40 vids x 1-3h | noisy | Very informal/taboo-free register absent from broadcast data |
| Football360 | youtube.com/channel/UCuSS7Q8O6Wv1SGQVAs1Uvgg | Iranian | Sports talk shows, debate, legend interviews (in-country) | 1181 vids, 240k subs | noisy | Excited overlapping sports argument — a register news data never covers |
| Channel One TV | youtube.com/channel/UCHhiGJozZgFKidRv5IL4c6g | Iranian | Diaspora satellite TV: nightly talk + viewer call-ins | 1200+ vids | noisy | Call-in spontaneity + older-generation diaspora speech |
| Dar Shahr (در شهر) | youtube.com/channel/UCh15eeQnv1zL5l4zBwKaVYg | Iranian | Street interviews / blind dates (in Iran) | 56 vids x 10-30 min | noisy | True vox-pop street audio from Iran; explicit/taboo topics (content warning, fine for ASR) |
| Sedaye Mardom (صدای مردم) | youtube.com/@Sedaye_mardom1 | Iranian | Street interviews | 82 vids, small (1k subs) | noisy | More street vox-pop; small but on-target |
| Golshid | youtube.com/channel/UChHzEpcNzipFcDAV9LezYww | Iranian | Migration interviews + live Q&A | 55 vids x ~1h | noisy | Two-speaker female-led conversational; live-stream audio |
| Mohammad Jorjandi | youtube.com/@jorjandi | Iranian | Cybercrime/scam exposes, commentary | 226 vids | noisy | Spontaneous single-speaker rants + phone evidence clips |
| Jadi Mirmirani | youtube.com/channel/UCgKePkWtPuF36bJy0n2cEMQ | Iranian | Tech/freedom monologues, live streams (Radio Geek) | 838 vids | noisy | Casual webcam monologue, heavy code-switching with tech loanwords |
| PUTAK (پوریا عرب) | youtube.com/channel/UCa20QQoV4gaLv9XRki-ynWQ | Iranian | Entertainment commentary/reaction | 1200+ vids, 1.16M subs | noisy | Youth slang register at scale |
| SoGang | youtube.com/channel/UCxym7GWKXPXYISgy7U-AOYw | Iranian | Vlogs/challenges/entertainment | 639 vids, 678k subs | noisy | Spontaneous multi-speaker youth banter |
| Mia Plays | youtube.com/@Miaplays | Iranian | Vlogs + gaming, couple banter | 379 vids, 804k subs | noisy | Female conversational speech; some English mixed in (language gate handles it) |
| Ali Sabouri | youtube.com/channel/UCbGE6dIy5S7RxovBbEGzwsg | Iranian | "کمدی نصف شب" comedy talk show | 391 vids | noisy | Comedy-show dialogue with audience |
| Hasan Reyvandi | youtube.com/@hassan_reyvandi | Iranian | Stand-up concert clips | 1200+ vids | noisy | On-stage spontaneous talk + crowd interaction (laughter/music intros — VAD/WER gate earns its keep) |
| Chef Javad Javadi | youtube.com/channel/UCqu8mtd9NBjgSaHUtHjRiLw | Iranian | Cooking vlogs (Australia) | 1200+ vids, 328k subs | noisy | Running cooking commentary, informal kitchen talk |
| mr taster | youtube.com/channel/UCAkoSMgQOmGyXYYg-HWnmEg | Iranian | Street-food travel vlogs | 784 vids | noisy | Street/restaurant ambience speech; CAUTION: recent uploads trend English — rely on per-clip language gate |
| Manoto TV | youtube.com/@manototv | Iranian | Variety/reality/talk archive (Befarmaeed Shaam, docs) | 1200+ vids, 953k subs | noisy | Reality-show dinner-table conversation; NOTE: new flagship shows are members-only, scrape = public archive |
| Zoomit | youtube.com/@ZoomitTV | Iranian | Tech news, reviews, studio debates | 1200+ vids | noisy | Studio two-speaker tech debate + reviews |
| MoboNews | youtube.com/channel/UCal5H0vgmsKcwMKiUflVpEw | Iranian | Phone/gadget reviews | 917 vids, 397k subs | clean | Semi-scripted studio voice-over, good mic quality |

### Iranian — narration & broadcast (clean fill)

| Name | Canonical URL | Variant | Content type | Est. size | Tier | Diversity rationale |
|---|---|---|---|---|---|---|
| Bplus Podcast | youtube.com/@BplusPodcast | Iranian | Book-summary narration (Ali Bandari) | 344 vids x 30-60 min, 279k subs | clean | Studio narration, careful diction — high-yield clean anchor |
| ChannelB Podcast | youtube.com/channel/UC4Ub2ZqIHcFC0X9q-QZln9A | Iranian | True-story documentary narration | 25 vids x 1-3h | clean | Same studio quality, long episodes |
| Rokh Podcast | youtube.com/@rokhpodcast | Iranian | History/biography narration | 257 vids, 159k subs | clean | Formal narration register |
| Movarekh Podcast | youtube.com/channel/UC36Lu8eWcnhmhfYrbMdQHNQ | Iranian | World-history narration | 288 vids, 217k subs | clean | More narration; distinct narrator voice |
| BBC Persian | youtube.com/channel/UCHZk9MrT3DGWmVqdsj5y0EA | Iranian | News + Pargar debate + radio shows | 1200+ vids, 1.19M subs | clean | Broadcast anchor speech; Pargar adds moderated multi-speaker debate |
| Iran International | youtube.com/@IRANINTL | Iranian | 24h news + talk segments | 1200+ vids, 1.13M subs | clean | Largest active Persian news firehose, many guest interviews |
| VOA Farsi | youtube.com/@VOAFarsi | Iranian | News + Safhe-ye Akhar archive | 1200+ vids | clean | Huge archive; NOTE: VOA shuttered March 2025, treat as static archive |
| Radio Farda | youtube.com/@radiofarda | Iranian | RFE/RL news + culture shows | 1200+ vids | clean | Radio-quality speech, mirrors the Tajik radio_ozodi pick |

### Afghan (Dari) — bonus tier

| Name | Canonical URL | Variant | Content type | Est. size | Tier | Diversity rationale |
|---|---|---|---|---|---|---|
| Bamdade Khosh | youtube.com/channel/UCqLaXPeh3VV3VyDROnbb4RA | Dari | TOLO morning talk show | 1200+ vids x 30-90 min | noisy | Flagship Dari live conversation (guests, phone-ins); some music/Quran segments |
| Afghan Daily | youtube.com/@AfghanDailyLife | Dari | Ariana TV street/social reports from Kabul | 1200+ vids, 193k subs | noisy | Dari street vox-pop — vendors, passers-by, markets |
| TOLOnews | youtube.com/channel/UCknKEREcKsqhB9DFvT61LZQ | mixed (Dari + Pashto) | News + Farakhabar/Mehwar debate shows | 1200+ vids, 1.31M subs | clean | Dari broadcast + panel debate; Pashto shows must be language-gated |
| TOLO TV | youtube.com/@ToloTv | mixed (Dari + Pashto) | Entertainment/series/shows | 1200+ vids, 1.35M subs | noisy | Dari entertainment talk; CAUTION: music-heavy shows (Hit Zarshomar etc.) — expect high discard |
| Afghanistan International | youtube.com/@afintltv | Dari | London-based 24h news + interviews | 1200+ vids, 565k subs | clean | Exile Dari broadcast, lots of one-on-one interviews |
| Amu TV | youtube.com/@AmuTelevision | Dari | Exile Afghan TV: news, talk, reports | 1200+ vids, 183k subs | clean | Post-2021 exile Dari journalism, varied correspondents |
| Ariana News | youtube.com/@ArianaNewsTV | mixed (Dari + Pashto + EN) | News bulletins/reports | 1200+ vids, 652k subs | clean | More Dari news; trilingual — language gate required |

Channels checked and REJECTED during vetting: Marz Radio (music remixes, not the Radio Marz
podcast — the podcast has no YouTube video channel), Salam Watandar's YouTube hits (movie-dump
channels), c/afghanvlogs (cracked-software spam), Rokshow (1 video), Tavaana (no videos tab,
livestream-only), HoomanTV / Max Amini (mostly English), 8am Media (no usable videos tab).

## 2. Ready-to-paste `_ch(...)` block (sources.py style)

```python
YOUTUBE_CHANNELS: tuple[Channel, ...] = (
    # --- Iranian: conversational talk / interview / call-in / street (the weak-split fix) -----
    _ch(
        "holakouee",
        "UCEUO9scRXBptMsPycZygmbg",
        "noisy",
        "Dr. Holakouee: call-in psychology show, thousands of unscripted callers.",
    ),
    _ch(
        "poletik",
        "UCEt1wJTpIl4oKMEEuT1V3vw",
        "noisy",
        "Poletik (Kambiz Hosseini): weekly political interviews.",
    ),
    _ch("tabaghe16", "@Tabaghe16", "noisy", "Tabaghe 16: long-form interview podcast."),
    _ch("king_raam", "@KingRaam", "noisy", "Masty o Rasty: uncensored interview podcast."),
    _ch(
        "football360",
        "UCuSS7Q8O6Wv1SGQVAs1Uvgg",
        "noisy",
        "Football360: sports talk/debate + legend interviews (in-country).",
    ),
    _ch(
        "channel_one",
        "UCHhiGJozZgFKidRv5IL4c6g",
        "noisy",
        "Channel One TV: diaspora nightly talk + viewer call-ins.",
    ),
    _ch(
        "dar_shahr",
        "UCh15eeQnv1zL5l4zBwKaVYg",
        "noisy",
        "Dar Shahr: street interviews/blind dates in Iran (explicit topics).",
    ),
    _ch("sedaye_mardom", "@Sedaye_mardom1", "noisy", "Sedaye Mardom: street interviews, small."),
    _ch(
        "golshid",
        "UChHzEpcNzipFcDAV9LezYww",
        "noisy",
        "Golshid: migration interviews + live Q&A.",
    ),
    _ch("jorjandi", "@jorjandi", "noisy", "Mohammad Jorjandi: scam exposes, spontaneous talk."),
    _ch(
        "jadi",
        "UCgKePkWtPuF36bJy0n2cEMQ",
        "noisy",
        "Jadi: tech/freedom monologues + livestreams (Radio Geek).",
    ),
    _ch("putak", "UCa20QQoV4gaLv9XRki-ynWQ", "noisy", "PUTAK: entertainment commentary, slang."),
    _ch("sogang", "UCxym7GWKXPXYISgy7U-AOYw", "noisy", "SoGang: vlogs/challenges, youth banter."),
    _ch("miaplays", "@Miaplays", "noisy", "Mia Plays: vlogs/gaming (some EN; gate handles it)."),
    _ch(
        "ali_sabouri",
        "UCbGE6dIy5S7RxovBbEGzwsg",
        "noisy",
        "Ali Sabouri: midnight-comedy talk show.",
    ),
    _ch(
        "reyvandi",
        "@hassan_reyvandi",
        "noisy",
        "Hasan Reyvandi: stand-up clips, crowd noise + music intros.",
    ),
    _ch(
        "javad_javadi",
        "UCqu8mtd9NBjgSaHUtHjRiLw",
        "noisy",
        "Chef Javad Javadi: cooking vlogs, running commentary.",
    ),
    _ch(
        "mr_taster",
        "UCAkoSMgQOmGyXYYg-HWnmEg",
        "noisy",
        "mr taster: street-food vlogs (recent uploads trend EN; gate decides).",
    ),
    _ch(
        "manototv",
        "@manototv",
        "noisy",
        "Manoto TV: reality/variety archive (new shows member-gated; public archive only).",
    ),
    _ch("zoomit", "@ZoomitTV", "noisy", "Zoomit: tech reviews + studio debates."),
    # --- Iranian: narration / broadcast (clean fill) ------------------------------------------
    _ch("mobonews", "UCal5H0vgmsKcwMKiUflVpEw", "clean", "MoboNews: studio gadget reviews."),
    _ch("bplus", "@BplusPodcast", "clean", "Bplus: studio book-summary narration."),
    _ch(
        "channelb",
        "UC4Ub2ZqIHcFC0X9q-QZln9A",
        "clean",
        "ChannelB: true-story documentary narration, 1-3h episodes.",
    ),
    _ch("rokh", "@rokhpodcast", "clean", "Rokh Podcast: history/biography narration."),
    _ch("movarekh", "UC36Lu8eWcnhmhfYrbMdQHNQ", "clean", "Movarekh: world-history narration."),
    _ch(
        "bbc_persian",
        "UCHZk9MrT3DGWmVqdsj5y0EA",
        "clean",
        "BBC Persian: news + Pargar debate + radio shows.",
    ),
    _ch("iranintl", "@IRANINTL", "clean", "Iran International: 24h news + talk segments."),
    _ch("voa_farsi", "@VOAFarsi", "clean", "VOA Farsi: news archive (static since 2025-03)."),
    _ch("radio_farda", "@radiofarda", "clean", "Radio Farda: RFE/RL news + culture shows."),
    # --- Dari (bonus tier; model targets Iranian Persian) -------------------------------------
    _ch(
        "bamdade_khosh",
        "UCqLaXPeh3VV3VyDROnbb4RA",
        "noisy",
        "Dari: Bamdade Khosh, TOLO morning talk (music/Quran segments).",
    ),
    _ch(
        "afghan_daily",
        "@AfghanDailyLife",
        "noisy",
        "Dari: Afghan Daily, Kabul street/social vox-pop reports.",
    ),
    _ch(
        "tolonews",
        "UCknKEREcKsqhB9DFvT61LZQ",
        "clean",
        "Dari+Pashto: TOLOnews bulletins + debate shows; gate drops Pashto.",
    ),
    _ch(
        "tolo_tv",
        "@ToloTv",
        "noisy",
        "Dari+Pashto: TOLO TV entertainment; music-heavy, expect high discard.",
    ),
    _ch(
        "afintl",
        "@afintltv",
        "clean",
        "Dari: Afghanistan International, exile news + interviews.",
    ),
    _ch("amu_tv", "@AmuTelevision", "clean", "Dari: Amu TV, exile news/talk/reports."),
    _ch(
        "ariana_news",
        "@ArianaNewsTV",
        "clean",
        "Dari+Pashto+EN: Ariana News bulletins; gate drops non-Dari.",
    ),
)
```

## 3. Persian-YouTube-specific risks

- **No geo-blocks on the channels themselves.** None of the 36 channels is region-locked for a
  non-Iranian scraper IP. The geo problem runs the other way: YouTube is blocked *inside* Iran,
  so in-country creators (Football360, Zoomit, MoboNews, Dar Shahr, Sedaye Mardom...) upload via
  VPN — upload schedules gap out during internet blackouts (e.g. the mid-2026 shutdown several
  verified titles reference).
- **Monetization squeeze -> member-gating.** YouTube's late-2025 ad-revenue change for
  Iran-VPN audiences cut creator income up to 80%; Manoto already moved new flagship shows
  (Befarmaeed Shaam LA) behind YouTube channel memberships. Public back-catalogs remain
  scrapeable, but expect more Iranian channels to gate or slow down. Scrape archives early.
- **In-country talk shows are platform-locked, not on YouTube.** The big celebrity talk shows
  (Shab Ahangi, Hamrefigh, Pishgoo, Dorehami) live on Filimo/Namava/Aparat — only pirated
  re-uploads exist on YouTube. They were deliberately excluded; don't be tempted by the
  full-episode mirror channels (unstable, takedown-prone).
- **Music mixes.** Persian entertainment loves music beds: Reyvandi concerts open with songs,
  TOLO TV is music-show heavy, Bamdade Khosh includes live music and Quran recitation. The
  VAD + WER gate handles it, but expect lower keep-rates than for the podcast channels.
- **Code-switching.** Bplus/ChannelB/Rokh title in English but narrate in Persian (fine).
  Mia Plays and mr taster genuinely mix English audio — per-clip language gating is load-bearing
  there, same as the Tajik bilingual channels.
- **Dari/Pashto mixing.** TOLOnews, TOLO TV and Ariana interleave Pashto (and some English)
  programs in the same uploads tab; there is no per-language playlist discipline. The language
  gate must distinguish Pashto (different language, easy) but ALSO tag Dari vs Iranian Persian
  (same script, same language ID for most detectors) — keep Dari channels in a separate ingest
  batch so the variants can be weighted or held out at export time.
- **Explicit content.** Dar Shahr (and some Sedaye Mardom) street interviews cover sex/taboo
  topics — irrelevant for ASR quality, but worth knowing before anyone audits clips.
- **Channel-ID hygiene.** Several channels' display handles differ from search results
  (`@DarShahrr` with double r, `@SoGang1`, `@dr.holakoueeofficialchannel`); the block above uses
  UC ids wherever the handle looked fragile.
