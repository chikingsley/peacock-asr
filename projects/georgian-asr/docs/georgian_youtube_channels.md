# Georgian (kat_Geor) YouTube channels — exhaustive, for the first Georgian scrape

Agent-researched 2026-06-13. Georgian only (Russian/other-language interleave flagged). 27 NEW channels beyond the ~18 already scaffolded in `sources.py`. Conversational-weighted — the v0 model (20.7 WER) has only 145h of scripted gold and badly needs real conversation. Handles resolved live.

Status 2026-06-25: the non-duplicate entries are wired into `sources.py`. The unresolved `@interpressnews` candidate was removed after `yt-dlp` returned 404. `audiobooks_geo_ena` is the same channel ID already present as `audiobooks_geo_ka`.

## Paste block for sources.py

```python
# -- news / TV broadcasters (new) ---------------------------------------------------------
_ch("maestro_tv", "@maestrotv3750", "noisy", "Maestro TV — news + analytical talk shows; large catalog."),
_ch("kavkasia_tv", "@KavkasiaTelevision", "noisy", "Telecompany Kavkasia — political talk shows, debates, analytics."),
_ch("postv", "@POSTV", "noisy", "POSTV — talk/analytical programs; pro-gov slant, Georgian-dominant."),
_ch("postv_analytics", "@POSTV.Analytics", "noisy", "POSTV analytics sub-channel — long-form panel discussion."),
_ch("publika_tv", "@publika_news", "noisy", "Publika — independent socio-political outlet; reports, interviews, street audio."),
_ch("batumelebi", "@BatumelebiNews", "noisy", "Batumelebi — independent Batumi/Adjara newsroom; reports, interviews."),
_ch("forbes_georgia", "@ForbesGeo", "clean", "Forbes Georgia — studio business interviews; small channel."),
_ch("tv_georgian_times", "@TVGeorgianTimes", "noisy", "TV Georgian Times — news + reporting."),
# -- talk / podcast / interview (new) -----------------------------------------------------
_ch("nanukas_channel", "@nanukaschannel6465", "noisy", "Nanuka Zhorzholiani's Show — flagship social/celebrity talk show, multi-guest."),
_ch("gattsu", "@Gattsu", "noisy", "Gattsu — geopolitics/culture commentary + interviews; spontaneous speech."),
_ch("octopus", "@Octopusi", "noisy", "Octopus — entertainment + sports talk shows, panel formats."),
_ch("studio_monitori", "@studiomonitori", "noisy", "Studio Monitor — investigative docs + on-camera interviews."),
_ch("tabula_tv", "@TabulaTelevision", "noisy", "Tabula — political analysis, interviews, longform talk."),
# -- comedy / entertainment / variety (new) ----------------------------------------------
_ch("comedy_group", "@c-comedy", "noisy", "Comedy Group (კომედი ჯგუფი) — sketches, talk shows, variety."),
_ch("komedi_arxi", "https://www.youtube.com/channel/UCMwRaK_tGpS8oCMOJk7s5Hg", "noisy", "Komedi Arxi — comedy sketches/shows; small re-upload, verify size."),
_ch("nichieri", "@nichieri", "noisy", "Nichieri — Georgia's Got Talent; live banter, judge/host speech."),
_ch("colis_dakalebi", "@ColisDakalebiTELEGE", "noisy", "'Chemi Tsolis Dakalebi' sitcom — scripted conversational dialogue."),
_ch("hungryman", "@hungrymantv", "noisy", "Hungryman — food challenges, pranks, street social experiments."),
# -- vlogs / lifestyle (new) --------------------------------------------------------------
_ch("soso_around_world", "@SosoAroundTheWorld", "noisy", "Soso Nebieridze — top Georgian travel vlogger; narration + field audio."),
# -- educational / children (new) ---------------------------------------------------------
_ch("eduwoes", "@Eduwoes", "clean", "Eduwoes — educational explainers, mostly single-speaker narration."),
_ch("emili_tv", "@EmiliTV", "noisy", "Emili TV — father/daughter edutainment for kids/teens, conversational."),
_ch("jirafi_joze", "https://www.youtube.com/channel/UC2VLd27TDiI8ZTpVg2wdL7Q", "clean", "'Giraffe Jose' — children's educational narration, clean single-speaker."),
# -- religious (new) ----------------------------------------------------------------------
_ch("patriarchate_georgia", "@patriarchateofgeorgia797", "clean", "Patriarchate of Georgia — sermons, liturgy readings; some chant."),
# -- audiobooks / literature (new) --------------------------------------------------------
_ch("nplg_official", "@NPLGofficial", "clean", "National Parliamentary Library — audiobook readings by actors/writers."),
_ch("audiobooks_geo_ena", "https://www.youtube.com/channel/UCUuogpIEDKdE17EQ45diGsg", "clean", "literature narration, clean single-speaker (tiny but gold)."),
# -- culture / niche (new) ----------------------------------------------------------------
_ch("voices_ancestors", "@voicesoftheancestors", "clean", "polyphony podcast — partly English, language gate will filter."),
```

## Risks

- **Russian/other interleave:** publika_tv, batumelebi, kavkasia_tv and most news carry occasional Russian-speaking guests; voices_ancestors is partly English. Rely on the per-clip language gate.
- **Tiny/low-yield (clean gold):** audiobooks_geo_ena (22 subs), nplg_official (485), forbes_georgia.
- **Music-heavy / skip if low VAD yield:** georgian_voices_official, voices_ancestors.
- **Already in sources.py (excluded):** GPB, adjara_tv, radio_tavisupleba, imedi, rustavi2, mtavari, formula, tv_pirveli, palitra_news, netgazeti, audiobooks_geo_ka, geobooks_audio, komedi_shou, etc.
