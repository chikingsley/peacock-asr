# Dari YouTube category pilot — findings (2026-06-17)

One-off enrichment pilot that fetched recent video titles per channel (`yt-dlp --flat-playlist`) and asked the LLM for a channel-level genre plus a per-video genre, to measure (a) which channel handles are dead, (b) how leaky channel-level genre is, and (c) whether the human clean/noisy prior matches the LLM read. The pilot itself was scratch (regenerable from `sources.py`); its conclusions live here, and the dead-channel fixes already landed in `src/dari_asr/sources.py`. The taxonomy/prescan design it fed is now folded into the repo-root `TODO.md` ("YouTube source quality" section).

## Dead / empty handles (no videos resolved)

`@`-handle was wrong; resolved to a working channel-id / `/c/` / `/user/` URL in `sources.py`:

- tamadon_tv, hasht_e_subh, etilaatroz, voa_dari, azadi_radio, begum_tv, rukhshana_media

Still unresolved (left as `@FarhadDaryaOfficial`): **farhad_darya** — music-only channel, low spoken-ASR value; revisit only if a working URL is found.

## Channel category distribution (19 live channels)

news_bulletin 7 · vlog_lifestyle 6 · panel_talkshow 2 · podcast 2 · documentary 1 · educational_explainer 1

## Tier flips (LLM disagrees with the human clean/noisy prior)

- amu_tv: noisy → clean (news_bulletin)
- negaah_tv: noisy → clean (news_bulletin)
- ahmad_shahram_wafaee: noisy → clean (educational_explainer)

## High channel→video genre leak (>=30% of videos off the channel genre)

- afghan_voice_radio podcast 100%
- rah_e_farda podcast 95%
- ariana_television vlog_lifestyle 35%
- daricha_watan vlog_lifestyle 35%
- ariana_news news_bulletin 33%
- ebrahim_danish documentary 30%

Takeaway: channel-level genre is a weak prior for several channels; the curator prescan should score genre per-video, not per-channel.
