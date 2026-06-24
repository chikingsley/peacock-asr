# Iranian Persian (fas_Arab / fa_ir) YouTube channels — exhaustive, language-pure

Agent-researched 2026-06-13. **Iranian Persian only** — no Dari, no Tajik, no Arabic-primary.
74 channels, ~40 conversational ("noisy"). Handles verified to resolve (HTTP 200 vs 404 control).
Supersedes the earlier mixed list. Dari moved to its own project (`projects/dari-asr`).

## Paste block for sources.py

```python
# --- News: diaspora / international Persian networks ---
_ch("iran_international", "@IRANINTL", "noisy", "24/7 Persian news; anchor reads plus heavy live call-in, panel debate, field reports."),
_ch("manoto", "@manototv", "clean", "London-based Persian general/news network; studio bulletins and documentaries."),
_ch("bbc_persian", "@BBCNewsPersian", "clean", "BBC Persian service; studio anchors, interviews, analysis in standard Farsi."),
_ch("voa_farsi", "@VOAFarsi", "clean", "Voice of America Persian (صدای آمریکا); news bulletins and reports."),
_ch("voa_persian_ofogh", "@voacapitol", "noisy", "VOA Persian 'Ofogh' talk/panel show; multi-guest debate and call-in."),
_ch("radio_farda", "@radiofarda", "clean", "RFE/RL Persian; news reads, explainers, interviews."),
_ch("dw_persian", "@dwpersian", "clean", "Deutsche Welle Persian (دویچه وله فارسی); news and reportage."),
_ch("euronews_persian", "@euronewspe", "clean", "Euronews in Persian; voiced-over world-news bulletins."),
_ch("independent_persian", "@IranIndependentTV", "noisy", "Independent Persian / Iran Independent TV; interviews, panels, talk shows."),
_ch("iranwire", "@iranwire", "noisy", "IranWire citizen journalism; field reports, interviews, mixed audio."),
_ch("kayhan_london", "@KayhanLondonOnline", "clean", "Kayhan London online; news segments and interviews."),
_ch("radio_zamaneh", "@Radiozamanehplus", "noisy", "Radio Zamaneh; interviews, discussion, reportage."),
_ch("iran_tv_network", "@IranTVNetwork", "noisy", "LA-based Persian satellite network (ITN); interviews and talk programming."),
# --- News: domestic Iranian outlets (geo-restricted — may need Iran VPN) ---
_ch("isna", "@ISNAVideo", "noisy", "Iranian Students' News Agency (ایسنا); studio talks, interviews, field clips."),
_ch("khabar_online", "@KHABARONLINIR", "noisy", "Khabar Online (خبر آنلاین); interviews and panel discussion."),
_ch("entekhab", "@EntekhabNewsSite", "noisy", "Entekhab (انتخاب); video interviews and debates."),
_ch("rouydad24", "@rouydad2443", "noisy", "Rouydad24 (رویداد۲۴); news analysis and interviews."),
_ch("etemad_online", "@etemadonline", "noisy", "Etemad Online (اعتماد), reformist daily; interviews and discussion."),
_ch("irna", "@IRNA_en", "clean", "IRNA (ایرنا) state news agency; Persian news (handle suffixed _en, content is Persian)."),
_ch("didban_iran", "@didbaniran-newsagency", "noisy", "Didban Iran (دیده‌بان ایران); investigative interviews and reports."),
_ch("hammihan", "@hammihanonline", "noisy", "Ham-Mihan (هم‌میهن) reformist daily; interviews and reports."),
_ch("fars_news", "@farsna", "noisy", "Fars News (فارس) video desk; reports and interviews."),
# --- Podcasts (conversational) ---
_ch("channelb_podcast", "@channelbpodcast", "noisy", "ChannelB (چنل‌بی), long-form Persian narrative/documentary podcast."),
_ch("bplus_podcast", "@bpluspodcast", "noisy", "BPlus (بی‌پلاس), Persian non-fiction book-summary / ideas podcast."),
_ch("radio_cafe", "@radiocafee", "noisy", "Radio Cafe (رادیو کافه); conversational podcast (note double-e handle)."),
_ch("radio_marz", "@radiomarz", "noisy", "Radio Marz (رادیو مرز); narrative/conversational social-topics podcast."),
_ch("soma_podcast", "@somapodcast", "noisy", "Soma (سُما); conversational podcast on psychology and mental health."),
_ch("aknon_podcast", "@aknonpodcast", "noisy", "Aknon (اکنون); interview podcast on people's work and life stories."),
_ch("rokh_podcast", "@rokhpodcast", "noisy", "Rokh (رخ); biographical/historical podcast on notable figures."),
_ch("bedahe_podcast", "@bedahepodcast", "noisy", "Bedahe (بداهه); unscripted two-host conversational podcast."),
_ch("chai_plus", "@ChaiPlus", "noisy", "Chai Plus (چای پلاس); conversational/interview podcast."),
_ch("everything_podcast", "@everythingpodcast", "noisy", "Everything / Hamechiz (همه‌چیز); wide-topic conversational podcast."),
_ch("raft_podcast", "@raftpodcast", "noisy", "Raft (رفت); conversational podcast."),
_ch("movarekh_podcast", "@movarekhpodcast", "noisy", "Movarekh (مورخ); Persian history podcast."),
_ch("ravikade", "@ravikade", "clean", "Ravikade (راوی‌کده); single-host narrative-history podcast."),
_ch("dialogue_box", "@DialogueBoxMedia", "noisy", "Dialogue Box; cinema/literature/music conversation podcast."),
_ch("jabe_siah", "@jabesiah", "noisy", "Jabe Siah (جعبه سیاه); arts/culture review podcast."),
_ch("dastan_podcast", "@dastanpodcast", "clean", "Dastan (داستان); single-narrator storytelling podcast."),
# --- Talk shows / interviews / debate ---
_ch("dorehami", "@Dorehami", "noisy", "Dorehami (دورهمی), Mehran Modiri's comedy talk show with celebrity interviews."),
_ch("khandevaneh", "@khandevane", "noisy", "Khandevaneh (خندوانه), Iranian comedy/variety show: stand-up, music, interviews."),
_ch("hamrafigh", "@Hamrafigh", "noisy", "Hamrafigh (همرفیق), Shahab Hosseini's long-form one-on-one artist interviews."),
_ch("iran_talk", "@irantalks", "noisy", "IranTalk (ایران‌تاک), Mohammad Fazeli's weekly long-form interview show."),
_ch("studio_paat", "@Studio_patt", "noisy", "Studio Paat; political/economic/sports interviews and debates."),
_ch("football_360", "@football_360", "noisy", "Football 360, Adel Ferdowsipour's football talk show: analysis, interviews, panel."),
_ch("cafe_khabar", "@cafekhabarofficial", "noisy", "Cafe Khabar; Tehran news/interview channel with long conversational segments."),
_ch("ecoiran", "@ecoiran", "noisy", "EcoIran; economics outlet, studio interviews and expert discussion."),
_ch("didar_news", "@didarnews", "noisy", "Didar News; independent outlet, interviews and political conversation."),
# --- Street interviews / vlogs / daily life ---
_ch("iranopedia", "@iranopedia", "noisy", "Street-interview / vox-pop filmed in Iran; spontaneous conversational Persian."),
_ch("iran_village_cooking", "@Iran.village.Cooking", "noisy", "Northern-Iran village cooking/daily-life vlog; conversational Farsi."),
_ch("iran_village_life", "@Iran_village_life", "noisy", "Northern-Iran village cooking vlog with family; conversational Persian."),
_ch("villagehouse_golbanoo", "@Villagehouse_golbanoo", "noisy", "Kurdistan-Iran rural-life cooking vlog; conversational Persian."),
# --- Audiobooks (clean, single-narrator) ---
_ch("avas_book_club", "@AvasBookClub", "clean", "باشگاه کتاب آوا; Persian narrated audiobooks."),
_ch("ketabe_goya", "@ketabegoya8305", "clean", "کتاب گویا; Persian audiobook channel promoting reading culture."),
_ch("taaghche_audiobook", "@taaghchee-bookandaudiobook1279", "clean", "Taaghche (طاقچه) e-book/audiobook store, major Iranian platform."),
_ch("ketabsoti", "@ketabsoti", "clean", "کتاب صوتی; Persian audiobooks."),
_ch("navar_media", "@navarmedia", "clean", "Navar (نوار); Iranian audiobook/podcast platform outlet."),
_ch("mashaledanesh_audiobooks", "channel/UCG-NudOQ-LqfPfWhUR2DE4Q", "clean", "کتاب های گویای مشعل; single-narrator Persian literary audiobooks."),
_ch("persian_audio_books", "channel/UCZJFid4Yt-WQiuukRC72wbw", "clean", "Persian Audio Books; Farsi classics to modern literature."),
_ch("donyaye_ketab_soti", "channel/UCfdc4HIplyS66VGURD7BjlA", "clean", "دنیای کتاب صوتی; Persian audiobooks and storytelling."),
_ch("iranseda_ketab_soti", "channel/UCczbnOdlq_GZAzYp2Ll1hWg", "clean", "ایران صدا; Iranian state radio audiobook/گویا channel."),
# --- Educational / lectures (clean) ---
_ch("roshd_channel", "@ROSHDCHANNEL", "clean", "Persian children's stories, read-aloud books, educational content."),
_ch("maktabkhooneh_yt", "@MaktabkhoonehYT", "clean", "Maktabkhooneh course/lecture clips."),
_ch("faradars_free", "@Faradars_Free", "clean", "FaraDars (فرادرس) free-tier lecture content."),
_ch("faradars", "@faradars1", "clean", "FaraDars main; Iran's largest online-ed platform, technical/academic lectures."),
_ch("alaa_sharif", "@SanatiSharifIr", "clean", "آلاء (Alaa); free konkur/exam-prep lectures, single instructor."),
_ch("khan_academy_farsi", "@KhanAcademyFarsi", "clean", "Khan Academy Farsi; Persian-dubbed educational lessons."),
_ch("maktabkhooneh", "channel/UCFwPa1mBpYCUmWUnvp03LAw", "clean", "Maktabkhooneh (مکتب‌خونه), Iran's largest MOOC; university lectures."),
_ch("iran_academia", "channel/UCRLzQ330Tkize2MvnAfauaQ", "clean", "Iran Academia; Persian academic lectures in social sciences/humanities."),
# --- Documentary / culture ---
_ch("iran_documentary", "@IranDocumentary", "clean", "مستند ایران; Persian documentary narration (screen for music beds)."),
_ch("avaye_buf", "@avayebuf", "noisy", "Avaye Buf (آوای بوف); Persian audiobook and cultural-discussion channel."),
# --- Religious (Persian-language lectures) ---
_ch("raefipour", "@raefipour_official", "clean", "علی‌اکبر رائفی‌پور; official Masaf channel, Persian religious/ideological lectures."),
_ch("panahian", "@PanahianFarsi", "clean", "علیرضا پناهیان; Persian religious lectures (سخنرانی), single speaker."),
_ch("qaraati_quran", "channel/UC5ilho5Q-YSIn3dn1FTcWNw", "clean", "درسهایی از قرآن; Mohsen Qaraati's Persian Quran-lesson/tafsir."),
```

## Risks

- **Geo-restriction:** most domestic outlets (ISNA, Khabar Online, Entekhab, Etemad, IRNA, Hammihan,
  Fars, Didban) and Iranian platforms (Maktabkhooneh, FaraDars, Taaghche, IranSeda) may be geo-fenced
  from non-Iran IPs — test with the cookie/VPN setup before bulk pull.
- **IRIB state TV** (Shabake Khabar, Navad) terminated on YouTube under US sanctions — Khandevaneh/
  Dorehami survive via official re-up channels (verified).
- **Handle traps:** Radio Cafe is `@radiocafee` (double-e); Maktabkhooneh/FaraDars main + several
  audiobook channels are channel-ID only (no @handle).
- **Excluded for purity:** National Geographic Farsi (Iran+Afghan+Tajik remit), Café Bagheri
  (English-narrated), Gol Bezan (bilingual EN+FA).
- **Lower-confidence (re-verify on first pull):** @kojam, @dictionpod, @metavision, @navarmedia,
  @ketabsoti — resolve but content not page-confirmed (YouTube consent wall from datacenter IP).
