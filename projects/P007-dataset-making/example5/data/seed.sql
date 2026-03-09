-- SpeechLab Seed Data
-- All data previously hardcoded across component files

-- ─── Corpus ───────────────────────────────────────────────────────────────────

INSERT INTO corpus (id, file, audio_size, total_length, transcript) VALUES
  (1,  '029810234.wav', '0.11M', '3.62s', 'whatever the difficulty there''s a solution'),
  (2,  '029810235.wav', '0.09M', '2.91s', 'she sells seashells by the seashore'),
  (3,  '029810236.wav', '0.13M', '4.20s', 'the quick brown fox jumps over the lazy dog'),
  (4,  '029810237.wav', '0.08M', '2.55s', 'how much wood would a woodchuck chuck'),
  (5,  '029810238.wav', '0.07M', '2.10s', 'to be or not to be that is the question'),
  (6,  '029810239.wav', '0.06M', '1.88s', 'i can''t believe it''s already friday'),
  (7,  '029810240.wav', '0.10M', '3.15s', 'the weather forecast calls for heavy rain tomorrow'),
  (8,  '029810241.wav', '0.05M', '1.65s', 'please turn off the lights when you leave'),
  (9,  '029810242.wav', '0.08M', '2.80s', 'artificial intelligence is transforming our world'),
  (10, '029810243.wav', '0.04M', '1.30s', 'good morning how are you doing today'),
  (11, '029810244.wav', '0.12M', '3.95s', 'the children were playing in the park all afternoon'),
  (12, '029810245.wav', '0.06M', '2.00s', 'can you repeat that more slowly please'),
  (13, '029810246.wav', '0.09M', '3.05s', 'we need to submit the report by end of day'),
  (14, '029810247.wav', '0.07M', '2.44s', 'three thousand three hundred and thirty three'),
  (15, '029810248.wav', '0.11M', '3.70s', 'she has been working at the company for fifteen years'),
  (16, '029810249.wav', '0.05M', '1.55s', 'what time does the next train arrive'),
  (17, '029810250.wav', '0.14M', '4.60s', 'the pronunciation of this particular word is very challenging'),
  (18, '029810251.wav', '0.06M', '2.20s', 'the meeting has been rescheduled to next monday'),
  (19, '029810252.wav', '0.10M', '3.40s', 'i would like to order a large coffee with no sugar'),
  (20, '029810253.wav', '0.08M', '2.70s', 'the results were far below our expectations this quarter');

-- corpus_words: corpus_id, position, word, phonetic, phoneme_score, phoneme_accuracy, stress_accuracy

INSERT INTO corpus_words (corpus_id, position, word, phonetic, phoneme_score, phoneme_accuracy, stress_accuracy) VALUES
  -- 1: whatever the difficulty there's a solution
  (1,0,'whatever','wɒt''evə',1,10,10),(1,1,'the','ðə',0,3,10),(1,2,'difficulty','''dɪfɪkəlti',1,10,10),
  (1,3,'there''s','ðɛəz',1,10,10),(1,4,'a','ə',1,10,10),(1,5,'solution','sə''luːʃn',1,10,10),
  -- 2: she sells seashells by the seashore
  (2,0,'she','ʃiː',1,10,10),(2,1,'sells','sɛlz',1,9,10),(2,2,'seashells','''siːʃɛlz',0,5,5),
  (2,3,'by','baɪ',1,10,10),(2,4,'the','ðə',1,10,10),(2,5,'seashore','''siːʃɔː',0,4,5),
  -- 3: the quick brown fox jumps over the lazy dog
  (3,0,'the','ðə',1,10,10),(3,1,'quick','kwɪk',1,10,10),(3,2,'brown','braʊn',1,9,10),
  (3,3,'fox','fɒks',1,10,10),(3,4,'jumps','dʒʌmps',1,8,10),(3,5,'over','''əʊvə',1,10,10),
  (3,6,'the','ðə',1,10,10),(3,7,'lazy','''leɪzi',1,10,10),(3,8,'dog','dɒɡ',1,10,10),
  -- 4: how much wood would a woodchuck chuck
  (4,0,'how','haʊ',1,10,10),(4,1,'much','mʌtʃ',1,10,10),(4,2,'wood','wʊd',1,10,10),
  (4,3,'would','wʊd',0,6,10),(4,4,'a','ə',1,10,10),(4,5,'woodchuck','''wʊdtʃʌk',0,4,5),
  (4,6,'chuck','tʃʌk',1,8,10),
  -- 5: to be or not to be that is the question
  (5,0,'to','tuː',1,10,10),(5,1,'be','biː',1,10,10),(5,2,'or','ɔː',1,10,10),
  (5,3,'not','nɒt',1,10,10),(5,4,'to','tuː',1,10,10),(5,5,'be','biː',1,10,10),
  (5,6,'that','ðæt',1,9,10),(5,7,'is','ɪz',1,10,10),(5,8,'the','ðə',1,10,10),
  (5,9,'question','''kwɛstʃən',1,8,8),
  -- 6: i can't believe it's already friday
  (6,0,'i','aɪ',1,10,10),(6,1,'can''t','kɑːnt',1,9,10),(6,2,'believe','bɪ''liːv',1,10,10),
  (6,3,'it''s','ɪts',1,10,10),(6,4,'already','ɔːl''rɛdi',0,5,5),(6,5,'friday','''fraɪdeɪ',1,10,10),
  -- 7: the weather forecast calls for heavy rain tomorrow
  (7,0,'the','ðə',1,10,10),(7,1,'weather','''wɛðə',1,9,10),(7,2,'forecast','''fɔːkɑːst',0,6,5),
  (7,3,'calls','kɔːlz',1,10,10),(7,4,'for','fɔː',1,10,10),(7,5,'heavy','''hɛvi',1,9,10),
  (7,6,'rain','reɪn',1,10,10),(7,7,'tomorrow','tə''mɒrəʊ',0,4,5),
  -- 8: please turn off the lights when you leave
  (8,0,'please','pliːz',1,10,10),(8,1,'turn','tɜːn',1,10,10),(8,2,'off','ɒf',1,10,10),
  (8,3,'the','ðə',1,10,10),(8,4,'lights','laɪts',1,9,10),(8,5,'when','wɛn',1,10,10),
  (8,6,'you','juː',1,10,10),(8,7,'leave','liːv',1,10,10),
  -- 9: artificial intelligence is transforming our world
  (9,0,'artificial','ˌɑːtɪ''fɪʃl',0,5,5),(9,1,'intelligence','ɪn''tɛlɪdʒəns',1,8,10),
  (9,2,'is','ɪz',1,10,10),(9,3,'transforming','træns''fɔːmɪŋ',0,6,5),
  (9,4,'our','aʊə',1,9,10),(9,5,'world','wɜːld',1,10,10),
  -- 10: good morning how are you doing today
  (10,0,'good','ɡʊd',1,10,10),(10,1,'morning','''mɔːnɪŋ',1,10,10),(10,2,'how','haʊ',1,10,10),
  (10,3,'are','ɑː',1,10,10),(10,4,'you','juː',1,10,10),(10,5,'doing','''duːɪŋ',1,10,10),
  (10,6,'today','tə''deɪ',1,9,10),
  -- 11: the children were playing in the park all afternoon
  (11,0,'the','ðə',1,10,10),(11,1,'children','''tʃɪldrən',0,5,10),(11,2,'were','wɜː',1,9,10),
  (11,3,'playing','''pleɪɪŋ',1,10,10),(11,4,'in','ɪn',1,10,10),(11,5,'the','ðə',1,10,10),
  (11,6,'park','pɑːk',1,10,10),(11,7,'all','ɔːl',1,10,10),(11,8,'afternoon','ˌɑːftə''nuːn',0,6,5),
  -- 12: can you repeat that more slowly please
  (12,0,'can','kæn',1,10,10),(12,1,'you','juː',1,10,10),(12,2,'repeat','rɪ''piːt',1,8,10),
  (12,3,'that','ðæt',1,10,10),(12,4,'more','mɔː',1,10,10),(12,5,'slowly','''sləʊli',0,5,5),
  (12,6,'please','pliːz',1,10,10),
  -- 13: we need to submit the report by end of day
  (13,0,'we','wiː',1,10,10),(13,1,'need','niːd',1,10,10),(13,2,'to','tuː',1,10,10),
  (13,3,'submit','səb''mɪt',0,6,5),(13,4,'the','ðə',1,10,10),(13,5,'report','rɪ''pɔːt',1,9,10),
  (13,6,'by','baɪ',1,10,10),(13,7,'end','ɛnd',1,10,10),(13,8,'of','ɒv',1,10,10),(13,9,'day','deɪ',1,10,10),
  -- 14: three thousand three hundred and thirty three
  (14,0,'three','θriː',0,4,10),(14,1,'thousand','''θaʊzənd',0,5,5),(14,2,'three','θriː',0,4,10),
  (14,3,'hundred','''hʌndrəd',1,9,10),(14,4,'and','ænd',1,10,10),(14,5,'thirty','''θɜːti',0,5,5),
  (14,6,'three','θriː',0,4,10),
  -- 15: she has been working at the company for fifteen years
  (15,0,'she','ʃiː',1,10,10),(15,1,'has','hæz',1,10,10),(15,2,'been','biːn',1,10,10),
  (15,3,'working','''wɜːkɪŋ',1,9,10),(15,4,'at','æt',1,10,10),(15,5,'the','ðə',1,10,10),
  (15,6,'company','''kʌmpəni',1,9,10),(15,7,'for','fɔː',1,10,10),(15,8,'fifteen','fɪf''tiːn',0,6,5),
  (15,9,'years','jɪəz',1,10,10),
  -- 16: what time does the next train arrive
  (16,0,'what','wɒt',1,10,10),(16,1,'time','taɪm',1,10,10),(16,2,'does','dʌz',1,9,10),
  (16,3,'the','ðə',1,10,10),(16,4,'next','nɛkst',1,10,10),(16,5,'train','treɪn',1,10,10),
  (16,6,'arrive','ə''raɪv',0,5,5),
  -- 17: the pronunciation of this particular word is very challenging
  (17,0,'the','ðə',1,10,10),(17,1,'pronunciation','prəˌnʌnsi''eɪʃn',0,3,5),
  (17,2,'of','ɒv',1,10,10),(17,3,'this','ðɪs',1,10,10),(17,4,'particular','pə''tɪkjʊlə',0,5,5),
  (17,5,'word','wɜːd',1,10,10),(17,6,'is','ɪz',1,10,10),(17,7,'very','''vɛri',1,9,10),
  (17,8,'challenging','''tʃælɪndʒɪŋ',0,4,5),
  -- 18: the meeting has been rescheduled to next monday
  (18,0,'the','ðə',1,10,10),(18,1,'meeting','''miːtɪŋ',1,10,10),(18,2,'has','hæz',1,10,10),
  (18,3,'been','biːn',1,10,10),(18,4,'rescheduled','riː''ʃɛdjuːld',0,5,5),
  (18,5,'to','tuː',1,10,10),(18,6,'next','nɛkst',1,10,10),(18,7,'monday','''mʌndeɪ',1,9,10),
  -- 19: i would like to order a large coffee with no sugar
  (19,0,'i','aɪ',1,10,10),(19,1,'would','wʊd',1,10,10),(19,2,'like','laɪk',1,10,10),
  (19,3,'to','tuː',1,10,10),(19,4,'order','''ɔːdə',1,9,10),(19,5,'a','ə',1,10,10),
  (19,6,'large','lɑːdʒ',1,10,10),(19,7,'coffee','''kɒfi',1,10,10),(19,8,'with','wɪð',0,6,10),
  (19,9,'no','nəʊ',1,10,10),(19,10,'sugar','''ʃʊɡə',1,9,10),
  -- 20: the results were far below our expectations this quarter
  (20,0,'the','ðə',1,10,10),(20,1,'results','rɪ''zʌlts',1,8,10),(20,2,'were','wɜː',1,9,10),
  (20,3,'far','fɑː',1,10,10),(20,4,'below','bɪ''ləʊ',1,9,10),(20,5,'our','aʊə',0,5,10),
  (20,6,'expectations','ˌɛkspɛk''teɪʃnz',0,4,5),(20,7,'this','ðɪs',1,10,10),
  (20,8,'quarter','''kwɔːtə',1,9,10);

-- corpus_sentence_scores
INSERT INTO corpus_sentence_scores (corpus_id, accuracy, fluency, prosody, integrity_words) VALUES
  (1, 7, 7, 8, 6), (2, 5, 6, 5, 4), (3, 9, 9, 9, 9), (4, 6, 7, 6, 5),
  (5, 9, 8, 9, 10), (6, 7, 8, 7, 5), (7, 6, 7, 6, 6), (8, 10, 10, 9, 8),
  (9, 6, 6, 7, 4), (10, 10, 10, 10, 7), (11, 7, 8, 7, 7), (12, 7, 6, 7, 6),
  (13, 8, 8, 8, 9), (14, 3, 4, 4, 2), (15, 8, 9, 8, 9), (16, 7, 8, 7, 6),
  (17, 4, 5, 4, 6), (18, 7, 8, 7, 7), (19, 9, 9, 8, 10), (20, 6, 7, 6, 7);

-- ─── Tasks ────────────────────────────────────────────────────────────────────

INSERT INTO tasks (id, project, batch, total, completed, audio_size, status, due_date, lang) VALUES
  (113373, 56, 722, 100, 60,  '0.10M', 'in-progress', '2024-03-15', 'EN-US'),
  (113201, 54, 710, 80,  80,  '0.08M', 'completed',   '2024-03-10', 'EN-UK'),
  (113450, 57, 730, 120, 0,   '0.14M', 'pending',     '2024-03-20', 'EN-US'),
  (113088, 53, 698, 60,  45,  '0.06M', 'in-progress', '2024-03-14', 'ZH-CN'),
  (112990, 52, 685, 50,  50,  '0.05M', 'completed',   '2024-03-08', 'EN-US'),
  (113600, 58, 741, 200, 12,  '0.22M', 'in-progress', '2024-03-22', 'FR-FR');

-- ─── Projects ─────────────────────────────────────────────────────────────────

INSERT INTO projects (id, title, lang, type, rate, tasks, deadline, difficulty, description, slots, applied) VALUES
  (59, 'English Conversational Speech', 'EN-US', 'Transcription + Scoring', '$0.12/task', 500, '2024-04-01', 'Medium',
   'Spontaneous conversational speech from native speakers. Annotators must assess phoneme accuracy, stress and fluency.', 8, 0),
  (60, 'Mandarin Tone Annotation', 'ZH-CN', 'Transcription', '$0.15/task', 300, '2024-03-28', 'Hard',
   'Tonal speech data from multiple regional dialects. Requires Mandarin native proficiency and IPA knowledge.', 3, 0),
  (61, 'British English Read Speech', 'EN-UK', 'Transcription + QA', '$0.09/task', 800, '2024-04-15', 'Easy',
   'Read aloud sentences from a prompted corpus. Clean, studio-quality recordings with standard British accents.', 20, 1),
  (62, 'French Prosody Evaluation', 'FR-FR', 'Scoring', '$0.18/task', 200, '2024-03-30', 'Hard',
   'Evaluate prosodic patterns in French speech: rhythm, stress, intonation and linking. Native French required.', 5, 0);

INSERT INTO project_tags (project_id, tag) VALUES
  (59,'Phonetic'),(59,'Scoring'),(59,'Native'),
  (60,'Tonal'),(60,'IPA'),(60,'Expert'),
  (61,'Read Speech'),(61,'Studio'),(61,'QA'),
  (62,'Prosody'),(62,'Native'),(62,'Scoring');

-- ─── Statistics ───────────────────────────────────────────────────────────────

INSERT INTO project_breakdown (project, completed, total, avg_score) VALUES
  ('P52', 50, 50, 8.9), ('P53', 45, 60, 8.2), ('P54', 80, 80, 9.1), ('P56', 60, 100, 7.8);

INSERT INTO user_stats (label, value, sub) VALUES
  ('Total Completed', '235', 'tasks'), ('Avg. Quality Score', '8.4', '/ 10'),
  ('Hours Contributed', '64.2', 'hrs'), ('Acceptance Rate', '97%', 'approved');

INSERT INTO weekly_activity (day, tasks) VALUES
  ('Mon',18),('Tue',24),('Wed',31),('Thu',20),('Fri',28),('Sat',10),('Sun',5);

INSERT INTO monthly_scores (week, avg) VALUES
  ('W1',7.2),('W2',7.8),('W3',8.1),('W4',7.9),('W5',8.4),('W6',8.7),('W7',8.5),('W8',9.0);

-- ─── User Profile ─────────────────────────────────────────────────────────────

INSERT INTO user_profile (name, email, user_id, timezone, primary_lang, proficiency, notify_new_task, notify_score, notify_payment) VALUES
  ('Alex Chen', 'alex.chen@speechlab.io', '10109', 'UTC+8 – Asia/Shanghai', 'English (US)', 'Native', 1, 1, 0);

INSERT INTO user_secondary_langs (user_id, lang) VALUES
  (1, 'Mandarin Chinese');
