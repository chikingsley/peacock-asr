# speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment

Junbo Zhang1
, Zhiwen Zhang2
, Yongqing Wang1
, Zhiyong Yan1
,
Qiong Song2
, Yukai Huang2
, Ke Li2
, Daniel Povey1
, Yujun Wang1
1
Xiaomi Corporation, Beijing, China
2
SpeechOcean Ltd., Beijing, China
{zhangjunbo1, wangyongqing3, yanzhiyong, dpovey, wangyujun}@xiaomi.com,
{zhangzhiwen01, songqiong, huangyukai, like}@speechocean.com
## Abstract
This paper introduces a new open-source speech corpus named
“speechocean762” designed for pronunciation assessment use,
consisting of 5000 English utterances from 250 non-native
speakers, where half of the speakers are children. Five experts annotated each of the utterances at sentence-level, wordlevel and phoneme-level. A baseline system is released in open
source to illustrate the phoneme-level pronunciation assessment
workflow on this corpus. This corpus is allowed to be used
freely for commercial and non-commercial purposes. It is available for free download from OpenSLR, and the corresponding
baseline system is published in the Kaldi speech recognition
toolkit.
Index Terms: corpus, computer-assisted language learning
(CALL), second language (L2)
## 1 Introduction
As an indispensable part of Computer-aided language learning
(CALL), computer-aided pronunciation training (CAPT) applications with pronunciation assessment technology are widely
used in foreign language learning [1, 2] and proficiency tests
[3]. CAPT has been proved very useful to improve the pronunciation of the foreign language learners [4]. Due to the acute
shortage of qualified teachers [5] and the increasing popularity
of online learning, the research of pronunciation assessment is
being paid more attention [6].
According to the real-world CAPT applications’ features,
we divide the practical pronunciation assessment tasks into
three categories by the assessment granularity: sentence-level,
word-level, and phoneme-level. The sentence-level assessment evaluates the whole sentence. Specifically, three types
of sentence-level scores frequently appear in practical CAPT
systems: accuracy, completeness, and fluency. The accuracy
indicates the level of the learner pronounce each word in the utterance correctly; the completeness indicates the percentage of
the words that are actually pronounced, and the fluency here is
in the narrow sense[7], which focuses on whether the speaker
pronounces smoothly and without unnecessary pauses. The
word-level assessment has a finer scale than the sentence-level
assessment. Typical word-level scores are accuracy and stress.
Furthermore, as the finest granularity assessment, the phonemelevel assessment evaluates each phone’s pronunciation quality
in the utterance. Note that the word-level accuracy score should
not be regarded as the simple average of the phone-level accuracy scores, although they have strong correlations. Take
the word “above” (/@"b2v/) as an example. A foreign language
learner may mispronounce it as /@"bAv/ (mispronounce /2/ to /A/
) or as /@"k2v/ (mispronounce /b/ to /k/). For the two incorrect
pronunciations, the numbers of the mispronounced phones are
both one, but most people may realize that the latter mispronunciation is worse than the former.
There are some public corpora for pronunciation assessment. The ISLE Speech Corpus [8] is an early and widely accepted [9, 10, 11] data set. It contains mispronunciation tags
at the word and phoneme level, and the speakers are all from
German and Italian. It is free for academic use, but it is charged
for commercial use. ERJ [12] is another famous non-native English corpus for pronunciation assessment, collected from 202
Japanese students annotated with phonemic and prosodic symbols. ATR-Gruhn [13] is a non-native English corpus with multiple accents. The annotations of ATR-Gruhn are speaker-level
proficiency ratings. TL-school [14] is a corpus of speech utterances collected in northern Italy schools for assessing the
performance of students learning both English and German.
The data set of a spoken CALL shared task [15] is available
to download, where Swiss students answer prompts in English,
and the students’ responses are manually labeled as “accept”
or “reject”. L2-ARCTIC [16] is a non-native English speech
corpus with manual annotations, which has been used in some
recent studies [17, 18], and it uses substitution, deletion, and insertion to annotate for the phoneme-level scoring. Sell-corpus
[19] is another multiple accented Chinese-English speech corpus with phoneme substitution annotations. Some corpora, such
as CU-CHLOE [20], Supra-CHLOE [21] and COLSEC [22],
have been used in many studies [23, 24, 25, 26] but are not
publicly available. Corpora for languages other than English
also exist. The Tokyo-Kikuko [27] is a non-native Japanese corpus with phonemic and prosodic annotations. The iCALL corpus [28] is a Mandarin corpus spoken by non-native speakers
of European descent with annotated pronunciation errors. The
SingaKids-Mandarin [29] corpus focuses on mispronunciation
patterns in Singapore children’s Mandarin speech.
To our knowledge, none of the existing non-native English
corpora for pronunciation assessment contains all the following
features:
• It is available for free download for both commercial and
non-commercial purposes.
• The speaker variety encompasses young children and
adults.
• The manual annotations are in many aspects at sentencelevel, word-level and phoneme-level.
To meet these features, we created this corpus to support researchers in their pronunciation assessment studies. The corpus
arXiv:2104.01378v2 [cs.CL] 2 Jun 2021
Figure 1: Recording setup. Speakers read the text holding their
mobile phones in a quiet room.
Figure 2: Speaker’s English pronunciation proficiency distributions.
is available on the OpenSLR 1
website, and the corresponding
baseline system has been a part of the Kaldi speech recognition
toolkit 2
.
The rest of this paper is organized as follows: Section 2
describes the audio acquisition. Section 3 details how we annotated the data for the pronunciation assessment tasks. In Section
4, a Kaldi recipe for this corpus is introduced, which illustrates
how to do phoneme-level pronunciation assessment, and the experiment results are provided as well.
## 2 Audio Acquisition
This corpus’s text script is selected from daily life text, containing about 2,600 common English words. As shown in Figure
1, speakers were asked to hold their mobile phones 20cm from
their mouths and read the text as accurately as possible in a
quiet 3×3 meters room. The mobile phones include the popular
models of Apple, Samsung, Xiaomi, and Huawei. The number of sentences read aloud by each speaker is 20, and the total
duration of the audio is about 6 hours.
Thespeakersare250Englishlearnerswhosemothertongue
is Mandarin. The training set and test set are divided randomly,
with 125 speakers for each.
We carefully selected the speakers considering gender, age
and proficiency of English. The experts roughly rated the
speaker’s English pronunciation proficiency into three levels:
good, average, and poor. Figure 2 shows the distributions of the
speaker’s English pronunciation proficiency. Figure 3 shows
the distributions of the speaker’s age. The gender ratio is 1:1
for both adults and children.
## 3 Manual Annotation
Manual annotations are the essential part of this corpus. The annotations are the scores that indicate the pronunciation quality.
Each utterance in this corpus is scored manually by five experts
independently under the same metrics.
1https://www.openslr.org/101
2https://github.com/kaldi-asr/kaldi/tree/
master/egs/gop_speechocean762
Figure 3: Speaker’s age distributions.
Figure 4: The “SpeechOcean uTrans” Application. Before this
dialog is displayed, the experts have reached an agreement on
the canonical phone sequences by voting. For the phonemelevel scoring, the expert selects the phone symbol and then
makes a score of 0 or 1. If a phone symbol is not be selected,
the score would be 2 as the default.
## 3.1 Manual Scoring Metrics
The experts discussed and formulated the manual scoring metrics. Table 1 shows the detailed metrics. The phoneme-level
score is the pronunciation accuracy of each phone. The wordlevel scores include accuracy and stress, and the sentence-level
scores include accuracy, completeness, fluency and prosody.
The sentence-level completeness score, which is not depicted
in Table 1, is the percentage of the words in the target text that
are actually pronounced.
## 3.2 The Multiple Canonical Phone Sequences Problem
The phoneme-level scoring requires determining the canonical
phone sequence. A problem in practice is that the canonical
phone sequence may not be unique. Take the word “fast” as
an example. In middle school, most Chinese students were
taughtthatthiswordshouldbepronouncedas/fA:st/, soaproper
canonical phone sequence is “F AA S T” with the phone set defined by the CMU Dictionary [30]. However, some speakers
may pronounce this word as /fæst/ following the American pronunciation. If that is the case, the phone “AA” in the canonical
phone sequence “F AA S T” would be misjudged as low score.
Table 1: Manual Scoring Metrics
Score Description
Phoneme-level Accuracy
## 2 The phone is pronounced correctly
## 1 The phone is pronounced with a heavy accent
## 0 The pronunciation is incorrect or missed
Word-level Accuracy
## 10 The pronunciation of the whole word is correct
7-9 Most phones in the word are pronounced correctly, but the word’s pronunciation has heavy accents
4-6 No more than 30% phones in the word are wrongly pronounced
2-3 More than 30% phones in the word are wrongly pronounced, or be mispronounced into some other word
0-1 The whole pronunciation is hard to distinguish or the word is missed
Word-level Stress
## 10 The stress position is correct, or the word is a mono-syllable word
## 5 The stress position is incorrect
Sentence-level Accuracy
9-10 The overall pronunciation of the sentence is excellent without obvious mispronunciation
7-8 The overall pronunciation of the sentence is good, with a few mispronunciations
5-6 The pronunciation of the sentence has many mispronunciations but it is still understandable
3-4 Awkward pronunciation with many serious mispronunciations
0-2 The pronunciation of the whole sentence is unable to understand or there is no voice
Sentence-level Fluency
8-10 Coherent speech, without noticeable pauses, repetition or stammering
6-7 Coherent speech in general, with a few pauses, repetition and stammering
4-5 The speech is incoherent, with many pauses, repetition and stammering
0-3 The speaker is not able to read the sentence as a whole or there is no voice
Sentence-level Prosodic
9-10 Correct intonation, stable speaking speed and rhythm
7-8 Nearly correct intonation at a stable speaking speed
3-6 Unstable speech speed, or the intonation is inappropriate
0-2 The reading of the sentence is too stammering to do prosodic scoring or there is no voice
Figure 5: Building LG directly for the word “fast” with the canonical phone sequence voted by the experts, with skippable silence.
Figure 6: The part related of the word “fast” in L.
The proper canonical phone sequence, in this case, should be
“F AE S T”.
Our solution is as follows. For each word, experts will be
shown several possible canonical phone sequences before scoring. The expert must first select the sequence that is closest to
the pronunciation in her or his belief. Since there are five experts, the sequence chosen by each expert may be different, so
the five experts vote to determine the final canonical sequence.
Then all the experts use the same canonical phone sequence to
score. The canonical phone sequences are carried as a part of
the corpus’s meta-information.
## 3.3 Scoring Workflow
We developed an application named “SpeechOcean uTrans” for
the experts to convieniently score the audio. The interface of
the application is shown in Figure 4.
Before the scoring, the experts read the transcript and listen
to the audio to get familiar with the utterance. Then the experts
are required to listen to the audio repeatedly at least three times.
As we mentioned, some words have more than one canonical
phone sequence. For those words, experts need to choose and
vote to reach an agreement on the canonical phone sequence.
Then the experts score the audio following the scoring metrics
expressedinTable1. Ifthescoresseemunreasonable, forexample, the word-level score is high but all the phone-level scores
are low, the “SpeechOcean uTrans” application would raise a
warning message to remind the expert to recheck the scores.
## 3.4 Score Distribution
Figure7showsthedistributionofthesentence-levelscores. The
phoneme-level and word-level score distributions are shown in
the Figure 8, where the phoneme-level scores are mapped linFigure 7: Sentence-level score distribution.
Figure 8: Score distribution in different levels.
early to the range 0 to 10 for comparison. The sentence-level
scores variety encompasses 3 to 10, while most of the wordlevelandphoneme-levelscores are from 8 to10. Thisbehaviour
stems from the fact that high sentence-level scores rely on a
consistently “good” word and phoneme pronouncation. Even
a single word mispronunciation can lead to a low overall score.
Due to limited space, we suggest readers to refer to the available
online corpus to obtain the detailed statistics.
## 4 The Kaldi Recipe
For demonstrating how to use this corpus to score at phonemelevel, we uploaded a recipe named “gop speechocean762” to
the Kaldi toolkit.
## 4.1 Pipeline
We believe that the classical method is more suitable for building the baseline system than the latest methods. So the pipeline
is built following the neural network (NN) based goodness of
pronunciation(GOP)method, whichiswidelyusedanddetailed
in [31]. Here we only represent some specifics of implementing it on Kaldi. The GOP method requires a pre-trained acoustic model trained by native spoken data, which is trained by
the “egs/librispeech/s5/local/nnet3/run tdnn.sh” script in Kaldi.
The frame-level posterior matrix is generated through forward
propagation on the native acoustic model, and the matrix is used
for the forced alignment and the computing to obtain the GOP
values and the GOP-based features, whose definitions could
be found in [31] as well. Then we train a regressor for each
phone using the GOP-based features to predict the phonemelevel scores.
## 4.2 Alignment Graph Building without Lexicon
Kaldi’s default alignment setup does not guarantee the alignment output to be identical to the canonical phone sequence
voted by the experts. We continue to use the word “fast” as the
example. Thetwopossiblephonesequencesofthisword, which
are “F AA S T” and “F AE S T” specifically, are both contained
Table 2: Performance of the recipe
MSE PCC
GOP value 0.69 0.25
GOP-based feature 0.16 0.45
in the lexicon finite state transducer (FST), shown in Figure 6.
In that case, the phone sequence produced by the alignment is
uncertain. If the experts’ canonical phone sequence differs from
the alignment result, the scores will not be comparable with the
manual scores.
Therefore, we build the lexicon-to-grammar (LG) FST directly using the canonical phone sequence voted by the experts
without composing the lexicon FST and the grammar FST. The
process of directly constructing LG is simple: first, construct a
linear FST structure, whose input labels are the canonical phone
sequences voted by the experts, whereas the output labels are
thecorrespondingwordsandepsilons[32]. Then, addskippable
silence between the words, and use the disambiguation symbol
to construct the tail at the end of LG, as shown in Figure 5.
## 4.3 Supervised Training and Data Balancing
With the GOP-based features and the corresponding manual
scores, we train a regressor for each mono phone. The model
structure is a support vector regressor (SVR) [33]. Besides, we
train polynomial regression models with the GOP values directly for each phone as an alternative lightweight method.
A problem is that the data’s phoneme-level scores are quite
unbalanced, as discussed in Section 3.4. We use the high-score
samples of other phones as the current phone’s low-score samples to supplement the training set to address this issue. For
example, a good pronunciation sample of the phone AE can be
considered as a poor pronunciation sample of the phone AA.
For the model training of a particular phone, we randomly select the samples of other phones with high manual scores, setting their scores as zero and add them to the training set.
## 4.4 Results
For evaluating the recipe’s performance, we compare the predicted scores with the manual scores to calculate the mean
squared error (MSE) and Pearson correlation coefficient (PCC).
The result is shown in Table 2.
As a baseline system, this recipe is based on the classical
NN-based GOP method without using latest techniques. So the
result is not quite strong, which is in line with our expectations.
## 5 Conclusions
We released an open-source corpus for pronunciation assessment tasks. The corpus includes bothchild and adult speechand
is manually annotated by five experts. The annotations are at
sentence-level, word-level and phoneme-level. A Kaldi recipe
is released to illustrate to use of the classic GOP method for
phoneme-level scoring. In the future, we will expand the recipe
to word-level and sentence-level scoring.
## 6 Acknowledgements
The authors would like to thank Jan Trmal for uploading this
corpus to OpenSLR. The authors would also like to thank Heinrich Dinkel and Qinghua Wu for their helpful suggestions.
## 7 References
[1] H. Franco, H. Bratt, R. Rossier, V. Rao Gadde, E. Shriberg,
V. Abrash, and K. Precoda, “Eduspeak®: A speech recognition
and pronunciation scoring toolkit for computer-aided language
learning applications,” Language Testing, vol. 27, no. 3, pp. 401–
418, 2010.
[2] G. Li, “The training skills of college students’ oral English based
on the computer-aided language learning environment,” in Journal of Physics: Conference Series, vol. 1578, no. 1. IOP Publishing, 2020, p. 012040.
[3] L. Gu, L. Davis, J. Tao, and K. Zechner, “Using spoken language
technology for generating feedback to prepare for the TOEFL
iBT® test: a user perception study,” Assessment in Education:
Principles, Policy & Practice, pp. 1–14, 2020.
[4] J. Wang, “On optimization of non-intelligence factors in college
English teaching in computer-aided language learning environments,” in Applied Mechanics and Materials, vol. 644. Trans
Tech Publ, 2014, pp. 6124–6127.
[5] K. P. McVey and J. Trinidad, “Nuance in the noise: The complex reality of teacher shortages.” Bellwether Education Partners,
2019.
[6] V. C.-W. Cheng, V. K.-T. Lau, R. W.-K. Lam, T.-J. Zhan, and
P.-K. Chan, “Improving English phoneme pronunciation with automatic speech recognition using voice chatbot,” in International
Conference on Technology in Education. Springer, 2020, pp.
88–99.
[7] P. Lennon, “The lexical element in spoken second language fluency,” in Perspectives on fluency. University of Michigan, 2000,
pp. 25–42.
[8] W. Menzel, E. Atwell, P. Bonaventura, D. Herron, P. Howarth,
R. Morton, and C. Souter, “The ISLE corpus of non-native spoken
English,” inProceedingsofLREC2000: LanguageResourcesand
Evaluation Conference, vol. 2. European Language Resources
Association, 2000, pp. 957–964.
[9] T. Oba and E. Atwell, “Using the HTK speech recogniser to anlayse prosody in a corpus of german spoken learner’s English,” in
UCRELTechnicalPapernumber16.Specialissue.Proceedingsof
the Corpus Linguistics 2003 conference. Lancaster University,
2003, pp. 591–598.
[10] F. Hönig, T. Bocklet, K. Riedhammer, A. Batliner, and E. Nöth,
“The automatic assessment of non-native prosody: Combining
classical prosodic analysis with acoustic modelling,” in Thirteenth
Annual Conference of the International Speech Communication
Association, 2012.
[11] S. Papi, E. Trentin, R. Gretter, M. Matassoni, and D. Falavigna,
“Mixtures of deep neural experts for automated speech scoring,”
Proc. Interspeech 2020, pp. 3845–3849, 2020.
[12] N. Minematsu, Y. Tomiyama, K. Yoshimoto, K. Shimizu, S. Nakagawa, M. Dantsuji, and S. Makino, “Development of English
speechdatabasereadbyJapanesetosupportcallresearch,” inProceedings of ICA, vol. 1. European Language Resources Association, 2004, pp. 557–560.
[13] R. Gruhn, T. Cincarek, and S. Nakamura, “A multi-accent nonnative English database,” in ASJ, 2004, pp. 195–196.
[14] R. Gretter, M. Matassoni, S. Bannò, and F. Daniele, “TLT-school:
a corpus of non native children speech,” in Proceedings of The
12th Language Resources and Evaluation Conference, 2020, pp.
378–385.
[15] C. Baur, C. Chua, J. Gerlach, E. Rayner, M. Russel, H. Strik, and
X. Wei, “Overview of the 2017 spoken call shared task,” in Workshop on Speech and Language Technology in Education (SLaTE),
2017.
[16] G. Zhao, S. Sonsaat, A. Silpachai, I. Lucic, E. ChukharevHudilainen, J. Levis, and R. Gutierrez-Osuna, “L2-ARCTIC: A
non-native English speech corpus,” Proc. Interspeech 2018, pp.
2783–2787, 2018.
[17] B.-C. Yan, M.-C. Wu, H.-T. Hung, and B. Chen, “An end-to-end
mispronunciation detection system for L2 English speech leveraging novel anti-phone modeling,” in Proc. Interspeech 2020, 2020,
pp. 3032–3036.
[18] Y. Feng, G. Fu, Q. Chen, and K. Chen, “SED-MDD: Towards sentence dependent end-to-end mispronunciation detection and diagnosis,” in IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP). IEEE, 2020, pp. 3492–3496.
[19] Y. Chen, J. Hu, and X. Zhang, “Sell-corpus: an open source multiple accented chinese-english speech corpus for l2 english learning assessment,” in IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP). IEEE, 2019, pp. 7425–
7429.
[20] K. Li, X. Qian, and H. Meng, “Mispronunciation detection and diagnosis in L2 English speech using multidistribution deep neural
networks,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 1, pp. 193–207, 2016.
[21] M. Li, S. Zhang, K. Li, A. M. Harrison, W.-K. Lo, and H. Meng,
“Design and collection of an L2 English corpus with a suprasegmental focus for chinese learners of English.” in ICPhS, 2011, pp.
1210–1213.
[22] H. Yang and N. Wei, Construction and data analysis of a Chinese learner spoken English corpus. Shanhai Foreign Languse
Eduacation Press, 2005.
[23] D. Luo, X. Yang, and L. Wang, “Improvement of segmental
mispronunciation detection with prior knowledge extracted from
large L2 speech corpus,” in Twelfth Annual Conference of the International Speech Communication Association, 2011.
[24] K. Li, X. Qian, S. Kang, and H. Meng, “Lexical stress detection
for L2 English speech using deep belief networks.” in Interspeech,
2013, pp. 1811–1815.
[25] K. Li, X. Wu, and H. Meng, “Intonation classification for L2 Englishspeechusingmulti-distributiondeepneuralnetworks,” Computer Speech & Language, vol. 43, pp. 18–33, 2017.
[26] K. Li, S. Mao, X. Li, Z. Wu, and H. Meng, “Automatic lexical stress and pitch accent detection for L2 English speech using
multi-distributiondeepneuralnetworks,” SpeechCommunication,
vol. 96, pp. 28–36, 2018.
[27] K. Nishina, Y. Yoshimura, I. Saita, Y. Takai, K. Maekawa,
N. Minematsu, S. Nakagawa, S. Makino, and M. Dantsuji, “Development of Japanese speech database read by non-native speakersforconstructingcallsystem,” inProc.ICA,2004, pp.561–564.
[28] N. F. Chen, R. Tong, D. Wee, P. Lee, B. Ma, and H. Li, “iCALL
corpus: Mandarin chinese spoken by non-native speakers of european descent,” in Sixteenth Annual Conference of the International Speech Communication Association, 2015.
[29] G. Shang and S. Zhao, “Singapore mandarin: Its positioning, internal structure and corpus planning,” in Paper presented atthe
22nd Annual Conference of the Southeast Asian Linguistics Society, Agay, France, 2012.
[30] R. Weide, “The CMU pronunciation dictionary.” Carnegie Mellon University, 1998.
[31] W. Hu, Y. Qian, F. K. Soong, and Y. Wang, “Improved mispronunciation detection with deep neural network trained acoustic
models and transfer learning based logistic regression classifiers,”
Speech Communication, vol. 67, pp. 154–166, 2015.
[32] M. Mohri, F. Pereira, and M. Riley, “Speech recognition with
weightedfinite-statetransducers,” inSpringerhandbookofspeech
processing. Springer, 2008, pp. 559–584.
[33] H. Drucker, C. J. Burges, L. Kaufman, A. Smola, V. Vapnik et al.,
“Support vector regression machines,” Advances in neural information processing systems, vol. 9, pp. 155–161, 1997.
