---
title: "L2-ARCTIC: A Non-Native English Speech Corpus"
authors:
  - "Guanlong Zhao"
  - "Sinem Sonsaat"
  - "Alif Silpachai"
  - "Ivana Lucic"
  - "Evgeny Chukharev-Hudilainen"
  - "John Levis"
  - "Ricardo Gutierrez-Osuna"
citation_author: "Zhao et al"
year: 2018
doi: "10.21437/Interspeech.2018-1243"
source_pdf: "zhao2018interspeech.pdf"
extraction_method: "Extracted from PDF OCR"
extracted_at: "2026-04-17"
llm_friendly: true
---

## Abstract

In this paper, we introduce L2-ARCTIC, a speech corpus of non-native English that is intended for research in voice conversion, accent conversion, and mispronunciation detection. This initial release includes recordings from ten non-native speakers of English whose first languages (L1s) are Hindi, Korean, Mandarin, Spanish, and Arabic, each L1 containing recordings from one male and one female speaker. Each speaker recorded approximately one hour of read speech from the Carnegie Mellon University ARCTIC prompts, from which we generated orthographic and forced-aligned phonetic transcriptions. In addition, we manually annotated 150 utterances per speaker to identify three types of mispronunciation errors: substitutions, deletions, and additions, making it a valuable resource not only for research in voice conversion and accent conversion but also in computer-assisted pronunciation training. The corpus is publicly accessible at [https://psi.engr.tamu.edu/l2-arctic-corpus/](https://psi.engr.tamu.edu/l2-arctic-corpus/).

## Keywords

speech corpus, voice conversion, accent conversion, mispronunciation detection

## 1 Introduction

Voice conversion (VC) [1] aims to transform utterances from a source speaker to make them sound as if a target speaker had uttered them. The closely related problem of accent conversion (AC) [2] goes a step further, mixing the source speech’s linguistic content and accent with the target speaker’s voice quality to create utterances with the target’s voice but the content and pronunciation of the source speaker. When teaching a second language (L2), accent conversion can be used to create a “golden speaker,” a synthesized voice that has the learner’s voice quality but with a native speaker’s accent (e.g., prosody, intonation, pronunciation) [3]. Several studies [4, 5] have suggested that having such a “golden speaker” to imitate can be beneficial in pronunciation training. Furthermore, in addition to providing language learners with a suitable voice to mimic, detecting mispronunciations is also a critical component for providing useful feedback to the learners in computer-assisted pronunciation training [6].

To train and evaluate voice and accent conversion systems designed for non-native speakers, one needs high-quality parallel recordings from the source and target speakers. Likewise, to develop and benchmark mispronunciation detection algorithms, detailed phoneme level annotations on pronunciation errors (e.g., phone substitution, additions, and deletions) are required. However, existing non-native English corpora (e.g., Speech Accent Archive [7] and IDEA [8]) do not fulfill these requirements.

## 2 The need for a new L2 English corpus

A number of voice conversion studies [9-12] have relied on the Carnegie Mellon University (CMU) ARCTIC speech corpus [13] and, more recently, the Voice Conversion Challenge (VCC) dataset [14]. However, little attention has been paid to voice conversion between non-native speakers of English, in part due to the lack of high-quality speech recordings from those speakers, despite 80% of the English speakers in the world being non-native [15]. These standard corpora are not suitable for either voice conversion between non-native speakers nor accent conversion tasks.

Among the non-native English corpora, the Speech Accent Archive [7] and IDEA [8] cover a wide range of native languages and speakers. However, each speaker only recorded a short paragraph (Speech Accent Archive) or a short free speech task (IDEA), and most of the recordings have strong background noise, making them ill-suited for voice/accent conversion.

## 3 Corpus curation procedure

This initial release of L2-ARCTIC contains English speech of speakers from five different first languages: Hindi [27], Korean, Mandarin, Spanish, and Arabic.

### 3.1 Participants

For this initial release, we recruited two speakers (one male and one female) for each of the L1s, for a total of ten speakers. Demographic information of the speakers is summarized in Table 1. The proficiency level of English was measured using TOEFL iBT scores [37].

| Speaker | L1 | Gender | TOEFL iBT |
| :--- | :--- | :--- | :--- |
| HKK | Korean | M | 114 |
| YDCK | Korean | F | 110 |
| BWC | Mandarin | M | 80 |
| LXC | Mandarin | F | 86 |
| YBAA | Arabic | M | 100 |
| SKA | Arabic | F | 79 |
| EBVS | Spanish | M | 70 |
| NJS | Spanish | F | 110 |
| RRBI | Hindi | M | 91 |
| TNI | Hindi | F | 99 |

*Table 1: Demographic information of the speakers*

### 3.2 Recording the corpus

To create the corpus, we used the 1,132 sentences in the CMU ARCTIC prompts. The speech was recorded in a quiet room at Iowa State University (ISU). We used a Samson C03U microphone and Earamble studio microphone pop filter for recordings.

### 3.3 Corpus annotations

Our corpus provides orthographic transcriptions at the word level. We used the Montreal forced-aligner [41] to produce phonetic transcriptions in PRAAT’s TextGrid format [42]. In addition, we manually annotated 150 utterances per speaker to identify three types of mispronunciation errors: substitutions, deletions, and additions.

## 4 Corpus statistics

In total, the dataset contains 11,026 utterances. The total duration of the corpus is 11.2 hours. Table 2 provides a breakdown of pronunciation errors by L1s.

| L1 | Substitutions | Deletions | Additions |
| :--- | :--- | :--- | :--- |
| Hindi | DH→D, Z→S, W→V, EY→EH, TH→T | R, D, T, ER, HH | R, AH, S, Y, AA |
| Korean | DH→D, Z→S, IH→IY, OW→AO, EH→AE | D, T, R, HH, K | AX, IH, AH, S, Y |
| Mandarin | Z→S, DH→D, IH→IY, N→NG, V→F | D, T, R, L, N | AH, AX, IH, N, R |
| Spanish | Z→S, IH→IY, DH→D, AE→AA, AH→AO | D, T, AH, Z, IH | EH, AX, AH, IH, IY |
| Arabic | P→B, OW→AO, R→ERR, DH→Z, Z→S | T, R, D, AH, IH | G, AH, IH, AX, EH |

*Table 2: Most frequent errors by native language; the top-5 error occurrences are listed in descending order*

## 5 Mispronunciation detection evaluation

This section provides initial results on mispronunciation detection using the 10 speakers that we have currently released. Our implementation is based on the conventional Goodness of Pronunciation (GOP) method as defined in [44]. The Precision-Recall curve is shown in Figure 4 (in the original paper).

## 6 Conclusion

This paper has presented L2-ARCTIC, a new non-native English speech corpus designed for voice conversion, accent conversion, and mispronunciation detection tasks. Each speaker in L2-ARCTIC produced sufficient speech data to capture their voice identity and accent characteristics. The corpus is released under the CC BY-NC 4.0 license [47] and is available at [https://psi.engr.tamu.edu/l2-arctic-corpus/](https://psi.engr.tamu.edu/l2-arctic-corpus/).

---

## References

1. S. H. Mohammadi and A. Kain, "An overview of voice conversion systems," Speech Communication, vol. 88, pp. 65-82, 2017.
2. S. Aryal and R. Gutierrez-Osuna, "Can Voice Conversion Be Used to Reduce Non-Native Accents?," in ICASSP, 2014, pp. 7879-7883.
3. D. Felps, H. Bortfeld, and R. Gutierrez-Osuna, "Foreign accent conversion in computer assisted pronunciation training," Speech Communication, vol. 51, no. 10, pp. 920-932, 2009.
4. K. Probst, Y. Ke, and M. Eskenazi, "Enhancing foreign language tutors–in search of the golden speaker," Speech Communication, vol. 37, no. 3, pp. 161-173, 2002.
5. M. P. Bissiri, H. R. Pfitzinger, and H. G. Tillmann, "Lexical stress training of German compounds for Italian speakers by means of resynthesis and emphasis," in Australian International Conference on Speech Science & Technology, 2006, pp. 24-29.
6. J. Levis, "Computer technology in teaching and researching pronunciation," Annual Review of Applied Linguistics, vol. 27, pp. 184-202, 2007.
7. S. Weinberger. Speech accent archive [Online]. Available: [http://accent.gmu.edu](http://accent.gmu.edu)
8. P. Meier. IDEA: International Dialects of English Archive [Online]. Available: [http://www.dialectsarchive.com/](http://www.dialectsarchive.com/)
9. T. Toda, A. W. Black, and K. Tokuda, "Voice conversion based on maximum-likelihood estimation of spectral parameter trajectory," IEEE Transactions on Audio, Speech, and Language Processing, vol. 15, no. 8, pp. 2222-2235, 2007.
10. L. Sun, S. Kang, K. Li, and H. Meng, "Voice conversion using deep bidirectional long short-term memory based recurrent neural networks," in ICASSP, 2015, pp. 4869-4873.
11. G. Zhao and R. Gutierrez-Osuna, "Exemplar selection methods in voice conversion," in ICASSP, 2017, pp. 5525-5529.
12. Y.-C. Wu, H.-T. Hwang, C.-C. Hsu, Y. Tsao, and H.-M. Wang, "Locally Linear Embedding for Exemplar-Based Spectral Conversion," in Interspeech, 2016, pp. 1652-1656.
13. J. Kominek and A. W. Black, "The CMU Arctic speech databases," in Fifth ISCA Workshop on Speech Synthesis, 2004, pp. 223-224.
14. T. Toda et al., "The Voice Conversion Challenge 2016," in Interspeech, 2016, pp. 1632-1636.
15. J. Jenkins. (2008). English as a lingua franca. Available: [http://www.jacet.org/2008convention/JACET2008_keynote_jenkins.pdf](http://www.jacet.org/2008convention/JACET2008_keynote_jenkins.pdf)
16. K. J. Van Engen, M. Baese-Berk, R. E. Baker, A. Choi, M. Kim, and A. R. Bradlow, "The Wildcat Corpus of native-and foreign accented English: Communicative efficiency across conversational dyads with varying language alignment profiles," Language and speech, vol. 53, no. 4, pp. 510-540, 2010.
17. T. Lander. CSLU: Foreign Accented English Release 1.2 LDC2007S08 [Online]. Available: [https://catalog.ldc.upenn.edu/ldc2007s08](https://catalog.ldc.upenn.edu/ldc2007s08)
18. T. Bent and A. R. Bradlow, "The interlanguage speech intelligibility benefit," The Journal of the Acoustical Society of America, vol. 114, no. 3, pp. 1600-1610, 2003.
19. K. Li, X. Qian, and H. Meng, "Mispronunciation detection and diagnosis in l2 english speech using multidistribution deep neural networks," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 1, pp. 193-207, 2017.
20. H. Yang and N. Wei, "Construction and data analysis of a Chinese learner spoken English corpus," ed: Shanhai Foreign Languse Eduacation Press, 2005.
21. W. Menzel et al., "The ISLE corpus of non-native spoken English," in Proceedings of LREC 2000: Language Resources and Evaluation Conference, vol. 2, 2000, pp. 957-964.
22. N. F. Chen, R. Tong, D. Wee, P. X. Lee, B. Ma, and H. Li, "SingaKids-Mandarin: Speech Corpus of Singaporean Children SpeakingMandarinChinese,"in Interspeech, 2016,pp.1545-1549.
23. W. Hu, Y. Qian, F. K. Soong, and Y. Wang, "Improved mispronunciation detection with deep neural network trained acoustic models and transfer learning based logistic regression classifiers," Speech Communication, vol. 67, pp. 154-166, 2015.
24. Y.-B. Wang and L.-s. Lee, "Supervised detection and unsupervised discovery of pronunciation error patterns for computer-assisted language learning," IEEE/ACM Transactions on Audio, Speech and Language Processing, vol. 23, no. 3, pp. 564-579, 2015.
25. H. Huang, H. Xu, X. Wang, and W. Silamu, "Maximum F1-score discriminative training criterion for automatic mispronunciation detection," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 4, pp. 787-797, 2015.
26. G. Zhao, S. Sonsaat, J. Levis, E. Chukharev-Hudilainen, and R. Gutierrez-Osuna, "Accent Conversion Using Phonetic Posteriorgrams," in ICASSP, 2018.
27. P. Pramod, "Indian English Pronunciation," in The Handbook of English Pronunciation, M. Reed and J. Levis, Eds.: Wiley Blackwell, 2015, pp. 301-319.
28. S.-A. Jun, "Prosody in sentence processing: Korean vs. English," UCLA Working Papers in Phonetics, vol. 104, pp. 26-45, 2005.
29. M. Ueyama and S.-A. Jun, "Focus realization of Japanese English and Korean English intonation," UCLA Working Papers in Phonetics, pp. 110-125, 1996.
30. J. Anderson-Hsieh, R. Johnson, and K. Koehler, "The relationship between native speaker judgments of nonnative pronunciation and deviance in segmentais, prosody, and syllable structure," Language learning, vol. 42, no. 4, pp. 529-555, 1992.
31. M. C. Pennington and N. C. Ellis, "Cantonese speakers' memory for English sentences with prosodic cues," The Modern Language Journal, vol. 84, no. 3, pp. 372-389, 2000.
32. J. Chang, "Chinese speakers," Learner English, vol. 2, pp. 310-324, 1987.
33. J. Morley, "Teaching American English Pronunciation," ed: JSTOR, 1993.
34. B. Smith, Learner English: A teacher's guide to interference and other problems. Cambridge University Press, 2001.
35. M. Benrabah, "Word-stress–a source of unintelligibility in English," IRAL-International Review of Applied Linguistics in Language Teaching, vol. 35, no. 3, pp. 157-166, 1997.
36. K. De Jong and B. A. Zawaydeh, "Stress, duration, and intonation in Arabic word-level prosody," Journal of Phonetics, vol. 27, no. 1, pp. 3-22, 1999.
37. Y. Cho and B. Bridgeman, "Relationship of TOEFL iBT® scores to academic performance: Some evidence from American universities," Language Testing, vol. 29, no. 3, pp. 421-442, 2012.
38. H. Zen et al., "The HMM-based speech synthesis system (HTS) version 2.0," in SSW, 2007, pp. 294-299.
39. D. Erro, E. Navas, and I. Hernaez, "Parametric voice conversion based on bilinear frequency warping plus amplitude scaling," IEEE Transactions on Audio, Speech, and Language Processing, vol. 21, no. 3, pp. 556-566, 2013.
40. Audacity®. Available: [http://www.audacityteam.org/](http://www.audacityteam.org/)
41. M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, "Montreal Forced Aligner: trainable text-speech alignment using Kaldi," in Interspeech, 2017, pp. 498-502.
42. P. P. G. Boersma, "Praat, a system for doing phonetics by computer," Glot international, vol. 5, 2002.
43. M. Munro, "How well can we predict L2 learners' pronunciation difficulties?," CATESOL Journal, vol. 30, no. 1, pp. 267-282, 2018.
44. S. M. Witt and S. J. Young, "Phone-level pronunciation scoring and assessment for interactive language learning," Speech communication, vol. 30, no. 2, pp. 95-108, 2000.
45. D. Povey et al., "The Kaldi speech recognition toolkit," in IEEE 2011 Workshop on Automatic Speech Recognition & Understanding, 2011.
46. V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books," in ICASSP, 2015, pp. 5206-5210.
47. Creative Commons Attribution-NonCommercial 4.0 International Public License. Available: [https://creativecommons.org/licenses/by-nc/4.0/legalcode](https://creativecommons.org/licenses/by-nc/4.0/legalcode)
