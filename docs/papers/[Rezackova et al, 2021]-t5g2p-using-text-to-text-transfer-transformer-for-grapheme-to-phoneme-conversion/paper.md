INTERSPEECH 2021
30 August – 3 September, 2021, Brno, Czechia



                               T5G2P: Using Text-to-Text Transfer Transformer
                                   for Grapheme-to-Phoneme Conversion
                                            Markéta Řezáčková, Jan Švec, Daniel Tihelka

                   New Technologies for the Information Society and Department of Cybernetics,
                  Faculty of Applied Sciences, University of West Bohemia, Pilsen, Czech Republic
                            juzova@ntis.zcu.cz, honzas@ntis.zcu.cz, dtihelka@ntis.zcu.cz


                                Abstract                                        both in the case of primary use of a dictionary supplemented
                                                                                by post-processing rules (e.g. for distinguishing pronunciation
     Despite the increasing popularity of end-to-end text-to-speech             for “the” in English based on the following phoneme), or vice
     (TTS) systems, the correct grapheme-to-phoneme (G2P) mod-                  versa the use of rules for languages with regular pronuncia-
     ule is still a crucial part of those relying on a phonetic input. In       tion (e.g. Slavic languages) and dictionaries look-up only for
     this paper, we, therefore, introduce a T5G2P model, a Text-to-             the conversion of some exceptions to the rules.
     Text Transfer Transformer (T5) neural network model which                       A big challenge for languages with more or less irregular
     is able to convert an input text sentence into a phoneme se-               pronunciation (with e.g. English spelling being notoriously ir-
     quence with a high accuracy. The evaluation of our trained T5              regular) is an automatic G2P conversion of hitherto unknown
     model is carried out on English and Czech, since there are dif-            words. While automatic converters often cannot predict cor-
     ferent specific properties of G2P, including homograph disam-              rect pronunciation, that is rarely a problem for native speakers
     biguation, cross-word assimilation and irregular pronunciation             of a given language thanks to the human ability to generalize
     of loanwords. The paper also contains an analysis of a homo-               things already learned. This thought was at the birth of G2P ap-
     graphs issue in English and offers another approach to Czech               proaches using different types of neural networks [5, 6], includ-
     phonetic transcription using the detection of pronunciation ex-            ing the state-of-the-art approaches based on transformer-based
     ceptions.                                                                  networks with an attention mechanism [7, 8]. As showed in
     Index Terms: grapheme-to-phoneme, phonetic transcription,                  the studies mentioned before, the neural networks model can
     T5, transformers, TTS system                                               generalize well and predict the correct pronunciation of unseen
                                                                                words. Such feature is usable for the use in dialog systems
                          1. Introduction                                       where the synthesised entities are not known in advance [9].
                                                                                     A large number of approaches work only at the word level,
     In text-to-speech (TTS) systems, the grapheme-to-phoneme                   trained from large dictionaries [5, 6, 7, 8]. Such an approach,
     (G2P) module is a crucial part affecting the overall intelligibil-         however, still requires some level of post-processing of the sen-
     ity and correctness of the synthesized text, as well as the process        tence/phrase phonetic transcription, including the consideration
     of voice building itself where the correctness of phonetic tran-           of relationships between the words, homographs disambigua-
     scription affects the overall quality of the synthetic voice [1].          tion etc. For this reason, it seems better to train G2P models
     Although there is a common shift towards the end-to-end TTS                directly on whole sentences or phrases, and let the model it-
     systems, taking a raw written text as an input [2, 3, 4], even             self to learn the inter-sentence word relations and cross-words
     these must carry out some kind of phonetic transcription inter-            specifics of the particular language [10, 11]. The training the
     nally, hidden in the neural network structure. However, the ex-            model for the phonetic transcription of input text on whole sen-
     plicit handling of G2P transcription and the possibility to anal-          tences actually corresponds to the task of machine translation
     yse and evaluate its output simply by text comparison, as well             [12, 13, 7, 14]: the input sequence of words in one language
     as the ability of fast-fixing of inappropriate pronunciations are          (orthographic words in our case) is converted to the output se-
     among the reasons why the G2P research remains highly topi-                quence of words in an another language (here, the sequence of
     cal.                                                                       phonetic word forms).
          Traditional approaches to G2P conversion usually depend
     on the sets of phonetic rules or pronunciation dictionaries, but
     mostly on a combination of both. The problem with the rule-
                                                                                                  2. Traditional G2P
     based approach is that even in the very regular languages,                 In the present paper we will focus on two languages – English
     from the pronunciation point of view, there are some excep-                and Czech. The reason of their choice was the different nature
     tions in phonetic transcription, especially for words originating          (irregular vs. regular) of how the pronunciations of words from
     from foreign languages (loanwords). On the other hand, the                 their orthographic forms are created, and thus the different way
     dictionary-based approaches suffer from the incompleteness of              of dealing with the problem of phonetic transcription in TTS
     the dictionary used, which, despite frequent updating, can never           systems.
     contain all the words of a given language – not mentioning huge                 Taking the ARTIC TTS system as an example [15], the G2P
     dictionaries for inflectional languages (containing all possible           module for English is based on a large pronunciation dictio-
     word forms), or additional automatic morphological analysis                nary containing about 300,000 words. That dictionary is ac-
     of words required in the case of lemma pronunciation dictio-               companied by more than 1,000 automatically derived rules [16]
     naries (possibly introducing other errors into the system). The            used as a fallback to handle out-of-vocabulary words. On the
     dictionary+rules combination can usually ensure a sufficiently             contrary, for Czech, an inflected language with rather regu-
     accurate conversion of the input text into its phonetic form,              lar phonetic transcription, there is a set of approximately 100




Copyright © 2021 ISCA                                                       6             http://dx.doi.org/10.21437/Interspeech.2021-546
context-dependent rules designed manually by phonetic experts                 • Pages with dirty and obscene words are removed.
[17] accompanied by a dictionary with the correct transcription               • Lines with the word ”JavaScript” and the curly braces
of more than 170,000 irregular word-forms (words in their in-                   { } are removed (remains of incorrect crawling of the
flected forms) in which the input is searched prior the use of                  webpage).
rules.
     In an effort to avoid the manual maintenance of the rela-                • The pages in the corpus were de-duplicated. The result-
tively large dictionaries for all the different languages supported             ing corpus contains each three-sentence span just once.
by [15], we have started to search for an universal and flexi-                 For the Czech language, we have collected the Common-
ble approach to G2P. The first attempt, based on sequence-to-             Crawl corpus at the end of August 2020. It contains 47.9GB of
sequence DNN-based approach inspired by machine translation               clean text, 20.5M unique URLs and 6.7B running words. The
[10] was able to reach similar or even higher accuracy than the           Czech T5 training procedure followed the original procedure
traditional baseline approach for the words of the most common            described in [19].
lengths. However, the accuracy started to drop in case of words                For both the English and Czech experiments, we used the
longer than 10 characters – sometimes the model was not able              t5-base architecture consisting of 220M parameters, 2×12
to “remember” the whole word and the failures started to appear           transformer block in the encoder and the decoder. The dimen-
towards the end of such words, as shown here1 :                           sionality of hidden layers and embeddings was 768. The atten-
    • suspiciously (EN) s@spIS@slll                                       tion mechanism uses 12 attention heads with inner dimension-
      (correct: s@spIS@slI)                                               ality 64.
    • autobiography (EN) O:t@UbIgrAffI                                    3.2. T5-model fine-tuning
      (correct: O:t@UbaIQgr@fI)
                                                                          For training the T5-based model for grapheme-to-phoneme task
    • přibývajı́cı́mi (CZ) pQibi:vaji:ci:mi:                            (T5G2P) we used the Tensorflow implementation of Hugging-
      (correct: pQibi:vaji:ci:mi)                                         Face Transformers library [21] together with the T5s4 library,
    • organizátorek (CZ) !organiza:ta:tek                                which simplifies mainly the training and prediction process by
      correct:!organiza:torek                                             accepting a simple tab-separated input-output pairs. The T5s li-
     Despite increasing encoder/decoder capacity, the model did           brary also uses the variable sequence length and variable batch
not yield better results and the accuracy on longer words was             size to fully utilize the underlying GPU used for training.
still significantly lower, as thoroughly shown in histogram in                 In the experiments, we used the ADAM optimization with
[11] and Figure 1. To mitigate these failures, we draw our at-            learning rate decay proportional to inverse square root of the
tention to Transformer-based model.                                       number of learning epochs.
                                                                               For both languages, we have a large amount of propri-
                                                                          etary sentences with their phonetic transcriptions at our dis-
                     3. T5-based G2P                                      posal; specifically, we have 128,532 English and 442,029 Czech
To examine the ability of transformer-based DNN to deal with              unique sentences. For both languages, the data were randomly
the problematic cases mentioned before, we used the pre-trained           split into train, valid and test (80%, 10% and 10%). For the
Text-to-Text Transfer Transformer (T5) model [19]. In general,            individual language-related models, whole sentences were used
the T5 model is trained as the full encoder-decoder transformer           as an input and the model was trained (fine-tuned) to generate
in a semi-supervised manner from a large text corpus. The input           the corresponding phonetic transcription of them. This training
sentence is perturbed and the goal of the model is to generate the        lasted 50 epochs, with 2000 steps per each.
output which corrects the perturbed input into an original one.
More specifically for T5, the random sub-sequences of tokens in                       4. Experiments and Results
the input are masked with a single token and the model predicts
the tokens hidden behind the masked token. The general advan-             The fine-tuned T5G2P model described in Section 3.2 was used
tage of the T5 model is the ability to perform many text-to-text          to predict the transcription of the test data (the selected 10% of
tasks like text summarization, topic detection or sentiment anal-         the fine-tuning dataset). The output, i.e. the predicted sequence
ysis.                                                                     of phones, was compared to the reference using the following
                                                                          measures: sentence accuracy, word accuracy and phoneme ac-
3.1. T5-model pre-training                                                curacy. The word accuracy clearly reflects the correctness of
                                                                          a transcription similarly to word-error-rate in ASR evaluation,
For experiments with English data, we used the Google’s T5-               as we can suppose invalid phonetic form as an invalid or unin-
base English model2 trained from Common Crawl data3 . We                  telligible word. On the other hand, the phoneme accuracy does
replicated the same pre-processing procedure to obtain the                not tell much about intelligibility (there may be few misses in
Czech data and we pre-trained our own T5 models for these lan-            many words or many misses in few words), but it is used in
guages. The pre-processing steps correspond with the steps pre-           other studies.As for the sentence accuracy, it gives us a higher-
sented in [19] for building the Colossal Clean Crawled Corpus             level overview of the failures and can be related to the presence
(C4) on which the Google’s T5 model was pre-trained. Such                 of uncommon words in the sentences.
rules are generally applied while processing web text [20]:                    The overall results for English and Czech are shown in Ta-
    • Only lines ending in a terminal punctuation are retained.           ble 1. The table compares the T5-based approach to the both
      Short pages and lines are discarded.                                baseline approaches, first based on pronunciation dictionaries
                                                                          and a sets of rules (see Section 2), and second the encoder-
   1 All the transcriptions are in SAMPA alphabet [18].
                                                                          decoder DNN-based G2P converter presented in [10, 11]. For
  2 https://github.com/google-research/
                                                                          the Czech, there is also an extra row showing the accuracy used
text-to-text-transfer-transformer
  3 https://commoncrawl.org/                                                 4 https://github.com/honzas83/t5s




                                                                      7
Table 1: Results of the tested English and Czech G2P outputs                        – in recent years, the published results range between 20% and
compared to the reference transcription.                                            30% word error rate, which means the accuracy 70-80% (e.g.
                                                                                    [8]). Our word accuracy value on English set is much higher,
                        approach         A sent         A word     A phoneme        however, we intentionally work on a sentence level to capture
                                                                                    cross-word assimilation [23, 24]. Therefore, although our test
                      dict + rules       54.49 %        90.93 %       97.20 %
 English
                                                                                    data contained only unseen sentences, many words occurred in
                    encoder-decoder      82.75 %        95.72 %       97.18 %
                                                                                    both training and testing data. More precisely, the English train-
                           T5            91.84 %        99.04 %       99.68 %
                                                                                    ing data contained almost 130 thousand of unique words, the
                       only rules        56.74 %        90.97 %       99.36 %       testing data contained less then 17 thousand and only 5% of

 Czech
                      dict + rules       98.86 %        99.99 %       99.99 %       them were unseen. To compare our results on word level to oth-
                    encoder-decoder      88.64 %        98.69 %       99.51 %       ers, we extracted the unseen words only, for which the error rate
                           T5            98.77 %        99.89 %       99.97 %       was 33.8%. This may seem to be relatively high value, but let’s
                                                                                    keep in mind that the unseen words are also rather rare in our
                                                                                    case (unseen in the whole text set).
                            Word accuracy dependence on word length
                                                                                         The same evaluation for the Czech language, with quite reg-
                  100                                                               ular pronunciation, showed the error rate 2.3% for more than
                                                                  enc/dec
                                                                  only rules        10,000 unseen words from all 78 thousand words in testing
                  90                                              dict+rules        data (the training data contained more than 350 thousand unique
                                                                  T5G2P
                                                                                    words).
                  80




  word accuracy
                                                                                    4.1. Loanwords detection in Czech
                  70
                                                                                    As written in Section 2, the Czech G2P transcription can be very
                                                                                    reliably handled by the relatively small set of rules, except for
                  60
                                                                                    the loanwords which are not that rare in Czech [25]. Thus, all
                                                                                    the errors for dict + rules system in Table 1 are caused by such
                  50
                                                                                    words, not included in the dictionary. Therefore, we explored
                                                                                    an idea to use a trained T5 model to automatically detect the
                  40                                                                words to be added to the exceptions dictionary, or to convert
                        8          10       12            14          16
                                          word length                               such words to the form which can then be correctly transcribed
                                                                                    by the rules (all loanwords can be transcribed to such a form).
Figure 1: Word accuracy comparison for longer words in Czech.                       Although this solution would not be as elegant and universal as
                                                                                    the T5G2P transcriber, it would avoid some of the errors caused
                                                                                    by T5 on cases handled correctly by the rules.
when the rules are used only, i.e. without the dictionary of                             To explore this possibility, we have trained another T5
exceptions. Let us note that the numbers differ slightly from                       model, the task of which is to detect exceptions from the rule-
those presented in the papers mentioned (here the accuracy of                       based transcriptions. The same split of fine-tuning Czech data
baselines is slightly higher), as there were some error correc-                     as described in Section 3.2 has been used, but the phonetic out-
tions made and the dictionaries were extended over the past two                     put was modified in such a way that the words transcribed by the
years.                                                                              rules were replaced by ”0” and the words from the exceptions
     The results clearly show the advantages of the proposed ap-                    dictionary were marked by ”1”. To decide about the exception,
proach. In general, it outperforms the basic encoder-decoder                        a dictionary with more than 170,000 loanwords from the base-
model for both languages in all three evaluation metrics. The                       line system, described in Section 2, has been used.
detailed analysis of the outputs showed that the new proposed                            Similarly to the T5-model tuning described in Section 3.2,
approach makes significantly fewer mistakes in long words,                          the training lasts 50 epochs, but with 1000 steps per epochs
which was the main problem of the previous encoder-decoder                          only. The ability of the model to predict these exceptions
model [11], as demonstrated in Figure 1. (Also, all the exam-                       from regular pronunciation was evaluated on the set of test
ples from section Section 2 were transcribed correctly by our                       data, which contained about 509 thousand words, 3.3% of
T5G2P model.)                                                                       which were the exceptions from the dictionary. Having approx.
     For English, the T5 model is able to beat both of the base-                    44 thousand test sentences, a loanword appears in every sec-
lines – the explanation for that lies in a large number of homo-                    ond/third sentence.
graphs in English which the T5G2P is able to deal with (con-                             The results of loanwords detection are presented in Table 2
trary to the dictionary+rules approach) as showed later in Sec-                     – as the task is actually a classification into two unbalanced
tion 4.2, and also in more accurate transcription of longer words                   classes, in addition to the accuracy we also show precision, re-
(contrary to the encoder-decoder DNN model). For Czech, the                         call and F1-score. High P, R and F1 values indicate that the T5-
accuracy on word and phone level is very close to the base-                         model has learned to recognize exception-marked words very
line dictionary+rules combination. Let us note, however, that                       well. Regarding the errors, there were only 47 false negatives
the baseline has a significant advantage since the rules are well                   (missed exceptions) and 84 false positives (extra detected ex-
tuned and the dictionary has been extended for many years since                     ceptions) in all the testing sentences.
[22].                                                                                    Naturally, we are aware that the dictionary does not contain
     As mentioned in the Section 1, most G2P studies are devel-                     all the possible loanwords and/or their word forms. Thus, some
oped and evaluated on dictionaries, e.g. NetTalk or CMUdict5                        of the words might be marked by ”0” even when they should be
                                                                                    marked by ”1”. To find whether the trained T5 model was able
        5 http://www.speech.cs.cmu.edu/cgi-bin/cmudict                              to detect such words, we have analysed the false positive cases,




                                                                                8
Table 2: Evaluation of the T5 model for loanwords detection on            Table 4: Results of phonetic prediction for representative En-
testing data                                                              glish homographs

         Accuracy     Precision      Recall     F1-score                     homograph         variant      errors   total   Accuracy
         99.97 %       99.51 %       99.72 %     99.62 %                                      [laIv]          0       19       100 %
                                                                                 live
                                                                                               [lIv]          0       48       100 %
Table 3: Results of the tested Czech G2P outputs using loan-                                   [rEd]          2       20        90 %
                                                                                 read
words detection.                                                                              [ri:d]          0       22       100 %
                                                                                            [’rEkOd]          0       19       100 %
        approach            A sent      A word      A phoneme                   record
                                                                                            [ri’kOd]          0       6        100 %
 LW detection + rules      98.31 %      99.71 %       99.94 %
  LW detection + T5        98.93 %      99.90 %       99.98 %

                                                                              – reference: [rEd], predicted: [ri:d]
since these are expected to be cases recognised by T5 model               You read the worst kinds of things about them.
but not included in the dictionary. As an example, the T5 model               – reference: [rEd], predicted: [ri:d]
marked the following words which are not directly contained in            The analysis of the fine-tuning texts used for the training
the dictionary, but derived from items being there:                       showed that there is 6× you read[rEd] and 11× you read[ri:d],
                                                                          1× they read[rEd] and 1× they read[ri:d]. Thus, we tend to
    • neoklasicismu (in dictionary: klasicismus)
                                                                          think that the model learned the more frequent context, since
    • meziresortnı́mi (in dictionary: meziresortnı́)                      it is unable to get a meaning or understanding of the phrase.
     Having a model trained to detect words with irregular pro-           Further investigation is, of course, needed.
nunciation, we tested another approach for phonetic transcrip-
tion of Czech. Contrary to T5 in Table 1, the main idea was                                    5. Conclusions
to detect and highlight loanwords which can be then added
to the dictionary by a user. The whole phrase has, there-                 In the present paper, we focused on the use of T5G2P, the
fore, been transcribed to phones either by the rules (with-               Text-to-Text Transfer Transformer (T5) model in the task of
out dictionary) or by the T5G2P model described in Sec-                   grapheme-to-phoneme (G2P) transcription, which is a natural
tion 3, and the words detected by the T5 detector were ex-                choice due to the similarity of G2P and machine translation
cluded from the comparison, simulating the case of their man-             tasks. The T5 model has been trained on Common Crawl
ual check/transcription/correction by a user. On average, there           data and then fine-tuned on a large set of texts with pho-
were about four such words excluded in every 10 sentences,                netic transcription. For English, it significantly outperforms
which may seem to be a laborious work to check them all, but              both the baseline traditional dictionary+rules and more mod-
the majority of such words are already in the dictionary.                 ern encoder-decoder approaches, and it is also capable to deal
     The results of the modified G2P transcription are shown in           with homographs surprisingly well. For Czech, it outperforms
Table 3. The combination of the loanwords (LW) detection and              the encoder-decoder approach and is very close to the very well
the set of rules led to a significant improvement over the rules-         tuned dictionary+rules G2P module. Let us emphasize that all
only approach in Table 1. The results came, naturally, very close         the G2P transcriptions work on the sentence level, and thus the
to the values for dict+rules approach in Table 1, since all the           cross-word assimilation is taken into account in the results.
loanwords are now treated as transcribed correctly. The table
                                                                               Moreover, the same data were used to fine-tune T5 model
also shows a slight improvement for the T5 model if the LW
                                                                          detecting words with irregular pronunciation in Czech. It has
T5-based detector is used, since the mistakes of the T5G2P in
                                                                          also been shown that a “hybrid” approach, where the irregular
these words are not affecting the results.
                                                                          words are first marked for a manual inspection, and the rest of
                                                                          the text is transcribed “as usual”, can iteratively improve the
4.2. T5 and homographs in English
                                                                          precision of the G2P transcription with high reliability and thus
There are several homographs in English among the words used              without too much labour put into the extension of exceptions
in ordinary communication, e.g. live, read, record. The pro-              dictionary used to handle loanwords with irregular pronuncia-
nunciation of these can be inferred from the context or addi-             tion.
tional meta-information, e.g. some part-of-speech (POS) anal-                 For the future work, we also aim to verify the model on
ysis [26, 27]. Although the presented T5G2P model (Section 3)             other languages, such as Russian, German or Spanish.
does not explicitly use any of such information, it is able to dis-
tinguish between the two pronunciation variants quite reliably,
as shown in Table 4.                                                                      6. Acknowledgements
     To demonstrate it, we selected the above-mentioned homo-
graphs and evaluated the word accuracy measure for all of their           This research was supported by the Czech Science Founda-
transcriptions in the testing data. It can be seen from Table 4           tion (GA CR), project No. GA19-19324S, and by the grant of
that the ability of the model to decide between the pronuncia-            the University of West Bohemia, project No. SGS-2019-027.
tion variants of the homographs is very high.                                 Computational resources were supplied by the project ”e-
     Looking at the failures in “read” transcription, they oc-            Infrastruktura CZ” (e-INFRA LM2018140) provided within the
curred in the following phrases:                                          program Projects of Large Research, Development and Innova-
I am furious that they read her letter, it is like being in prison.       tions Infrastructure.




                                                                      9
                        7. References                                           [16] J. Zelinka and L. Müller, “Automatic general letter-to-sound rules
                                                                                     generation for german text-to-speech system,” in Text, Speech
 [1] J. Matoušek and D. Tihelka, “Annotation errors detection in TTS                and Dialogue, ser. Lecture Notes in Computer Science, P. Sojka,
     corpora,” in Proceedings of INTERSPEECH 2013, Lyon, France,                     I. Kopeček, and K. Pala, Eds., vol. 3206. Berlin, Heidelberg:
     2013, pp. 1511–1515. [Online]. Available: http://www.kky.zcu.                   Springer Berlin Heidelberg, 2004, pp. 537–543.
     cz/en/publications/MatousekJ 2013 AnnotationErrors
                                                                                [17] J. Matoušek, D. Tihelka, and J. Psutka, “New Slovak unit-
 [2] Y. Wang, R. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss,                        selection speech synthesis in ARTIC TTS system,” in Proceed-
     N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio, Q. Le,                         ings of the World Congress on Engineering and Computer Science
     Y. Agiomyrgiannakis, R. Clark, and R. A. Saurous, “Tacotron:                    2011, San Francisco, USA, 2011, pp. 485–490.
     Towards end-to-end speech synthesis,” 2017. [Online]. Available:
     https://arxiv.org/abs/1703.10135                                           [18] J. C. Wells, “SAMPA computer readable phonetic alphabet,” in
                                                                                     Handbook of Standards and Resources for Spoken Language Sys-
 [3] J. Sotelo, S. Mehri, K. Kumar, J. F. Santos, K. Kastner, A. C.                  tems, D. Gibbon, R. Moore, and R. Winski, Eds. Berlin and New
     Courville, and Y. Bengio, “Char2wav: End-to-end speech synthe-                  York: Mouton de Gruyter, 1997.
     sis,” in ICLR, 2017.
                                                                                [19] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,
 [4] Y. Ren, C. Hu, X. Tan, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu,                 Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer
     “Fastspeech 2: Fast and high-quality end-to-end text to speech,”                learning with a unified text-to-text transformer,” 2020. [Online].
     2021.                                                                           Available: arXiv:1910.10683
 [5] K. Rao, F. Peng, H. Sak, and F. Beaufays, “Grapheme-to-                    [20] J. Švec, J. Lehečka, P. Ircing, L. Skorkovská, A. Pražák,
     phoneme conversion using long short-term memory recurrent neu-                  J. Vavruška, P. Stanislav, and J. Hoidekr, “General framework for
     ral networks,” 2015 IEEE International Conference on Acoustics,                 mining, processing and storing large amounts of electronic texts
     Speech and Signal Processing (ICASSP), pp. 4225–4229, 2015.                     for language modeling purposes,” Language Resources and Eval-
 [6] K. Yao and G. Zweig, “Sequence-to-sequence neural net                           uation, vol. 48, no. 2, pp. 227–248, 2014.
     models for grapheme-to-phoneme conversion,” CoRR, vol.                     [21] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi,
     abs/1506.00196, 2015.                                                           P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer,
 [7] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.                P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao,
     Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,”               S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, “Transform-
     2017. [Online]. Available: arXiv:1706.03762                                     ers: State-of-the-art natural language processing,” in Proceedings
                                                                                     of the 2020 Conference on Empirical Methods in Natural Lan-
 [8] S. Yolchuyeva, G. Németh, and B. Gyires-Tóth, “Transformer                    guage Processing: System Demonstrations. Online: Association
     Based Grapheme-to-Phoneme Conversion,” in Proc. Interspeech                     for Computational Linguistics, Oct. 2020, pp. 38–45.
     2019, 2019, pp. 2095–2099.
                                                                                [22] J. Matoušek, “Building a new Czech text-to-speech system using
 [9] J. Švec and L. Šmı́dl, “Prototype of czech spoken dialog system               triphone-based speech units,” in Text, Speech and Dialogue, ser.
     with mixed initiative for railway information service,” in Text,                Lecture Notes in Computer Science, P. Sojka, I. Kopeček, and
     Speech and Dialogue, ser. Lecture Notes in Computer Science,                    K. Pala, Eds. Berlin–Heidelberg, Germany: Springer, 2000, vol.
     P. Sojka, A. Horák, I. Kopeček, and K. Pala, Eds. Springer Berlin             1902, pp. 223–228.
     Heidelberg, 2010, vol. 6231, pp. 568–575.
                                                                                [23] P. Roach, English Phonetics and Phonology: A Practical Course.
[10] M. Jůzová, D. Tihelka, and J. Vı́t, “Unified language-independent             Cambridge University Press, 1983.
     DNN-based G2P converter,” in Interspeech 2019, 20th Annual
     Conference of the International Speech Communication Asso-                 [24] P. Machač and R. Skarnitzl, Principles of Phonetic Segmentation.
     ciation, Graz, Austria, 15-19 September 2019, G. Kubin and                      Epocha, 2013.
     Z. Kacic, Eds. ISCA, 2019, pp. 2085–2089.                                  [25] J. Lehečka and J. Švec, “Improving speech recognition by de-
[11] M. Jůzová and J. Vı́t, “Using auto-encoder biLSTM neural                      tecting foreign inclusions and generating pronunciations,” in Text,
     network for Czech grapheme-to-phoneme conversion,” in Text,                     Speech, and Dialogue, I. Habernal and V. Matoušek, Eds. Berlin,
     Speech, and Dialogue - 22nd International Conference, TSD                       Heidelberg: Springer Berlin Heidelberg, 2013, pp. 295–302.
     2019, Ljubljana, Slovenia, September 11-13, 2019, Proceedings,             [26] S. H. Burton, “The parts of speech: A chapter for reference,” in
     ser. Lecture Notes in Computer Science, K. Ekstein, Ed., vol.                   Mastering English Grammar. London: Palgrave Macmillan UK,
     11697. Springer, 2019, pp. 91–102.                                              1984, pp. 115–141.
[12] K. Cho, B. van Merrienboer, C. Gülcehre, D. Bahdanau,                     [27] S. Bird, E. Klein, and E. Loper, “Categorizing and tagging words,”
     F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase rep-                   in Natural Language Processing with Python, 1st ed. O’Reilly
     resentations using RNN encoder-decoder for statistical machine                  Media, Inc., 2009, pp. 179–220.
     translation.” in EMNLP, A. Moschitti, B. Pang, and W. Daele-
     mans, Eds. ACL, 2014, pp. 1724–1734.
[13] Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi,
     W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey,
     J. Klingner, A. Shah, M. Johnson, X. Liu, Łukasz Kaiser,
     S. Gouws, Y. Kato, T. Kudo, H. Kazawa, K. Stevens, G. Kurian,
     N. Patil, W. Wang, C. Young, J. Smith, J. Riesa, A. Rudnick,
     O. Vinyals, G. Corrado, M. Hughes, and J. Dean, “Google’s
     neural machine translation system: Bridging the gap between
     human and machine translation,” CoRR, vol. abs/1609.08144,
     2016. [Online]. Available: http://arxiv.org/abs/1609.08144
[14] X. Liu, K. Duh, L. Liu, and J. Gao, “Very deep transformers
     for neural machine translation,” 2020. [Online]. Available:
     https://arxiv.org/abs/2008.07772
[15] D. Tihelka, Z. Hanzlı́ček, M. Jůzová, J. Vı́t, J. Matoušek, and
     M. Grůber, “Current state of text-to-speech system ARTIC: A
     decade of research on the field of speech technologies,” in Text,
     Speech, and Dialogue, ser. Lecture Notes in Computer Science.
     Springer International Publishing, 2018, vol. 11107, pp. 369–378.




                                                                           10
