    Read to Hear: A Zero-Shot Pronunciation Assessment Using Textual
                         Descriptions and LLMs

                       Yu-Wen Chen Melody Ma Julia Hirschberg
               Department of Computer Science, Columbia University, United States
                                 yuwchen@cs.columbia.edu



                      Abstract                             why a particular score was assigned. Collecting
                                                           more informative and descriptive feedback, such
    Automatic pronunciation assessment is typi-            as detailed comments from human raters, can be
    cally performed by acoustic models trained on          time-consuming and expensive.
    audio-score pairs. Although effective, these
                                                              Recently, Large Language Models (LLMs) have
    systems provide only numerical scores, with-
    out the information needed to help learners un-        gained popularity for their ability to generate natu-
    derstand their errors. Meanwhile, large lan-           ral, context-aware responses. We propose that this
    guage models (LLMs) have proven effective              generative capability can be leveraged to produce
    in supporting language learning, but their po-         explainable feedback in pronunciation assessment,
    tential for assessing pronunciation remains un-        going beyond simple scoring. Furthermore, LLMs
    explored. In this work, we introduce TextPA,           have demonstrated the potential to provide valuable
    a zero-shot, Textual description-based Pronun-
                                                           insights into language learning (C Meniado, 2023).
    ciation Assessment approach. TextPA utilizes
    human-readable representations of speech sig-          Most studies focus on the use of LLMs in writ-
    nals, which are fed into an LLM to assess pro-         ing tasks (Lo et al., 2024), but LLMs also capture
    nunciation accuracy and fluency, while also            knowledge of language speaking, as humans have
    providing reasoning behind the assigned scores.        documented their knowledge about pronunciation
    Finally, a phoneme sequence match scoring              in written form to facilitate sharing and teaching. In
    method is used to refine the accuracy scores.          addition, previous studies have shown that LLMs,
    Our work highlights a previously overlooked            such as GPT, have the potential to interpret tex-
    direction for pronunciation assessment. Instead
                                                           tual descriptions of speech signals. In (Wang et al.,
    of relying on supervised training with audio-
    score examples, we exploit the rich pronun-            2023), researchers wrote the pause durations in a
    ciation knowledge embedded in written text.            sentence – e.g., “it (<10 ms) is (<10 ms) nothing
    Experimental results show that our approach            (10 ms–50 ms) like (<10 ms) this,” – and put the
    is both cost-efficient and competitive in perfor-      sentence into GPT to assess whether the pauses
    mance. Furthermore, TextPA significantly im-           are correct. However, this study focused only on
    proves the performance of conventional audio-          detecting inappropriate pauses using duration in-
    score-trained models on out-of-domain data by
                                                           formation, without exploring the ability of LLMs
    offering a complementary perspective.
                                                           to interpret other key dimensions of pronunciation,
1   Introduction                                           such as articulation or intonation.
                                                              To bridge the gap between the textual under-
Automatic pronunciation assessment offers an alter-        standing of LLMs and the physical acoustic signal,
native to traditional language instruction by provid-      audio-language models (ALMs) (Elizalde et al.,
ing learners with accessible, scalable, and timely         2023; Tang et al., 20234; Chu et al., 2023) have
feedback on their speaking abilities. Most prior           emerged. ALMs integrate audio and text by en-
work in this area relies on supervised learning: col-      coding audio into audio tokens, which are then
lecting speech recordings annotated with pronun-           processed by the LLM with text tokens. How-
ciation scores from human instructors and training         ever, most open-source ALMs are pre-trained on
acoustic models to assess proficiency scores (Chen         audio captioning or speech recognition datasets and
et al., 2024; Gong et al., 2022). Although effec-          show limited ability to assess speech without fine-
tive, models trained on audio-score pairs provide          tuning (Deshmukh et al., 2024; Wang et al., 2025b).
only numerical scores, offering little insight into        In addition, due to computational constraints, these
                                                        2682
      Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2682–2694
                          November 4-9, 2025 ©2025 Association for Computational Linguistics
studies used smaller LLMs (e.g., 7B or 13B Llama),          (ASR) model; pause information and the recog-
limiting their ability to fully leverage LLM capabil-       nized CMU sequence are derived from a phonetic
ities. On the other hand, commercial large ALMs             aligner; and the IPA phoneme sequence is gener-
such as GPT-audio and Gemini-audio have demon-              ated using a phoneme recognition model. These
strated the potential for pronunciation assessment          textual representations are then provided as input to
in zero-shot settings (Wang et al., 2025a), but these       an LLM, which is prompted to assess the pronuncia-
ALMs are costly to operate with an audio input.             tion and produce both accuracy and fluency scores,
Since audio tokens are much more expensive than             along with the reasoning behind its evaluations.
text tokens1 and the number of audio tokens gen-            Lastly, IPA match scoring is introduced to further
erated from a speech signal can be much greater             refine the accuracy score. Figure 1 presents an
than the number of text tokens in its corresponding         overview of TextPA, which operates in a zero-shot
transcript, using a large ALM with audio inputs is          setting by leveraging pre-trained acoustic models
considerably more expensive than using LLM with             and LLMs, and thus does not require audio-score
text inputs.                                                paired pronunciation data for training.
   Therefore, we explore an alternative method to
bridge the gap between LLM’s textual knowledge              2.1     Textual Acoustic Cues for LLM Input
and physical speech signals. Instead of relying on          2.1.1 Transcript
audio tokens, our method uses the existing capa-            A transcript lacking semantic coherence may result
bilities of LLMs by selecting text-based acoustic           from inaccurate recognition due to poor pronunci-
descriptors common in written text. Pre-trained             ation. Repeated words within a sequence or filler
acoustic models are used to generate these, includ-         words such as “hmm,” can indicate a lack of fluency.
ing transcripts, phoneme sequences (in both Inter-          In Case study A (Figure 2), the speaker is told to
national Phonetic Alphabet (IPA) and CMU Pro-               say “his head hurts even worse,” but their pronun-
nouncing Dictionary (CMU) formats), and pause               ciation is highly inaccurate. Except for "His.", all
durations. The descriptors are provided as input            other words received only 3 out of 10 points. Due
to LLMs for pronunciation assessment. Lastly, we            to poor pronunciation and lack of fluency, the ASR
incorporate a similarity score between the recog-           model produced an inaccurate transcript (i.e., “His
nized IPA sequence and the canonical IPA sequence           hand hands very well”) which is semantically in-
mapped from the transcript to improve the assess-           coherent, signaling low pronunciation proficiency
ment of pronunciation accuracy.                             for the LLM, as reflected in its reasoning. How-
   The contributions of this work are summarized            ever, since ASR model is designed to recognize
as follows: (1) We propose TextPA, a zero-shot              words rather than analyze pronunciation, it may au-
pronunciation assessment model that uses textual            tomatically correct inaccurately pronounced words
descriptions of speech signals. (2) Our method              to produce a semantically coherent sentence. For
produces interpretable and explainable feedback,            example, in Figure 3, the speaker is instructed to
unlike conventional pronunciation assessment sys-           say “maybe we should get some cake” but mis-
tems that yield only numeric scores. In addition,           pronounced “cake.” Although the pronunciation is
incorporating TextPA enhances the performance               inaccurate, the ASR transcript (“maybe we should
of an audio-score-trained model on out-of-domain            get some cards,”) is still semantically reasonable.
data. (3) Compared to large ALMs, our approach              As a result, the transcript alone is insufficient to
significantly reduces API costs while delivering            reveal the finer details of articulation. To address
competitive or superior assessment performance.             this, we incorporate the IPA and CMU phoneme
                                                            sequences that explicitly represent spoken sounds.
2   TextPA
                                                            2.1.2   Recognized IPA and CMU Phoneme
To assess English pronunciation in terms of accu-                   Sequence
racy and fluency, textual acoustic cues are extracted       IPA, widely used in linguistics, dictionaries, and
using a set of pre-trained models: the transcript           language education materials, is a standardized
is obtained from an automatic speech recognition            phonetic notation system that represents the sounds
   1
                                                            of spoken language using a consistent set of sym-
     For example, the OpenAI GPT-4o-mini-audio model
charges $10.00 per 1M audio tokens, compared to $0.15 per   bols. Each symbol corresponds to a specific speech
1M text tokens (as of April 2025).                          sound, providing a one-to-one mapping between
                                                        2683
                                                                                                                                                                      Mapped
                                                                                         Transcript                                                               (canonical) IPA
                                  Automatic speech                                                                                                    Word to
    Speech                          recognition
    signal
                                                                                         maybe we should get some cards

                                                                                         Phonemes (Recognized) CMU
                                                                                                                                                       IPA
                                                                                                                                                      mapping              /
                                                                                                                                                                  m eɪ b iː w iː ʃ ʊ d
                                                                                                                                                                  ɡ ɛ t s ʌ m k ɑːɹ d z

       /                            Phonetic aligner
                                     (CMU-based)                                          M EH M B IY W IY SH UH D (0.12s pause)
                                                                                          G EH T S AH M (0.21s pause) K AH T

                               Phoneme recognition                                       Phonemes (Recognized) IPA                                                 IPA match
                                  (IPA-based)                                             m ɛ m b i w iː ʃ ʊ d ɡ ɛ s s ʌ m k ɑː t
                                                                                                                                                                     scoring
                                                                                                                                                                        norm
       You are an expert evaluator of English pronunciation. Assess the accuracy and fluency of the given text input on a scale of 1 to 5,
       with higher scores indicating better performance. A score of 5 represents native-speaker-level proficiency.
       Input format:
       {"Transcript": "<Recognized ASR sentence>",
                                                                                                                                             Accuracy
                                                                                                                                                /          norm         Mean
        "Phonemes_CMU": "<Recognized CMU pronouncing phoneme sequence, with (time.s pause) indicating pauses in speech.>",                    (LLM)
        "Phonemes_IPA": "<Recognized IPA pronouncing phoneme sequence.>” }

       Task: Return a dictionary with the following format:
       {"Accuracy": <the assessment accuracy score>,
                                                                                                                                                                    Accuracy
        "Fluency": <the assessment fluency score>,
        "Reasoning": <detailed reasoning for the assigned score>}                                                                             LLM
       Note: Do not include any other text other than the json object.
       Input:                                                                                                                                                         Fluency
            Transcript:   maybe we should get some cards
                                                                                                                                                /
            Phonemes CMU: M EH M B IY W IY SH UH D (0.12s pause) G EH T S AH M (0.21s pause) K AH T
            Phonemes IPA: m ɛ m b i w iː ʃ ʊ d ɡ ɛ s s ʌ m k ɑː t
                                                                                                                                                                    Reasoning


                                                                               Figure 1: An overview of TextPA.


 sound and notation. The CMU phoneme sequence                                                                             phoneme sequence and compares it with the pro-
 is a phonetic transcription format based on the                                                                          vided recognized phoneme sequence. Although
 Carnegie Mellon University Pronouncing Dictio-                                                                           LLMs are capable of this, as shown in Case study
 nary (CMUdict). Unlike IPA, which is universal                                                                           B (Figure 3) where the model correctly identifies
 in language and more fine-grained, CMU uses a                                                                            the mispronunciation of the word “cards”, they
 simplified set of phonemes tailored for American                                                                         may still overlook some errors. For example, in the
 English, which is widely used in speech process-                                                                         same case, a discrepancy is observed between the
 ing applications due to its compatibility with ASR                                                                       canonical phoneme sequence for the word “maybe”
 systems and phoneme-based models. Because both                                                                           (m eI b i: / M EY B IY) and the recognized se-
 representations are widely used, LLMs trained on                                                                         quence (m E m b i / M EH M B IY), indicating
 extensive text corpora have encountered and in-                                                                          inaccurate pronunciation. Although the human
 ternalized the mapping between IPA and CMU                                                                               annotation assigns a score of 10 out of 10 to the
 phoneme annotations and the word. For exam-                                                                              pronunciation accuracy of “maybe”, our manual
 ple, in Case study B (Figure 3), by comparing the                                                                        inspection suggests that the word is not clearly ar-
 recognized IPA and CMU sequences, the LLM                                                                                ticulated. However, the LLM does not reflect this
 identifies that the word “cards” may have been mis-                                                                      error in its reasoning.
 pronounced and uses this information to assess pro-
 nunciation accuracy. It can align transcript words                                                                          To further refine accuracy assessment, we in-
 with the corresponding phoneme sequences even                                                                            troduce IPA match scoring, which measures the
 when word boundaries are not explicitly marked.                                                                          similarity between the recognized and canonical
 We also embed pause information from the pho-                                                                            IPA sequences and uses this as an indicator of pro-
 netic aligner into the recognized CMU phoneme                                                                            nunciation accuracy. We use IPA instead of CMU
 sequence. Pauses are annotated in an easily inter-                                                                       because IPA offers more fine-grained phonetic de-
 pretable format, e.g. “D (0.12s pause) G” indicates                                                                      tail. In addition, our empirical results suggest that
 a 0.12-second pause between the phones “D” and                                                                           match scoring using IPA consistently outperforms
“G”. As shown in Case study B (Figure 3), the LLM                                                                         scoring with CMU phonemes. To perform IPA
 leverages this pause information when reasoning                                                                          match scoring, the transcript is first mapped to its
 about the speaker’s fluency.                                                                                             canonical IPA phoneme sequence using a pronun-
                                                                                                                          ciation dictionary. We then compute the similar-
2.2 IPA Match Scoring                                                                                                     ity between the transcript-mapped and the recog-
                                                                                                                          nized IPA sequences using the Smith-Waterman
To assess pronunciation, the LLM internally                                                                               algorithm (Smith et al., 1981). This algorithm is
maps each word in the transcript to its canonical                                                                         a dynamic programming method originally devel-
                                                                                                                2684
                           Case study A                                                  Case study B

         Ground-truth          his head hurts even worse                Ground-truth      maybe we should get some cake
           (Human             (10, 3,     3, 3,      3, )                 (Human          ( 10, 10, 10, 10, 10, 3,)
          annotation)                                                    annotation)     *incorrect

          LLM input                                                     LLM input
          Transcript      his hand hands very well                       Transcript      maybe we should get some cards
          Phonemes        HH IH Z HH AE N D HH AE N                      Phonemes        M EH M B IY W IY SH UH D
            CMU           D Z (0.05s pause) ER N EH R IH                   CMU           (0.12s pause) G EH T S AH M
                          IY R IH IY AH L                                                (0.21s pause) K AH T
          Phonemes        hɪzhændhændsmɛɹiɹɪ                             Phonemes        m ɛ m b i w iː ʃ ʊ d ɡ ɛ s s ʌ m k
             IPA                                                            IPA          ɑː t

         Mapped IPA       hɪzhændhændzvɛɹiwɛl                           Mapped IPA       m eɪ b iː w iː ʃ ʊ d ɡ ɛ t s ʌ m k
                                                                                         ɑːɹ d z
        LLM reasoning
        The accuracy is low due to several mispronunciations           LLM reasoning
        and inaccuracies in the phoneme recognition. "very             Accuracy: The pronunciation is generally accurate,
        well" is completely different from "ER N EH R                  with most phonemes being correctly produced. The
        IH IY R IH IY AH L" and "m ɛ ɹ i ɹ ɪ".                         vowel in 'cards' is slightly off. It appears to be
        The word "hands" is repeated and there's a pause.              truncated to 'cot' sound rather than 'cards'. Still
        The fluency is slightly affected by the pause and the          quite understandable.
        unnatural sequence of words.                                   Fluency: The presence of pauses affects the
                                                                       fluency score. (0.12s) is acceptable, (0.21s) pause
                                                                       in a short sentence is a bit long. The pace is
Figure 2: Case study A. Due to the inaccurate pro-                     slightly uneven overall. More natural speech would
nunciation, the ASR model produced unnatural word                      exhibit smoother transitions between words.
sequences, which in turn signaled to the LLM that the
pronunciation was flawed.
                                                                   Figure 3: Case study B. The transcript is semantically
                                                                   reasonable despite inaccurate pronunciation. To address
oped for local alignment of biological sequences                   this, the mismatch between the recognized phoneme se-
such as DNA or proteins. It identifies the most                    quence and the transcript provides the LLM with insight
similar subsequences between two inputs, making                    into potential articulation inaccuracies. The mapped
                                                                   IPA (i.e., the canonical IPA of the transcript) is shown
it well-suited for phoneme sequence comparison,
                                                                   for reference and is not provided as input to the LLM.
which consists of subsequences representing differ-
ent words. Since the IPA-matched scores and the
LLM-generated accuracy scores operate on differ-                   and prosody. The utterances in Speechocean are
ent scales, we apply min-max normalization across                  scripted. Participants were instructed to read prede-
the test set to align them within a common range.                  fined sentences, making the ground-truth transcript
The final accuracy score is then calculated as the av-             available. However, our method operates without
erage of the normalized Smith-Waterman similarity                  the need for ground-truth information. Most sen-
and the normalized LLM-generated score.                            tences in Speechocean are short, as shown in Fig-
                                                                   ure 1, 2, and 3, with corresponding audio dura-
3       Experimental Setup                                         tions ranging from 2 to 20 seconds. Since TextPA
3.1 Data and Evaluation Metric                                     requires no training, we used only the Speechocean
                                                                   test set, which contains 2,500 utterances.
We evaluated TextPA on the open-source Spee-                          The MultiPA data contains 50 audio clips, each
chocean762 (Zhang et al., 2021) and Mul-                           ranging from 10 to 20 seconds in duration, col-
tiPA (Chen et al., 2024) datasets 2 , both of which                lected from ~20 anonymous users interacting with
focus on English speech produced by native Man-                    a dialogue-based chatbot. Unlike Speechocean,
darin speakers. The Speechocean762 (abbreviated                    where speakers are asked to read predefined sen-
as Speechocean) dataset consists of 5,000 utter-                   tences, MultiPA data captures open-ended re-
ances spoken by 250 speakers, with annotations at                  sponses, allowing learners to speak freely or answer
the sentence, word, and phoneme levels. In this                    questions. This allows for a more authentic assess-
study, we focus on sentence-level accuracy, fluency,               ment of learners’ speaking abilities. Table 1 shows
    2
        License: Attribution 4.0 International (CC BY 4.0)         example transcriptions from both datasets. We use
                                                                2685
    Speechocean Two, four, seven.                                    apply min-max normalization to each model’s out-
                  It was good for me.                                puts. The final prediction is obtained by averaging
        MultiPA   I'm an active person and I enjoy playing a         the normalized scores. Despite the simplicity of
         data     variety of sports. One of my favorite              this fusion strategy, the combined model achieves
                  sports to play is basketball as it is a great
                  way to stay fit and socialize with friends         notable performance improvement over using ei-
                  at the same time.                                  ther model alone. This improvement is likely due
                  I often go to the zoo. I think the zoo is a        to the distinct sources of information. MultiPA is
                  very interesting place. And I go, I went to        trained on paired audio-score data, learning directly
                  the zoo once a week now.
                                                                     from acoustic examples, whereas TextPA operates
                                                                     solely on text and leverages prior knowledge about
Table 1: Example transcriptions from Speechocean
and MultiPA. Speechocean consists of relatively short,               pronunciation assessment. Differing approaches
scripted utterances from read-aloud tasks, whereas Mul-              offer diverse perspectives, enabling the combined
tiPA data captures open-ended, conversational speech.                system to achieve improved performance.
                                                                        Due to the limited amount of paired audio-score
the Pearson correlation coefficient (PCC) as the                     pronunciation data, MultiPA may have difficulty ac-
main evaluation metric since it has often been used                  curately assessing words that were not encountered
in prior studies and provides better interpretability                during training. In contrast, TextPA has access to a
when comparing performance on different datasets.                    much broader vocabulary, leading to higher perfor-
                                                                     mance on accuracy assessment. However, because
3.2 Implementation Details                                           MultiPA analyzes raw audio recordings, it can cap-
We use Whisper (Radford et al., 2023) (large-v3-                     ture acoustic cues such as detailed phone-level du-
en) for transcription, the model from (Xu et al.,                    rations or pitch variations. These cues are typically
2021)3 for IPA sequence, Charsiu (Zhu et al., 2022)                  not represented in written descriptions or are dif-
predictive aligner for CMU sequence, and Phone-                      ficult to capture accurately in text, making them
mize (Bernard and Titeux, 2021)4 for word-to-IPA                     challenging for LLMs to interpret. In fact, we also
mapping. Acoustic models were run on an NVIDIA                       explore the LLM’s ability to assess prosody using
RTX 4500 GPU. The LLMs use default API set-                          ToBI annotations (Beckman and Hirschberg, 1994)
tings, and results are from a single run.                            which offer a text-based representation of tonal pat-
                                                                     terns and phrase boundaries. However, the LLM
4       Results                                                      appears to struggle with assessing prosody by ac-
                                                                     curately interpreting these annotations, even when
4.1 Performance on Free-speech
                                                                     given explicit instructions (see the Appendix B for
Table 2 shows the performance on MultiPA data.                       details). In essence, the two approaches provide
We compare TextPA with different LLM back-                           complementary advantages on the assessment task,
ends. Since TextPA (gpt-4o-mini) outperforms                         and combining them could be beneficial by lever-
TextPA (gemini-2.0-flash), we used GPT-4o-mini-                      aging the strengths of both.
audio for the performance comparison. Results
suggest that the proposed TextPA outperforms GPT-
4o-mini-audio in assessing pronunciation, achiev-                                                Accuracy     Fluency
ing better performance in both accuracy and flu-                               TextPA
                                                                                                   0.697       0.557
ency. We also compare performance with the Mul-                           (gemini-2.0-flash)
tiPA model (Chen et al., 2024), an acoustic model                              TextPA
                                                                                                   0.728       0.650
trained on Speechocean. Results show that the                               (gpt-4o-mini)
proposed TextPA achieves higher accuracy and pro-                        GPT-4o-mini-audio         0.674       0.648
vides competitive fluency assessment, showing the
                                                                          MultiPA model            0.618       0.683
effectiveness of TextPA in a zero-shot setting.
   We evaluate the performance of combining the                            MultiPA model +
                                                                                                   0.769       0.784
MultiPA and TextPA models. To account for differ-                        TextPA (gpt-4o-mini)
ences in the scale of their prediction scores, we first
                                                                     Table 2: Model performance on MultiPA data. Note that
    3
     https://huggingface.co/facebook/                                MultiPA model was trained on Speechocean.
wav2vec2-lv-60-espeak-cv-ft
   4
     EspeakBackend("en-us")

                                                                  2686
4.2 Performance on Scripted Utterances                                                        Accuracy     Fluency
Table 3 shows the performance on Speechocean.                                          Zero-shot
We first compare the performance of TextPA us-                              TextPA
ing different LLM back-ends. Results indicate that                                                 0.507    0.466
                                                                         (gpt-4o-mini)
gemini-2.0-flash outperforms gpt-4o-mini; there-                            TextPA
fore, we conducted another experiment using                                                        0.532    0.557
                                                                       (gemini-2.0-flash)
Gemini-2.0-flash-audio for our performance com-
parison. In contrast to its strong performance on the               Gemini-2.0-flash-audio         0.562    0.556
MultiPA dataset, TextPA performs relatively poorly                                    In-domain
on Speechocean. This discrepancy might arise                        (Lin and Wang, 2022)        0.72          -
from fundamental differences between the datasets.                    (Liu et al., 2023b)         -         0.795
Speechocean consists of shorter, more constrained
                                                                       MultiPA model           0.705        0.772
utterances (as shown in Table 1), which offer lim-
ited phonetic and semantic variation. Moreover,                        Table 3: Model performance on Speechocean.
Speechocean prompts students to repeat predefined
sentences, unlike the MultiPA data, which includes
free-form speech. As a result, both the pause                    nized CMU sequence through normalized Smith-
cues between words and the semantic content of                   Waterman similarity scores. However, the results
the transcripts offer weaker indicators of language              indicate that the CMU sequence is less effective
proficiency, thereby reducing the effectiveness of               for accuracy assessment compared to the IPA se-
TextPA. These dataset differences may also explain               quence. This difference may stem from the greater
the performance inconsistency between Gemini                     phonetic detail provided by the IPA, which contains
and GPT across the two datasets. Nevertheless,                   more than 107 syllable letters, while the CMU set
TextPA remains competitive on Speechocean. Note                  contains only 39 phonemes.
that TextPA relies solely on text tokens, whereas                   Table 4 also reports an ablation study evaluat-
Gemini-2.0-flash-audio uses text tokens for instruc-             ing which textual descriptions of acoustic cues are
tions and audio tokens for input speech signals5 .               most effective for language models in pronuncia-
We also include in-domain models’ performance as                 tion assessment. When using an LLM, the tran-
references. Since TextPA is a zero-shot approach                 script alone can offer insights. Augmenting the
without using training data, the in-domain models                input with recognized IPA sequences improves per-
naturally perform better. Directly combining the                 formance, particularly in accuracy, as the LLM can
predictions as done with MultiPA data does not                   compare word transcriptions with their phonetic
lead to improvements for the in-domain setting due               transcriptions to better identify mispronunciations.
to the performance gap. Further investigation is                 Adding CMU sequences alongside the transcript
needed to explore more effective ways of leverag-                helps to enhance both accuracy and fluency as well:
ing TextPA for in-domain models.                                 accuracy improves for similar reasons as with IPA,
                                                                 while fluency benefits from the pause information
4.3    Ablation Study on Textual Descriptions of                 encoded in CMU sequences. Overall, combining
       Speech Signals                                            the transcript, CMU, and IPA sequences leads to
First, we evaluated the performance of accuracy                  the best performance, with IPA match scoring pro-
scoring based on phoneme sequence matching (Ta-                  viding additional boosts in accuracy.
ble 4). Our findings demonstrate that IPA match                  4.4    Impact of ASR Transcription Quality
scoring is a straightforward yet highly effective
method for assessing pronunciation accuracy. We                  Transcripts play a crucial role in TextPA. To ex-
also investigated the performance of CMU match                   amine the affect of ASR model quality (i.e., tran-
scoring. Similar to IPA match scoring, the words                 scription quality), we compared LLM-based assess-
in the transcript are mapped to CMU labels using                 ment using transcripts generated by two Whisper
the dictionary, and then compared with the recog-                variants: large-v3-en (denoted as large-en) and
                                                                 tiny. The large-en model, with 1550M parame-
   5
    The cost of gemini-2.0-flash is 0.1 per 1M text tokens and   ters, is English-only and generates higher-quality
$0.7 per 1M audio tokens, making Gemini-2.0-flash-audio
approximately 3.5 times more expensive in API calls than         transcripts that are more robust to inaccurate pro-
running TextPA (Gemini-2.0-flash) on the Speechocean.            nunciation. In contrast, the tiny model, with only
                                                             2687
                    MultiPA data                           ability (i.e., tiny) can understand you easily, it indi-
                           Accuracy        Fluency         cates that your pronunciation is good.
          TextPA                                              Although the transcripts from tiny models per-
                               0.728        0.650          form better on their own, the large-en model is
       (gpt-4o-mini)
                                                           more effective within the TextPA framework. In
        LLM: all               0.643        0.650
                                                           TextPA, we incorporate the IPA and CMU se-
     LLM: trans.+cmu           0.491        0.485          quences along with the transcript. Inaccurate pro-
     LLM: trans.+ipa           0.452        0.410          nunciation can lead to unnatural IPA and CMU se-
      LLM: transcript          0.404        0.432          quences, offering similar insights to the transcript
    IPA match scoring          0.653           -           of tiny model. In addition, because the transcript
                                                           serves as a baseline for comparison, excessive ASR
   CMU match scoring           0.208           -
                                                           errors introduce noise that reduces reliability. Over-
                    Speechocean                            all, we believe that a stronger ASR model, such
                           Accuracy        Fluency         as large-en, is the better choice within the TextPA
        TextPA                                             structure.
                               0.532        0.557
    (gemini-2.0-flash)
                                                                                       Accuracy          Fluency
        LLM: all               0.456        0.557                                  large-en   tiny   large-en   tiny
     LLM: trans.+cmu           0.427        0.553                                   MultiPA data
     LLM: trans.+ipa           0.448        0.458                LLM: all
                                                                                    0.643    0.569    0.650    0.546
                                                               (gpt-4o-mini)
      LLM: transcript          0.313        0.310
                                                              LLM: transcript       0.404   0.556     0.432    0.442
    IPA match scoring          0.507           -                                    Speechocean
   CMU match scoring           0.263           -                  LLM: all
                                                                                    0.456    0.481    0.557    0.523
                                                              (gemini-2.0-flash)
Table 4: Ablation study of text-based acoustic cues. We        LLM: transcript      0.313    0.409    0.310    0.431
selected the LLM with the best performance on each
dataset as the representative model: gpt-4o-mini for the         Table 5: Impact of ASR transcription quality.
MultiPA data and gemini-2.0-flash for the Speechocean
data. LLM: transcript uses only the transcript as input.
LLM: trans.+ ipa and trans.+ cmu add IPA or CMU            4.5    Analysis of Basic vs. Detailed Scoring
sequences, respectively. LLM: all combines all three              Guidelines
inputs: transcript, IPA, and CMU. Note that the fluency
scores for LLM: all and TextPA are identical, as IPA       We investigated the impact of providing different
score matching is only used to refine accuracy.            instructions to the LLM, including basic and de-
                                                           tailed scoring guidelines (Table 6). The basic scor-
                                                           ing guideline prompts the LLM to assign a scoring
39M parameters and multilingual training, is more          range (1-5), where a higher score indicates better
likely to produce transcription errors or misclassify      pronunciation, with a score of 5 reflecting native-
English as a different language when pronunciation         speaker proficiency. The detailed scoring guideline,
is inaccurate.                                             on the other hand, provides the same detailed anno-
   As shown in Table 5, when transcripts alone are         tation guidelines used by human annotators. The
used as input to the LLM, tiny yields better assess-       detailed guidelines define the language proficiency
ment results than large-en. This observation can           for each score level. For example, for MultiPA
be illustrated through an analogy: using large-en          data, an accuracy score of 5 means “Excellent:
is like speaking to a listener with excellent English      The overall pronunciation is nearly perfect with
comprehension – they can understand you even if            accurate articulation of all sounds,” while a score
your pronunciation is poor. In contrast, the tiny          of 4 means “Good: Minor pronunciation errors
model resembles a listener with limited English            may be present, but overall, the pronunciation is
ability, who can only understand clearly articulated       highly accurate and easily understandable”, and
speech. Whether a person with strong English lis-          so on. Results suggest that the effectiveness is
tening comprehension (i.e., large-en) can under-           dataset-dependent, possibly influenced by how the
stand you provides less insight into your pronunci-        guidelines are written. However, incorporating a
ation. In contrast, if people with weaker listening        detailed scoring guideline has the potential to re-
                                                       2688
duce performance, while also lengthening the input                       lucination, correct, constructive, and irrelevant.
text prompt and increasing model operating costs.                        Hallucination refers to cases where the reason-
                                                                         ing clearly misrepresents the audio, such as stat-
                             Accuracy               Fluency
                                                                         ing “closely matches standard native speaker ar-
                          Basic Detailed        Basic Detailed
                                                                         ticulation” when the pronunciation is clearly non-
                           MultiPA data
        LLM: all
                                                                         standard. Correct indicates reasoning that aligns
                          0.643      0.500      0.650      0.543         with the audio but does not provide actionable de-
      (gpt-4o-mini)
        LLM: all                                                         tails; for example, “The accuracy score of 3 re-
                          0.554      0.596      0.556      0.499
    (gemini-2.0-flash)                                                   flects a moderate level of pronunciation correct-
                            Speechocean                                  ness. While there are identifiable phonetic errors,
        LLM: all                                                         the core message is still comprehensible.” Con-
                          0.420      0.474      0.466      0.544
      (gpt-4o-mini)                                                      structive reasoning identifies specific areas for im-
        LLM: all
                          0.456      0.470      0.557      0.561         provement, such as “**calls us** is recognized
    (gemini-2.0-flash)
                                                                         as **cars**” Finally, irrelevant refers to reasoning
Table 6: Performance with basic or detailed guidelines.                  that is unrelated to pronunciation, such as com-
                                                                         ments on grammar or the transcript.

5     Analysis of TextPA Reasoning
Table 7 presents examples of TextPA reasoning.                              We then measured the coverage of each category
We observed that gemini-2.0-flash is more likely                         in the TextPA-generated reasoning (Figure 4). Cov-
to refer to the detailed CMU and IPA phoneme                             erage was determined by tokenizing the reasoning
sequences to support its assessment, while gpt-4o-                       descriptions and calculating the proportion of to-
mini tends to provide more generalized observa-                          kens belonging to each category. For MultiPA data,
tions. Also, gpt-4o-mini often frames its evaluation                     53% of gemini-2.0-flash’s generated descriptions
as verifying whether the phoneme sequence aligns                         relate to accuracy and 44% to fluency, while 40%
with the pronunciation, focusing on the “phoneme                         of gpt-4o-mini’s descriptions relate to accuracy
sequence” or the “ASR system” rather than com-                           and 43% to fluency. Gemini-2.0-flash allocates a
menting directly on pronunciation.                                       greater proportion of content to accuracy than to flu-
                                                                         ency, whereas gpt-4o-mini’s content is more evenly
                             Transcript                                  split. The rest contains irrelevant reasoning or gen-
     I want to find more about you.                                      eral overviews of pronunciation proficiency. For
                             gpt-4o-mini                                 Speechocean, we randomly selected 25 samples
     The accuracy score is 3 because there are noticeable                for annotation. Compared to MultiPA data, both
     issues with the CMU and IPA phoneme sequences                       LLMs place considerably greater emphasis on accu-
     that deviate from a standard pronunciation. For
     instance, 'more' is represented incorrectly, and there              racy than on fluency on Speechocean, with 68% vs.
     are additional phonemes that do not correspond well                 31% for Gemini, and 50% vs. 41% for GPT. This
     to the transcript. However, the overall attempt to
     convey the sounds aligns with understandable speech.                difference is likely due to the shorter utterances
     The fluency score is 4, reflecting a reasonably smooth              in Speechocean, which provide limited material
     delivery despite pauses, suggesting a good pace and
     rhythm in the speech, although the mispronunciations                to observe natural speech flow or identify disrup-
     do affect the overall fluency slightly.                             tions, making fluency assessment less feasible. On
                          gemini-2.0-flash                               both datasets, gemini-2.0-flash generally provides
     The accuracy is reasonable but not perfect. Several                 more constructive reasoning compared to gpt-4o-
     phoneme errors are present compared to a native
     speaker pronunciation. Specifically, the pronunciation              mini. For both LLMs, constructive reasoning oc-
     of **to find more about you** has noticeable                        curs more frequently for accuracy than for fluency,
     deviations (e.g., 'f eh n d er' instead of 'f ai n d m oo r').
     The schwa insertions are frequent. The fluency is                   likely because accuracy is more clearly defined and
     slightly affected by the pauses and some                            can be evaluated more objectively. Overall, roughly
     mispronunciations, making the speech sound a bit
     hesitant.                                                           76% of the context in gpt-4o-mini-based TextPA
                                                                         reasoning is either correct or constructive, while
          Table 7: Examples of TextPA reasoning.                         over 90% of gemini-2.0-flash-based reasoning falls
                                                                         into these categories, highlighting the strong poten-
  To gain deeper insights, we manually annotated                         tial of TextPA to generate meaningful descriptive
the TextPA reasoning into four categories: hal-                          feedback for pronunciation assessment.
                                                                      2689
    gemini
                         Accuracy                        Fluency                               model. However, like other previous studies, it
     -2.0-      12.9%            36.4%                 29.6%         13.8%                     provided only numerical feedback instead of more
                                                                              MultiPA
     flash          Accuracy                  Fluency                                          interpretable or explainable assessments.
    gpt-4o-             26.6%                    41.3%
     mini
                                Accuracy                         Fluency
                                                                                               6.2    LLM for Language Learning
    gemini


                                                                              Speechocean
     -2.0-
     flash
                 18.3%                47.4%                      27.0%                         LLMs have had a significant impact on education,
                         Accuracy                      Fluency
                                                                                               with many studies exploring how tools like Chat-
    gpt-4o-             28.2%        14.9%               32.7%
     mini                                                                                      GPT can support language learning (Lo et al., 2024;
          0%            20%         40%          60%           80%         100%                C Meniado, 2023). These models have proven
               A-hallucination       A-correct             A-constructive                      effective in helping learners identify and correct
               F-hallucination       F-correct             F-constructive
               Irrelevant            Other                                                     writing errors, improve the quality of their writ-
                                                                                               ing (Barrot, 2023), and receive automated feed-
                                                                                               back (Mizumoto and Eguchi, 2023). Few studies
Figure 4: Coverage analysis of TextPA reasoning. “A”
denotes accuracy, and “F” denotes fluency.                                                     have focused on using LLMs to support speaking
                                                                                               skills. (Kim and Park, 2023) used ChatGPT as a
                                                                                               conversational partner in role-playing tasks, while
6     Background                                                                               (Lee et al., 2023) used it to generate topics for oral
6.1 Speech Pronunciation Assessment                                                            practice. A study by (Wang et al., 2023) used Chat-
                                                                                               GPT to assess how well ESL learners placed pauses
Speech pronunciation assessment models can be                                                  in their speech. However, the potential of LLMs to
categorized into closed- or open-response scenar-                                              support other aspects of oral language skills, such
ios. In closed-response settings, learners read a pre-                                         as pronunciation accuracy and fluency as in TextPA,
determined sentence, which serves as the ground-                                               remains underexplored.
truth transcript for the model to guide the assess-
ment. A common approach in this scenario ex-                                                   7     Conclusion
tracted Goodness of Pronunciation (GoP) features
to train an acoustic model (Gong et al., 2022; Do                                              We propose TextPA, a zero-shot pronunciation as-
et al., 2023). In addition to GoP, various other                                               sessment method that leverages interpretable, tex-
features have been explored for model training, in-                                            tual representations of speech signals to assess pro-
cluding acoustic embeddings from self-supervised                                               nunciation accuracy and fluency. These descrip-
learning (SSL) models, prosodic features such as                                               tions include transcripts, IPA, and CMU phoneme
duration and energy, and transcript-based features                                             sequences, collectively reflecting pronunciation
such as word embeddings (Chao et al., 2022; Yan                                                characteristics. Specifically, semantically unnat-
et al., 2025). In (Wu et al., 2025), researchers fine-                                         ural transcripts may signal pronunciation issues,
tuned an LLM using audio tokens and text prompts                                               mismatches between canonical and recognized
to provide feedback on phone errors. However,                                                  phoneme sequences reflect articulation errors, and
the performance of models trained with ground-                                                 inappropriate pauses embedded in CMU sequences
truth transcripts may degrade significantly when                                               reveal disfluencies. Experimental results demon-
such transcripts are unavailable. On the other hand,                                           strate that LLMs can effectively leverage textual
open-response scenarios allow learners to speak                                                description of speech to assess different aspects of
freely or respond to prompts, enabling a more                                                  pronunciation. Unlike conventional models trained
authentic evaluation of their pronunciation skills.                                            on audio-score pairs, TextPA operates without su-
Models designed for open-response tasks do not                                                 pervision. TextPA focuses on human-readable rep-
rely on ground-truth transcripts. Instead, they lever-                                         resentations and prior knowledge of pronunciation,
age ASR outputs or avoid ASR entirely (Lin and                                                 aiming to provide interpretable and explainable
Wang, 2021; Kim et al., 2022; Chen et al., 2024;                                               feedback that go beyond a score. We hope this work
Liu et al., 2023b). Most prior studies rely on audio-                                          offers a new perspective on pronunciation assess-
score pair data to train acoustic models for pronun-                                           ment. Building on our initial exploration, future re-
ciation assessment, whereas zero-shot approaches                                               search could further develop methods to more effec-
have been largely unexplored. In (Liu et al., 2023a),                                          tively integrate TextPA with audio-trained models,
researchers scored pronunciation based on the num-                                             combining their strengths to improve assessment
ber of incorrectly recovered tokens from an SSL                                                accuracy and feedback quality for learners.
                                                                                            2690
Limitations                                            undermining the rich diversity of global English
                                                       accents.
While prosody is an important aspect of pronun-
ciation, we found it difficult to effectively assess
using our text-based approach. Compared to accu-       References
racy and fluency, prosodic features such as rhythm     Jessie S Barrot. 2023. Using ChatGPT for second lan-
and intonation are harder to describe precisely in        guage writing: Pitfalls and potentials. Assessing
written form, making them less suitable for meth-        Writing, 57:100745.
ods that rely solely on textual representations. As    Mary E Beckman and Julia Hirschberg. 1994. The tobi
a result, the LLM struggled to reliably evaluate        annotation conventions. Ohio State University.
prosody without compromising assessment perfor-
mance on accuracy and fluency. In addition, both       Mathieu Bernard and Hadrien Titeux. 2021. Phonem-
                                                        izer: Text to phones transcription for multiple lan-
the LLM and the ASR system introduce variabil-          guages in python. Journal of Open Source Software,
ity across runs, leading to inconsistent assessment     6(68):3958.
results. In addition, budget constraints limited our
ability to use the most advanced LLMs or to eval-      Joel C Meniado. 2023. The impact of ChatGPT on
                                                         english language teaching, learning, and assessment:
uate large ALMs across all settings. Lastly, the         A rapid review of literature. Arab World English
LLM occasionally produces hallucinations or con-         Journals (AWEJ) Volume, 14.
tent irrelevant to the reasoning. While most outputs
                                                       Fu-An Chao, Tien-Hong Lo, Tzu-I Wu, Yao-Ting Sung,
align with the audio, providing more practically
                                                         and Berlin Chen. 2022. 3M: An effective multi-
actionable feedback could better support learners.       view, multi-granularity, and multi-aspect modeling
Exhaustive manual review of reasoning results is         approach to english pronunciation assessment. In
beyond the scope of this study, and no established       Proc. APSIPA ASC 2022, pages 575–582. IEEE.
metric currently exists to automatically verify cor-   Yu-Wen Chen, Zhou Yu, and Julia Hirschberg. 2024.
rectness. Further investigation is needed to deter-      MultiPA: a multi-task speech pronunciation assess-
mine the conditions under which the LLM is more          ment model for open response scenarios. In Proc.
likely to generate errors and to develop strategies      INTERSPEECH 2024, pages 297–301.
that both prevent such errors and enhance action-      Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shil-
able feedback. These limitations suggest future          iang Zhang, Zhijie Yan, Chang Zhou, and Jingren
work in prosody modeling, dataset expansion, and         Zhou. 2023. Qwen-audio: Advancing universal
automatic reasoning evaluation.                          audio understanding via unified large-scale audio-
                                                         language models. arXiv preprint arXiv:2311.07919.
   Although certain words may have multiple valid
pronunciations depending on the speaker’s accent,      Soham Deshmukh, Dareen Alharthi, Benjamin Elizalde,
our study did not consider accent variation, since       Hannes Gamper, Mahmoud Al Ismail, Rita Singh,
                                                         Bhiksha Raj, and Huaming Wang. 2024. PAM:
the majority of the data involved attempts to mimic      Prompting audio-language models for audio quality
General American English. Consequently, a po-            assessment. In Proc. INTERSPEECH 2024.
tential risk of this study is an overemphasis on a
single accent. While many English learners aim to      Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023.
                                                         Hierarchical pronunciation assessment with multi-
emulate native speakers, the more practical goal in      aspect attention. In Proc. ICASSP 2023. IEEE.
everyday communication is to express one’s opin-
ions clearly and be understood. This highlights        Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Is-
the importance of balancing pronunciation assess-        mail, and Huaming Wang. 2023. CLAP: Learning
                                                         audio concepts from natural language supervision. In
ment systems between intelligibility and nativeness.     Proc. ICASSP 2023. IEEE.
When such systems overemphasize native-like pro-
nunciation, which is often tied to a specific ac-      Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang,
                                                         and James Glass. 2022. Transformer-based mmulti-
cent, they might erroneously mark understandable
                                                         aspect multi-granularity non-native english speaker
speech as “wrong.” Failing to strike this balance        ppronunciation assessment. In Proc. ICASSP 2022,
can marginalize learners’ linguistic identities and      pages 7262–7266. IEEE.
encourage unnecessary accent reduction at the ex-
                                                       Eesung Kim, Jae-Jin Jeon, Hyeji Seo, and Hoon Kim.
pense of communicative effectiveness. In addition,       2022. Automatic pronunciation assessment using
an overly narrow model can reinforce the idea that       self-supervised speech representation learning. In
only a single variety of English is valid, thereby       Proc. INTERSPEECH 2022, pages 1411–1415.
                                                   2691
Sol Kim and Seon-Ho Park. 2023. Young korean EFL          Siyin Wang, Wenyi Yu, Yudong Yang, Changli Tang,
  learners’ perception of role-playing scripts: ChatGPT     Yixuan Li, Jimin Zhuang, Xianzhao Chen, Xiaohai
  vs. textbooks. Journal of English Language and            Tian, Jun Zhang, Guangzhi Sun, and 1 others. 2025b.
  Linguistics, 23:1136–1153.                                Enabling auditory large language models for auto-
                                                            matic speech quality evaluation. In Proc. ICASSP
Hyungmin Lee, Chen-Chun Hsia, Aleksandr Tsoy,               2025. IEEE.
  Sungmin Choi, Hanchao Hou, and Shiguang Ni.
  2023. VisionARy: Exploratory research on contex-        Zhiyi Wang, Shaoguang Mao, Wenshan Wu, Yan Xia,
  tual language learning using AR glasses with Chat-        Yan Deng, and Jonathan Tien. 2023. Assessing
  GPT. In Proceedings of the 15th biannual conference       phrase break of ESL speech with pre-trained lan-
  of the Italian SIGCHI chapter, pages 1–6.                 guage models and large language models. In Proc.
                                                            INTERSPEECH 2023, pages 4194–4198.
Binghuai Lin and Liyuan Wang. 2021. Deep feature
  transfer learning for automatic pronunciation assess-   Minglin Wu, Jing Xu, Xueyuan Chen, and Helen Meng.
  ment. In Proc. INTERSPEECH 2021, pages 4438–              2025. Integrating potential pronunciations for en-
  4442.                                                     hanced mispronunciation detection and diagnosis
                                                            ability in llms. In Proc. ICASSP 2025. IEEE.
Binghuai Lin and Liyuan Wang. 2022. Exploiting in-        Qiantong Xu, Alexei Baevski, and Michael Auli. 2021.
  formation from native data for non-native automatic       Simple and effective zero-shot cross-lingual phoneme
  pronunciation assessment. In Proc. SLT 2022, pages        recognition. In Proc. INTERSPEECH 2021, pages
  708–714. IEEE.                                            2113–2117.
Hongfu Liu, Mingqian Shi, and Ye Wang. 2023a. Zero-       Bi-Cheng Yan, Yi-Cheng Wang, Jiun-Ting Li, Meng-
  shot automatic pronunciation assessment. In Proc.         Shin Lin, Hsin-Wei Wang, Wei-Cheng Chao, and
  INTERSPEECH 2023, pages 1009–1013.                        Berlin Chen. 2025. ConPCO: Preserving phoneme
                                                            characteristics for automatic pronunciation assess-
Wei Liu, Kaiqi Fu, Xiaohai Tian, Shuju Shi, Wei Li,         ment leveraging contrastive ordinal regularization.
 Zejun Ma, and Tan Lee. 2023b. An ASR-free fluency          In Proc. ICASSP 2025. IEEE.
 scoring approach with self-supervised learning. In
 Proc. ICASSP 2023. IEEE.                                 Junbo Zhang, Zhiwen Zhang, Yongqing Wang, Zhiy-
                                                            ong Yan, Qiong Song, Yukai Huang, Ke Li, Daniel
Chung Kwan Lo, Philip Leung Ho Yu, Simin Xu, Davy           Povey, and Yujun Wang. 2021. Speechocean762: An
  Tsz Kit Ng, and Morris Siu-yung Jong. 2024. Explor-       open-source non-native English speech corpus for
  ing the application of ChatGPT in ESL/EFL educa-          pronunciation assessment. In Proc. INTERSPEECH
  tion and related research issues: a systematic review     2021, pages 3710–3714.
  of empirical studies. Smart Learning Environments,
  11(1):50.                                               Jian Zhu, Cong Zhang, and David Jurgens. 2022. Phone-
                                                             to-audio alignment without text: A semi-supervised
Atsushi Mizumoto and Masaki Eguchi. 2023. Exploring          approach. In Proc. ICASSP 2022, pages 8167–8171.
  the potential of using an AI language model for auto-      IEEE.
  mated essay scoring. Research Methods in Applied
  Linguistics, 2(2):100050.                               A    Prompt
Alec Radford, Jong Wook Kim, Tao Xu, Greg Brock-          Figure 5 shows the TextPA prompt for LLM; ALM
  man, Christine McLeavey, and Ilya Sutskever. 2023.      prompt follows a similar format, but does not in-
  Robust speech recognition via large-scale weak su-      clude input format instructions. We observed that
  pervision. In Proc. ICML 2023, pages 28492–28518.
                                                          Gemini is more likely to return results that do not
Temple F Smith, Michael S Waterman, and 1 others.         match the required format, whereas GPT tends to
  1981. Identification of common molecular subse-         produce outputs that can be directly saved as JSON
  quences. Journal of molecular biology, 147(1):195–      files. If the model fails to generate a correctly for-
  197.
                                                          matted output for a given test sample, we re-run it
Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao            until a valid result is obtained.
  Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao
  Zhang. 20234. SALMONN: Towards generic hear-            B   Prosody assessment
  ing abilities for large language models. Proc. ICLR
  2024.                                                   We investigate whether LLM could assess prosody
                                                          from textual descriptions. We only used the Mul-
Ke Wang, Lei He, Kun Liu, Yan Deng, Wenning               tiPA data for this part of the study, as most sen-
  Wei, and Sheng Zhao. 2025a. Exploring the poten-
  tial of large multimodal models as effective alterna-
                                                          tences in Speechocean are short and do not contain
  tives for pronunciation assessment. arXiv preprint      sufficient prosodic variation for a reliable assess-
  arXiv:2503.11229.                                       ment. First, we prompted the LLM to evaluate
                                                      2692
 You are an expert evaluator of English pronunciation.          index and the tone index, both of which are crucial
 Assess the accuracy and fluency of the given text              for understanding the prosody of speech signals.
 input on a scale of 1 to 5, with higher scores indicating      The break index ranges from 0 to 4 and is defined
 better performance. A score of 5 represents native-
                                                                as follows:
 speaker-level proficiency.                                         0: Clear phonetic marks for clitic groups
 Input format:                                                      1: Most phrase-medial word boundaries
 {"Transcript": "<Recognized ASR sentence>",                        2: Strong disjuncture, pause or virtual
 "Phonemes_CMU": "<Recognized CMU pronouncing                             pause, no tonal marks
 phoneme sequence, with (time.s pause) indicating                   3: Intermediate intonation phrase bound-
 pauses in speech.>",
                                                                          ary
 "Phonemes_IPA": "<Recognized IPA pronouncing
 phoneme sequence.>"}                                               4: Full intonation phrase boundary
                                                                   The tone index includes the following categories:
 Task: Return a dictionary with the following format:               H:          High pitch in the local pitch
 {"Accuracy": <the assessment accuracy score>,                                  range
 "Fluency": <the assessment fluency score>,                         L:          Low pitch in the local pitch
 "Reasoning": <detailed reasoning for the assigned
                                                                                range
 score>}
                                                                    * :         Pitch accent, indicating that the
 Note: Do not include any other text other than the json                        word is stressed
 object.                                                            %:          The end of an intonation phrase
 Input:                                                             - or ––: A phrase’s accent
                                                                   Table 9 presents a selection of examples from
                  Figure 5: LLM prompt.                         our attempts to assess prosody using an LLM. The
                                                                experimental results indicate that the LLM is less
prosody in addition to accuracy and fluency. As                 effective in assessing prosody, and requiring it to do
shown in Table 8, the model performs worse in                   so leads to a decline performance in accuracy and
terms of prosody assessment compared to fluency                 fluency. A possible reason for this is that prosody is
and accuracy. In addition, introducing prosody as               harder to capture accurately using textual descrip-
an additional assessment criterion leads to a de-               tions. Since prosody is less commonly expressed in
crease in the model’s performance in both accuracy              written form, the LLM has more difficulty leverag-
and fluency.                                                    ing its inherent knowledge for prosody assessment.

                      Accuracy     Fluency     Prosody
    LLM: all
                        0.633       0.678           -
  (gpt-4o-mini)
    LLMp : all
                        0.590       0.549        0.243
  (gpt-4o-mini)

Table 8: LLM performance with and without prosody
assessment. LLMp : all is LLM: all with the introduction
of prosody as an additional assessment criterion. Note
that the transcript is generated using turbo version of
Whisper, an optimized version of large-v3 that provides
faster transcription with minimal loss in accuracy. The
results indicate that turbo performs comparably to large-
v3-en. (Section 4.1)

   We explore textual descriptions of prosody us-
ing annotations from the ToBI (Tones and Break
Indices) system (Beckman and Hirschberg, 1994)6 ,
which provides a standardized approach to annotate
intonation and phrasing patterns in spoken English.
ToBI includes two primary components: the break
   6
       https://github.com/monikaUPF/PyToBI

                                                             2693
  Index   Accuracy   Fluency   Prosody              Prompt                         Textual description of prosody

 LLMA      0.467      0.561     0.294    ToBI_sequence":                  "L-L% !H* L-L% L* L* H*+L L+H* L-H%
                                         "<Recognized ToBI                L+H* L* L* L-L% L* H* L* L*+H L-H%
                                         sequence.>                       H*+L L* L-L% H-L% L-L% L* H* H-L% L*
                                                                          L*+H LH- L*"

                                                                          (Note: raw ToBI tone indices.)

 LLMB      0.545      0.500     0.172    "Prosody_annotated_text":        "depends (%) i mean it depends (*, %) on (*)
                                         "<Sequence of ASR-               what (*) i'm looking (*) for (*, %) if i'm (*)
                                         recognized words with            going to buy (*, %) like (*) a phone or (*)
                                         prosodic labels. '*' indicates   computer (*, %) i would definitely (*) choose
                                         a pitch accent, and '%'          big ones (*, %) because (%) the (%) quality
                                         indicates a phrase               (*) of the product (%) is more (*) reliable (*, -
                                         boundary. Labels appear in       -) for sure (*)"
                                         parentheses after the
                                         corresponding word."             (Note: Simplified ToBI tone indices,
                                                                          including pitch accents, phrase accents, and
                                                                          boundary tones, are provided along with the
                                                                          corresponding words in the transcript.)

 LLMC      0.494      0.617     0.231    "Prosody_annotated_text":        "depends (%). i mean it depends (*). on (*)
                                         "<Sequence of ASR-               what (*) i'm looking, for (*). if i'm (*) going
                                         recognized words with            to buy (*). like (*) a phone or (*) computer. i
                                         prosodic labels. '*' indicates   would     definitely,    choose      big   ones
                                         a pitch accent, '--' indicates   (*). because (%). the (%). quality (*) of the
                                         a phrase accent, and '%'         product (%). is more (*) reliable, for sure (*)"
                                         indicates a phrase
                                         boundary. Labels appear in       (Note: Simplified ToBI tone indices are used.
                                         parentheses after the            Only the final tone index for each word is
                                         corresponding word."             considered.)

 LLMD      0.593      0.604     0.353    "Prosody_annotated_text":        "depends (--,%).... i mean it depends (*).... on
                                         "<Sequence of ASR-               (*) what (*) i'm looking (*).. for (*).... if i'm (*)
                                         recognized words with            going to buy (*).... like (*) a phone or (*)
                                         prosodic labels. '*' indicates   computer (*).... i would definitely (*).. choose
                                         a pitch accent, '--' indicates   big ones (*).... because (--,%).... the (--,%)....
                                         a phrase accent, and '%'         quality (*) of the product (--,%).... is more (*)
                                         indicates a phrase               reliable (*)... for sure (*)"
                                         boundary. Labels appear in
                                         parentheses after the            (Note: Simplified ToBI tone indices are used.
                                         corresponding word."             Break index information is represented by
                                                                          the number of dots, with more dots ("....")
                                                                          indicating a longer break.)

 LLME      0.539      0.680    0.3043    "Transcript_prosody":            "depends ....i mean it depends ....on what i'm
                                         "<Sequence of ASR                looking ..for ....if i'm going to buy ....like a
                                         recognized word with             phone or computer ....i would definitely ..choose
                                         prosody information.>"           big ones ....because ....the ....quality of the
                                                                          product ....is more reliable ...for sure"


Table 9: LLM performance in the presence of textual prosody descriptions. The Prompt column displays the
additional instructions given to the LLM, beyond the standard prompt shown in Figure 5. The Textual Description
of Prosody column illustrates an example input provided to the LLM.




                                                         2694
