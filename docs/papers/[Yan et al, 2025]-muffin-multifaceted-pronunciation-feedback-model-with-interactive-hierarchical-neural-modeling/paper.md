                                                                                                                                    1
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <


  MuFFIN: Multifaceted Pronunciation Feedback
Model with Interactive Hierarchical Neural Modeling
               Bi-Cheng Yan, Student Member, IEEE, Ming-Kang Tsai, and Berlin Chen, Member, IEEE


                                                                     reference for professionals (e.g., interviewers and examiners) in
Abstract—Computer-assisted pronunciation training (CAPT)             high-stakes assessments, with the goals of reducing the
manages to facilitate second-language (L2) learners to practice      workload [3][4], alleviating the burdens of recruiting new
pronunciation skills by offering timely and instructive feedback.    human experts, and achieving consistent and objective
To examine pronunciation proficiency from multiple facets,
existing methods for CAPT broadly fall into two categories:
                                                                     assessment results [5][6][7].
mispronunciation detection and diagnosis (MDD) as well as                A de-facto archetype system for CAPT is normally
automatic pronunciation assessment (APA). The former aims to         instantiated in a read-aloud scenario, where an L2 learner is
pinpoint phonetic pronunciation errors and provide diagnostic        provided with a reference text and instructed to pronounce it
feedback, while the latter seeks instead to quantify pronunciation   correctly. By taking the learner’s speech paired with the
proficiency pertaining to various aspects. Despite the natural       reference text as input, CAPT systems are anticipated to assess
complementarity between MDD and APA, researchers and
practitioners, however, often treat them as independent tasks with
                                                                     the learner’s oral competence from multiple facets, providing
disparate modeling paradigms. In light of this, we in this paper     detailed and potentially diagnostic performance feedback with
first introduce MuFFIN, a Multi-Faceted pronunciation Feedback       a near-instant turnaround. To this end, mispronunciation
model with an Interactive hierarchical Neural architecture, to       detection and diagnosis (MDD) and automatic pronunciation
jointly address the tasks of MDD and APA. To better capture the      assessment (APA) are two active strands of research in
nuanced distinctions between phonemes in the feature space, a        developing pronunciation feedback modules for CAPT. The
novel phoneme-contrastive ordinal regularization mechanism is
then put forward to optimize the proposed model to generate more
                                                                     former seeks to pinpoint phonetic pronunciation errors and
phoneme-discriminative features while factoring in the ordinality    provides L2 learners with the corresponding diagnostic
of the aspect scores. In addition, to address the intricate data     feedback [8][9]. The latter, in contrast, concentrates more on
imbalance problem in MDD, we design a simple yet effective           assessing the learner’s pronunciation quality through multi-
training objective, which is specifically tailored to perturb the    faceted pronunciation scores, reflecting his/her proficiency
outputs of a phoneme classifier with the phoneme-specific            pertaining to specific aspects or some extent of spoken language
variations, so as to better render the distribution of predicted
phonemes meanwhile considering their mispronunciation
                                                                     usage [10][11]. One time-tested approach for MDD is goodness
characteristics. A series of experiments conducted on the            of pronunciation (GOP) and its derivatives [12][13], which
Speechocean762 benchmark dataset demonstrates the efficacy of        calculate the ratio between the likelihoods of the canonical and
our method in relation to several cutting-edge baselines, showing    most likely pronounced phonemes. Phoneme-level erroneous
state-of-the-art performance on both the APA and MDD tasks.          pronunciations are subsequently detected if the likelihood ratios
                                                                     of certain phoneme segments fail below predetermined
Index Terms—Computer-assisted pronunciation training,
                                                                     thresholds. On a separate front, the models of iconic APA
automatic pronunciation assessment, mispronunciation detection
and diagnosis, multi-aspect and multi-granular pronunciation         methods are typically trained to mimic human ratings based on
assessments, contrastive learning.                                   surface features (viz. a set of hand-crafted features). These
                                                                     models either employ a classifier to predict a holistic score
                                                                     representing learners’ oral proficiency [10] or use regressors to
                      I. INTRODUCTION                                estimate continuous analytic scores for specific pronunciation



F
       UELED by the amplified demand for foreign language            aspects, such as phoneme-level accuracy [14], word-level
       acquisition, research on computer-assisted pronunciation      lexical stress [15], and utterance-level pronunciation quality
       training (CAPT) has aroused significant attention amidst      [16][17].
the tide of globalization, figuring prominently in the field of          In spite of the complementary nature of MDD and APA,
computer-assisted language learning (CALL) [1][2]. To bridge         most existing efforts treat them as independent tasks, thereby
the gap between insufficient supplies and pressing needs from        developing two disparate feedback modules for use in CAPT.
language teachers and learners, CAPT systems have emerged            However, some prior studies reveal that an L2 English learner
as appealing learning tools ubiquitously, shifting the               tends to have lower utterance-level assessment scores of
conventional pedagogical paradigm from teacher-led to self-          intelligibility and fluency [18] whenever his or her utterances
directed learning. Beyond their critical roles in education and      frequently contain phoneme-level pronunciation errors
language learning, CAPT systems also serve as a handy                [19][20]. In the view of this, we in this paper first propose a
                                                                     novel CAPT modeling paradigm, dubbed MuFFIN, which is a
                                                                                                                                                                   2
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <




                                                                               (a) Naïve Training Process           (b) Training with Phoneme-specific Variation

                                                                                Feature                            Phoneme
                                                                                Region      Many    Medium   Few   Category      AH      EH       UH

                                                                        Fig. 2. The motivation of the proposed phoneme-specific
                                                                        variation. In the feature space, each point represents a data
                                                                        instance predicted by a phoneme classifier, with different
                                                                        colors indicating distinct categories. (a) The naïve training
                                                                        process tends to bias toward majority phoneme categories,
                                                                        leading to the compression of minority phoneme categories
  Fig. 1. Data imbalance problem of MDD exists in                       into a narrow area. (b) By applying the proposed phoneme-
  Speechocean762 dataset ( Many-shot,            Medium-shot,           specific variation training strategy, the feature spaces of
  and      Few-shot), where the frequencies of correct                  minority phoneme categories expand, achieving a more
  pronunciations are notably higher than those of                       balanced feature distribution while incorporating
  mispronunciations. Moreover, the correct and incorrect                pronunciation difficulty to modulate feature areas.
  pronunciations exhibit two distinct long-tailed distributions.
                                                                       many-shot in mispronunciations. Similarly, the vowel /UW/ is
Multi-Faceted pronunciation Feedback model with an                     found in the many-shot region of correct pronunciations but
Interactive hierarchical Neural structure. MuFFIN unifies the          shifts to the few-shot region of mispronunciations.
individual feedback modules of MDD and APA into a                          Typically, a naïve training process of a phoneme-level
streamlined, hierarchical neural architecture through a multi-         pronunciation classifier for MDD is susceptible to an
task learning scheme. Building on a language hierarchy aware           undesirable bias toward correct pronunciations due to their
neural architecture with the tailor-made convolution-                  higher occurrence frequency, which therefore dominate the
augmented Branchformer blocks, MuFFIN can effectively                  entire training process [22]. As a remedy, our proposed strategy
capture interactions across the linguistic granularities (i.e.,        for modeling phoneme-specific variations is built around the
phoneme, word, and utterance) and preserve fine-grained                hypothesis that logits of phoneme categories with higher
articulatory cues at different linguistic units. Next, to render the   occurrence counts (viz. majority phoneme categories) may
subtle differences between phonemes in the feature space, we           occupy a larger portion of the feature space, whereas those of
introduce a novel phoneme-contrastive ordinal regularizer to           with lower occurrence counts (viz. minority phoneme
facilitate the proposed model in generating more phoneme-              categories) are compressed into a narrower region [23], as
discriminative features. This training regime leverages                depicted in Fig. 2(a). The proposed training strategy augments
contrastive learning to better align the phoneme representations       the logits of phoneme predictions with randomly sampled
of a scoring model with the textual embeddings of their                Gaussian noise, where the radius is determined by the proposed
corresponding canonical phonemes, while also accounting for            phoneme-specific variation. To address the intricate data
the ordinal relationships among the regression targets (i.e.,          imbalance problem of MDD, the modeling of phoneme-specific
phoneme-level accuracy scores). Furthermore, a simple yet              variations comprises two complementary factors: a quantity
effective training objective, phoneme-specific variation, is           factor and a pronunciation difficulty factor. The former assigns
explored to ease the data imbalance problem incurred by MDD            smaller variances to majority phoneme categories and larger
[21]. Data imbalance is a long-standing problem in MDD,                variances to minority phoneme categories. In contrast, the latter
where phoneme distributions are often skewed between correct           modulates feature areas based on the mispronunciation rates of
and incorrect pronunciation instances. As illustrated in Fig. 1,       phonemes. By doing so, as shown in Fig. 2(b), the synergy of
we demonstrate the distributions of vowels with correct and            the two factors not only balances the feature distributions of
incorrect pronunciations in the training set of the                    disparate phonemes but also adjusts the regions corresponding
Speechocean762 dataset (see Section IV that follows), which            to pronunciation difficulties. In summary, this paper presents a
are further categorized into many-shot, medium-shot, and few-          continuation of our previous work described in [24] and [25]
shot regions based on their occurrence counts. It is evident that      with a significant extension of novel technical contents,
the occurrence of correct pronunciations significantly exceeds         experiments, and analysis, whose main contributions are at least
that of mispronounced ones. Compounding this issue, correct            four-fold:
and incorrect pronunciations exhibit unique long-tailed trends,        • We present MuFFIN, a multi-faceted pronunciation
respectively. For instance, the vowels /EH/ and /EY/ are                   feedback model that jointly addresses the tasks of MDD and
categorized as medium-shot in correct pronunciations but as                APA through an interactive hierarchical neural framework.
                                                                                                                                 3
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

  This model signifies a paradigm shift from separate             pronunciation scoring-based methods first gauge the
  modeling of APA and MDD to a unified assessment                 pronunciation scores for each phoneme in the canonical
  approach, opening up a new avenue in CAPT.                      phoneme sequence. Mispronounced phoneme segments are
• A contrastive phonetic ordinal regularizer is proposed to       then detected when their scores fall below predetermined
  align the speech-derived phoneme representations with the       thresholds, signifying a deviation from the expected
  corresponding phoneme-level textual embeddings, while           pronunciation. However, pronunciation scoring-based methods
  organically engaging the ordinality of pronunciation            are untenable to provide diagnostic feedback for the detected
  accuracy scores. A series of graphical examinations are         mispronounced phoneme segments. As a remedy, dictation-
  conducted through the lens of the ordinality and phoneme        based methods strive to formulate MDD as a phoneme
  properties.                                                     recognition task by employing a phoneme recognizer to dictate
• To the best of our knowledge, this is the first attempt to      the most likely phoneme sequence uttered by an L2 learner. The
  address data imbalance issues in MDD by incorporating           erroneous pronunciation portions can be easily identified by
  phoneme-specific variations into the training process. Our
                                                                  comparing the dictation result with the corresponding canonical
  method highlights that the data imbalance problem in MDD
                                                                  phoneme sequence. For instance, Leung et al. ventured into
  stems from two intertwined and equally crucial factors, viz.
  the quantity and the pronunciation difficulty of the training   employing a CTC-based phoneme recognizer for L2 English
  data. Our empirical findings reveal that addressing the data-   learners, showing comparative performance with pronunciation
  imbalance problem of MDD by solely considering the data         scoring-based methods in the mispronunciation detection
  quantity factor primarily enhances the recall metric but        subtask, where the performance gains mainly contributed from
  sacrifices the precision metric. This analysis has led us to    the accurate diagnosis of mispronunciations in unvoiced
  propose a training strategy that incorporates a pronunciation   phoneme segments [37]. Yan et al. exploited the hybrid CTC-
  difficulty factor, achieving a better balance between recall    Attention ASR model as the dictation model and sought to
  and precision metrics compared to strategies that consider      capture deviant (non-categorical) phoneme productions uttered
  either the quantity factor or the pronunciation difficulty      by accented L2 learners with anti-phone modeling [38]. Both of
  factor individually.                                            the above-mentioned methods rely on precise alignments to
• Extensive sets of experiments carried out on the                identify mispronounced segments; however, in practical
  Speechocean762 benchmark dataset [26] confirm the               applications, alignment errors might arise when comparing the
  effectiveness of our proposed methods, which improves the       canonical phoneme sequence to accented or disfluent speech
  performance of state-of-the-art ones on both the APA and        produced by L2 learners. In response, prompt-based methods
  MDD tasks.                                                      leverage an attention mechanism to derive the soft alignment
                                                                  between the canonical phoneme sequence and the learner’s
                     II. RELATED WORK                             input speech in an end-to-end manner, offering a promising
    Computer-assisted pronunciation training (CAPT) is a          approach to reduce alignment errors. As one of the first
subfield of computer-assisted language learning (CALL),           attempts, PeppaNet aligns canonical phonemes with the
whose research and development can trace back to pioneering       learner’s speech via a Transformer decoder, where any
efforts in the 1960s [27] and have gained significant attention   discrepancies are captured in the matching degree vectors
recently due to the unprecedented advancements in speech and      through end-to-end neural modeling [39]. Among other things,
language technologies [28][29][30]. According to the              MDDGCN introduces a graph-based prompt encoder for
diagnostic feedback of CAPT, research endeavors typically fall    canonical phonemes, aiming to improve diagnosis accuracy by
into phoneme-level mispronunciation detection and diagnosis       regularizing the relationships between canonical and actually
(MDD) as well as automatic pronunciation assessment (APA),        pronounced phonemes through a pre-defined phonetic graph
both mostly developed under read-aloud learning scenarios.        [40].
A. Mispronunciation Detection and Diagnosis                       B. Automatic Pronunciation Assessment
    Mispronunciation detection and diagnosis (MDD) manages            Automatic Pronunciation Assessment (APA) quantifies an
to detect erroneous pronunciation at phoneme segments, and in     L2 learner’s pronunciation proficiency in a target language by
turn provide L2 learners with the corresponding diagnostic        providing either analytic scores (viz. continuous numerical
feedback [31][32]. Common approaches to MDD can be                values) for specific pronunciation aspects [41][42] or a holistic
grouped into three categories: pronunciation scoring-based,       assessment (viz. discrete categorical values) to reflect overall
dictation-based, and prompt-based methods. Pronunciation          spoken competence [10]. Early efforts in APA predominantly
scoring-based methods typically exploit various types of          focused on single-aspect assessment, typically by constructing
confidence measurements to evaluate pronunciation quality via     individual scoring modules to predict proficiency scores at
a well-trained ASR system (e.g., hybrid DNN-HMM ASR               specific linguistic levels with various sets of hand-crafted
system). Frequently-used measurements include, but are not        features. These hand-crafted features, extracted from the
limited to, phoneme durations [33][34], likelihood ratios [13],   learner's input speech or its corresponding ASR-generated
phoneme posterior probabilities [35], and their combinations
[36]. Given an input utterance and its corresponding canonical
phoneme sequence (viz. phoneme-level text prompt),
                                                                                                                                                                                                                                                                                 4
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <


                                                                                                                                                                                                     Utterance
                          Phoneme-level Textual Embedding (E ! )                                                                                                                              Acc/Flu/Comp/Pros/Total
                                                                                                    Word                   Word               Word
                                                           "                                       [Call]                   [It]             [Bear]
                          Condensed acoustic features (X )                                     Acc/Stress/Total       Acc/Stress/Total   Acc/Stress/Total
                                                                                                                                                                                                          ……
                      Phone      Phone    Phone                      Phone           Phone                                                                                                                                                            Weighted Comb
                        [K]   …    [L]     [IH]   …                    [B]     …       [R]
                     Feedback   Feedback Feedback                   Feedback        Feedback                                                                                             Utterance-level Regressor
                                                                                                                                                                                                                                 …
                                                                                                                  Word-level Regressor                                                                   ……                                                           Dropout
                                                                                                                                                                                                                                            Dropout
                               Phoneme-level Pronunciation Feedback Module                                                                                                                                                                                      Pointwise Conv
                                                                                                                                                                                                                                          Feed-forward
                                                                                                         CNN                    CNN                CNN                                          Atten Pooling                  …                               ReLU Activation
                                                Phoneme Encoder                                                                                                                                                 ……                         LayerNorm
                                                                                                                                                                                                                                                                  LayerNorm
                                                                                                                      Word Encoder                                                            Utterance Encoder



                                                                                                                                                             Weighted Merge
                                                  3            4      5        6     7                                                                                                                                                                          Depthwise Conv
                      0          1      2
                                                                                                                                                                                                                ……
                                                                                                                                                                                                                                            Dropout
                                                                                                                                                                                                                                                                GLU Activation
                                                                                                  Cat & Proj            Cat & Proj         Cat & Proj                                              Cat & Proj
                                                Feature Extraction                                                                                                                                               ……                       Self-Attention        Pointwise Conv

                                                                                               Atten Pooling           Atten Pooling     Atten Pooling                                                                    CNNCNN           LayerNorm              LayerNorm

   Input Audio (X)                                                                                                                                                                                              ……

                          K      AO         L         IH       T          B    EH     R                                                                                                                         ……
                              Word 1 (Call)           Word 2 (It)         Word 3 (Bear)             Phone-level Feature Representations (X ! , H ! )                                                                           $ ")
                                                                                                                                                                              Phone-level and Word-level Features (X ! , H ! , H
   Text Prompt (T)                                                                                                                                                                                                                       (b) Convolution-
            (a) Multi-faceted Pronunciation Feedback Model with an Interactive Hierarchical Neural Architecture (MuFFIN)                                                                                                              augmented Branchformer

  Fig. 3. The proposed multi-faceted pronunciation feedback model with an interactive hierarchical neural architecture
  (MuFFIN). (a) The overall model architecture processes input audio and text prompts, hierarchically representing the learner’s
  pronunciation to generate assessment scores across various aspects. (b) The proposed convolution-augmented Branchformer
  block functions as the backbone of MuFFIN, operating in encoders at different linguistic levels. The pronunciation aspects of
  accuracy, fluency, completeness, and prosody are denoted as Acc, Flu, Comp, and Pros, respectively.

transcript, may include acoustic features, confidence scores of                                                                                              Error states                              Diagnostic Feedback               Accuracy Scores
recognized linguistic units, time-alignment information, and                                                                                                0, 1, 1,…, 0, 0                             K, AA, O,…, EH, R                 2, 0, 0, … , 2,2
statistical measures [43][44]. To scrutinize learners'
pronunciation comprehensively, recent advances in APA have
advocated multi-aspect and multi-granular pronunciation                                                                                                                Sigmoid                                   Softmax
                                                                                                                                                                                                                                           Feed-forward
assessment, leveraging unified scoring models that evaluate                                                                                                 Feed-forward                                      Feed-forward
pronunciation proficiency across multiple linguistic levels (viz.
phoneme, word, and utterance) with diverse aspects (e.g.,                                                                                                    LayerNorm                                          LayerNorm                   LayerNorm
accuracy, fluency, and completeness). Drawing on this research                                                                                              Error Detector                                Diagnosis Predictor           Accuracy Regressor

tend, Gong et al. proposed a parallel pronunciation modeling
architecture dubbed GOPT, which took GOP features as input
and adopted a Transformer encoder as the backbone model to
jointly model multiple pronunciation aspects across various                                                                                 Fig. 4. The proposed phoneme-level pronunciation
linguistic granularities [45]. Following this school of thought,                                                                            feedback module, which takes the output of phoneme
                                                                                                                                            encoder and simultaneously generate phoneme-level error
3M extended GOPT by augmenting the model’s input
                                                                                                                                            states, diagnostic phonemes, and accuracy scores.
embeddings with prosodic features and self-supervised learning
(SSL)-based features, aiming to achieve multi-view, multi-
granularity, and multi-aspect pronunciation modeling [46].                                                                                III. MULTI-FACETED PRONUNCIATION FEEDBACK MODEL WITH
Despite their decent performance, the hierarchical structure of                                                                              AN INTERACTIVE HIERARCHICAL NEURAL ARCHITECTURE
an utterance is largely set aside. To capture the language                                                                                    The overall architecture of the proposed MuFFIN is
hierarchy of an utterance, Do et al. proposed a hierarchical APA                                                                          schematically depicted in Fig. 3(a), which contains three main
model and explored a novel multi-trait attention layer to                                                                                 components: phoneme-level modeling, word-level modeling,
strengthen the connection between scoring aspects [47]. Chao                                                                              and utterance-level modeling. The encoder at each different
et al. introduced sub-phoneme modeling and employed a depth-                                                                              linguistic level adopts a novel convolution-augmented
wise separable convolution layer to construct a hierarchical                                                                              Branchformer block [25], as shown in Fig. 3(b), which consists
APA model, facilitating better modeling of local context cues                                                                             of two branches with one branch designed to capture supra-
at the sub-word level [48]. Apart from the above, Gradformer                                                                              segmental pronunciation cues with multi-head attention (MHA)
leveraged a granularity-decoupled Transformer network that                                                                                layers while the other tailored to capture fine-grained
first separates the granularity of an utterance into lower-level                                                                          pronunciation cues with a series of convolution layers.
(phoneme- and word-level) ones and higher-level (utterance-                                                                               Furthermore, as illustrated in Fig. 4, a novel phoneme-level
level) one. A Conformer encoder in turn jointly models                                                                                    pronunciation feedback module is devised to assess phoneme-
                                                                                                                                          level accuracy and perform mispronunciation detection and
pronunciation aspects at the lower levels, while a Transformer
                                                                                                                                          diagnosis.
decoder processes a set of trainable aspect vectors and interacts
with the encoder outputs for utterance-level pronunciation
assessment [42].                                                                                                                          A. Problem Formulation
                                                                                                                                                  Given an input utterance U, consisting of a time sequence
                                                                                                                                             5
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

of audio signals X produced by an L2 learner, and a reference              where Linear𝑑𝑒𝑡 (⋅) is a linear layer followed by layer
text prompt T with 𝑀 words, which is converted into 𝑁                      normalization. The diagnostic predictor performs a sequential
canonical phonemes based on a pronunciation dictionary1, the               multi-class labeling process to derive the probability
proposed multi-faceted pronunciation feedback model aims to                distribution of diagnostic feedback for the 𝑛 -th canonical
estimate proficiency scores at various linguistic granularities,           phoneme as:
while pinpointing phoneme-level pronunciation errors for the                     P𝑑𝑖𝑎𝑔 (𝑦𝑛̂ ∣𝐪, 𝑋) = Softmax (Linear𝑑𝑖𝑎𝑔 (𝐡𝑝𝑁 )),  (5)
canonical phoneme sequence. Formally, let G = {𝑝, 𝑤, 𝑢} be a
                                                                           where Linear𝑑𝑖𝑎𝑔 (⋅) used to convert hidden dimensions into the
set of linguistic granularities, where 𝑝, 𝑤, 𝑢 stands for the
                                                                           size of pronunciation dictionary. Finally, the phoneme-level
phoneme-, word-, and utterance-level, respectively. For each
                                                                           accuracy score is estimated by an accuracy score regressor.
granularity 𝑔 ∈ 𝐺, our model aims to predict a set of aspect               Word-level Modeling. For the word-level assessments, a
score sequences A𝑔 = {𝐚1𝑔 , 𝐚2𝑔 , … , 𝐚𝑁
                                       𝑔
                                         𝑔
                                           } , where 𝑁𝑔 is the             word-level attention pooling is introduced to produce a word
number of pronunciation aspects at granularity 𝑔 . In the                  representation vector from its constituent phonemes,
meantime, for the canonical phoneme sequence 𝐪 =                           instantiated with a 1-D depth-wise convolution layer followed
                                                                           by an MHA layer and an average operation. The word-level
(𝑞1 , 𝑞2 , … , 𝑞𝑁 ), the proposed model seeks to detect an error
                                                                           input representations X𝑤 are computed by individually passing
state sequence 𝐞 = (𝑒1 , 𝑒2 , … , 𝑒𝑁 ) and generate a phonetic
                                                                           X𝑝 and H𝑝 into the word-level attention pooling and
diagnosis sequence 𝐲 = (𝑦1 , 𝑦2 , … , 𝑦𝑁 ). Both 𝑒𝑛 and 𝑦𝑛 are             subsequently packing them together with a linear projection:
phoneme-level pronunciation feedback for 𝑞𝑛 , where 𝑒𝑛 = 1                       ̂    ̂ 𝑤 = AttPool𝑤 (X𝑝 ), AttPool𝑤 (H𝑝 ),
                                                                                 X𝑤 , H                                              (6)
denotes a mispronounced phoneme segment and 𝑒𝑛 = 0 for a                                               1                2


correct one, while 𝑦𝑛 specifies the phoneme produced by the                                             ̂𝑤; H
                                                                                         X𝑤 = Linear𝑤 ([X   ̂ 𝑤 ]).                    (7)
learner.                                                                   Next, the word-level textual embeddings E𝑤 are added to X𝑤 ,
B. Interactive Hierarchical Neural Modeling                                and a word encoder is employed to generate word-level
Phoneme-level Modeling. For an input utterance, we first                   contextualized representations H𝑤 :
extract various pronunciation features to portray the                                          H𝑤       𝑤
                                                                                                 0 =X +E ,
                                                                                                               𝑤
                                                                                                                                  (8)
pronunciation quality of the L2 learner at phoneme-level, which
are then concatenated and projected to obtain a sequence of                                 H𝑤 = WordEnc(H𝑤
                                                                                                          0 ),                         (9)
condensed acoustic features X𝑝 . The feature extraction process                    𝑤
                                                                           where E are obtained by mapping the text prompt T into the
is formulated as:                                                          corresponding embedding sequence via a word and position
       X𝑝 = Linear𝑝 ([EGOP ; EDur ; EEng ; ESSL ]),        (1)             embedding layer, and WordEnc(⋅) consists of 2 convolution-
                                                                           augmented Branchformer blocks. Finally, three distinct 1-D
where Linear𝑝 (⋅) is a single feedforward layer, EGOP is
                                                                           depth-wise convolution layers are performed on H𝑤 to generate
goodness of pronunciation (GOP)-based features including log
                                                                           word-level aspect representations (i.e., H𝑤1 , H𝑤2 , and H𝑤3 ),
phoneme posterior (LPP) and log posterior ratio (LPR) [12][14].
                                                                           which are then transformed into the pronunciation score
EDur and EEng are prosodic features related to duration and                sequences by the corresponding word-level regressors.
energy statistics [49][50], while ESSL are self-supervised                 Utterance-level Modeling. For the utterance-level assessments,
learning (SSL) based features [46]. We then add phoneme-level              we first extract an utterance-level SSL-based feature ̅̅E̅ SSL by
textual embeddings E𝑝 to X𝑝 , followed by a phoneme encoder                applying average pooling over the time dimension of frame-
to obtain aspect representations H𝑝 = (𝐡1𝑝 , 𝐡2𝑝 , … , 𝐡𝑁
                                                        𝑝
                                                          ):               level SSL-based features. Next, we merge H𝑤1 , H𝑤2 , and H𝑤3
                         H𝑝0 = X𝑝 + E𝑝 ,                             (2)   with a weighted combination to obtain word-level
                                                                           representations H   ̅̅̅ ̅𝑤 . A sequence of utterance-level input
                      H𝑝 = PhnEnc(H𝑝0 ).                             (3)   representations H𝑢0 is obtained by first applying 1-D depth-wise
Here, E𝑝 is generated by passing 𝐪 into a phoneme-level                    convolution layers to X𝑝 , H𝑝 , and ̅̅̅H̅𝑤 , followed by
prompt encoder which comprises a phoneme and position                      concatenation and linear projection. Consequently, an utterance
embedding layer. PhnEnc(⋅) is composed of a stack of 3                     encoder is exploited to generate contextualized representations
convolution-augmented Branchformer blocks.                                 H𝑢 :
    Afterward, the pronunciation feedback module builds on                              ̅̅̅ ̅𝑤 = Merge(H𝑤1 , H𝑤2 , H𝑤3 ),
                                                                                        H                                              (10)
H𝑝 to estimate the multi-faceted pronunciation feedback,
comprising three components: an error detector, a diagnosis                                                             ̅̅̅ ̅𝑤 )]),
                                                                             H𝑢0 = Linear𝑢 ([DC1 (X𝑝 ); DC2 (H𝑝 ); DC3 (H             (11)
predictor, and an accuracy score regressor. The error detector is
a binary labeling model which predicts the error state 𝑒𝑛̂ ,                                 H𝑢 = UttEnc(H𝑢0 ),                       (12)
indicating whether the 𝑛-th phoneme of 𝐪 is identified as a                where Merge(⋅) is a weighted average operation [51],
mispronunciation:                                                          UttEnc(⋅) is a single convolution-augmented Branchformer
        P𝑑𝑒𝑡 (𝑒 ̂𝑛 |𝐪, X) = Sigmoid(Linear𝑑𝑒𝑡 (𝐡𝑝𝑛 )),     (4)             block, and DC1 (⋅), DC2 (⋅), DC3 (⋅) are distinct 1-D depthwise

  1
      CMU dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
                                                                                                                                                                         6
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

convolution layers, each with a kernel size of 3. Afterward, five               Centroid Vector Calculation                                "
                                                                                                                     !%"        !$"       !#       …
                                                                                                                                                         "
                                                                                                                                                        !!
separate attention pooling modules are built on top of H! to
generate utterance-level aspect representation vectors. These                      Phoneme Encoder
                                                                                                              !%&          $          $
                                                                                                                    !!" . !! !!" . !& !!" . !%
                                                                                                                                               $
                                                                                                                                                   …
                                                                                                                                                             $
                                                                                                                                                       !!" . !#
features are then combined with ̅̅E̅ SSL via the residual
connections and converted into utterance-level aspect scores                                      …           !$&          $          $
                                                                                                                    !&" . !! !&" . !& !&" . !%
                                                                                                                                               $
                                                                                                                                                   …
                                                                                                                                                             $
                                                                                                                                                       !&" . !#
through the respective regressors.                                                 Audio Signals X                         $          $        $             $
Training Objective. The training objective of the proposed                                                    !#&   !%" . !! !%" . !& !%" . !%     …   !%" . !#
                                                                               Text Prompt T: Call It Bear
model is calculated from the losses of APA and MDD:                            K     AO      L   … IH         ...     ...       …         …              …
                                                                                                                                                   …
             ℒ𝑀𝑢𝐹𝐹𝐼𝑁 = ℒ𝐴𝑃𝐴 + ℒ𝑀𝐷𝐷 .                      (13)
                                                                                                               &     "      $"       "$        $        "    $
                                                                                   Prompt Encoder             !!    !# . !! !# . !& !# . !%        …   !# . !#
The APA loss is a weighted sum of the mean square error (MSE)
losses gathered from different granularity levels:                              Centroid Vector Calculation
                    ℒ𝑝𝑗𝑝        ℒ 𝑗𝑤         ℒ 𝑗𝑢                         Fig. 5. The visualization of the calculation process for the
      ℒ𝐴𝑃𝐴 = ∑           +∑ 𝑤 +∑ 𝑢 ,                    (14)
                 𝑗
                    𝑁𝑝      𝑗
                                 𝑁𝑤       𝑗
                                              𝑁𝑢                          contrastive term ℒ𝑐𝑜𝑛 .
                 𝑝              𝑤             𝑢

where ℒ𝑝𝑗𝑝 , ℒ𝑤𝑗𝑤 , and ℒ𝑢𝑗𝑢 are phoneme-level, word-level,             contrastive term ℒ𝑐𝑜𝑛 aiming to maximize the similarity
and utterance-level losses for disparate aspects, and 𝑁𝑝 , 𝑁𝑤 ,         between paired phoneme representations while minimizing the
𝑁𝑢 mark the numbers of aspects at each granularity. On a                similarity of unpaired ones [52][53]. The contrastive term ℒ𝑐𝑜𝑛
separate front, the training objective of MDD comes with the            includes two losses, with a temperature hyper-parameter 𝜏 that
tasks of mispronunciation detection ℒ𝑑𝑒𝑡 and diagnosis ℒ𝑑𝑖𝑎𝑔 :          controls the strength of penalties on negative samples:

                ℒ𝑀𝐷𝐷 = ℒ𝑑𝑒𝑡 + ℒ𝑑𝑖𝑎𝑔 ,                         (15)                               ℒ𝑐𝑜𝑛 = ℒ𝑝2𝑡 + ℒ𝑡2𝑝 ,                                             (19)
                     𝑁                                                              1       exp(𝜙(𝐳𝑝𝑖 , 𝐳𝑡𝑖 )/𝜏)
                                                                                                     𝑀

         ℒ𝑑𝑒𝑡 = − ∑ logP𝑑𝑒𝑡 (𝑒𝑛̂ = 𝑒𝑛 |𝐪, X),                 (16)          ℒ𝑝2𝑡 = − ∑ log 𝑀                        ,                                             (20)
                                                                                    𝑀 𝑖=1 ∑𝑗=1 exp(𝜙(𝐳𝑝𝑖 , 𝐳𝑡𝑗 )/𝜏 )
                     𝑛=1
                      𝑁
                                                                                    1 𝑀     exp(𝜙(𝐳𝑡𝑖 , 𝐳𝑝𝑖 )/𝜏)
        ℒ𝑑𝑖𝑎𝑔 = − ∑ logP𝑑𝑖𝑎𝑔 (𝑦𝑛̂ = 𝑦𝑛 ∣𝐪, X),                (17)          ℒ𝑡2𝑝 = − ∑ log 𝑀                         ,                                            (21)
                     𝑛=1
                                                                                    𝑀 𝑖=1 ∑𝑗=1 exp(𝜙(𝐳𝑡𝑖 , 𝐳𝑝𝑗 )/𝜏 )
where ℒ𝑑𝑒𝑡 and ℒ𝑑𝑖𝑎𝑔 represent the negative log-likelihood              where 𝜙(𝐳𝑝𝑖 , 𝐳𝑡𝑗 ) is dot product between ℓ2 normalized vectors
used for training the detector and the predictor, respectively.         𝐳𝑝𝑖 and 𝐳𝑡𝑗 (cosine similarity). During training, ℳ is
                                                                        constructed from each batch, where we empirically sample the
    IV. CONTRASTIVE PHONEMIC ORDINAL REGULARIZER
                                                                        data instances with the highest proficiency score to compute
    To generate more phoneme-discriminative features for the            centroid vectors.
multi-faceted pronunciation assessment model, we proposed               Phonemic Characteristic Term. The phonemic characteristic
contrastive phonemic ordinal regularizer (ConPCO), which                term ℒ𝑝𝑐 preserve the phonemic proximity information by
consists of three mathematical terms: the contrastive term ℒ𝑐𝑜𝑛 ,       minimize the negative distances between centroid vectors 𝐳𝑝𝑖 :
the phonemic characteristic term ℒ𝑝𝑐 , and the ordinal term ℒ𝑜 .                                         𝑀
                                                                                                  1
ℒ𝑐𝑜𝑛 aims to simultaneously project the phoneme                                 ℒ𝑝𝑐 = −                 ∑ ∑∥𝐳𝑝𝑖 − 𝐳𝑝𝑗 ∥2 ,        (22)
                                                                                             𝑀(𝑀 − 1) 𝑖=1 𝑖≠𝑗
representations generated from a pronunciation assessment
model and the embeddings of phoneme-level text prompt into a            where ℒ𝑝𝑐 is equivalent to maximizing the distances between
joint feature space. ℒ𝑝𝑐 and ℒ𝑜 adjust the distances between            phoneme categories during the optimization process.
inter- and intra-phoneme categories, where the former enhances          Ordinal Term. To reflect ordinal relationships of regression
inter-phoneme discrepancy, and the latter improves intra-               targets in the feature space, the ordinal term ℒ𝑜 is defined to
phoneme compactness with ordinal relationship. The proposed             minimize the distance between the feature representations 𝐡𝑖𝑝
ConPCO regularizer is formulated as:                                    and their corresponding phoneme centroid vectors 𝐳𝑝𝑖 with
            ℒ𝐶𝑜𝑛𝑃𝐶𝑂 = ℒ𝑐𝑜𝑛 + ℒ𝑝𝑐 + ℒ𝑜 .                       (18)      relative differences of proficiency score:
                                                                                                      1 𝑁
Contrastive Term. Let H     𝑝
                                = (𝐡1𝑝 , 𝐡2𝑝 , … , 𝐡𝑁
                                                    𝑝
                                                      ) stand for the                      ℒ𝑜 =         ∑ 𝑤 ‖𝐡𝑝 − 𝐳𝑝𝑖 ‖2 ,                                        (23)
                                                                                                      𝑁 𝑖=1 𝑖 𝑖
phoneme representation sequence of an utterance generated by
a phoneme encoder in a pronunciation scoring model, and E𝑝 =            where 𝑤𝑖 = |𝐶 − 𝑦𝑖𝑝 | is a compactness weight for each 𝐡𝑖𝑝 ,
(𝐞𝑝1 , 𝐞𝑝2 , … , 𝐞𝑝𝑁 ) denote the textual embedding of canonical        reflecting the ordinal behaviors within the label space, with 𝑦𝑖𝑝
phonemes generated by a phoneme-level prompt encoder. Next,             denoting the corresponding phoneme-level accuracy score. The
a set of paired phoneme representations ℳ = {(𝐳𝑝𝑖 , 𝐳𝑡𝑖 ), 𝑖 =          tunable constant 𝐶 is set to be 3, representing the highest
1, … , 𝑀} is obtained by first applying separate linear                 accuracy score plus a small margin.
projections to H𝑝 and E𝑝 , and then calculating the centroid
vectors for each phoneme category. Next, as illustrated in Fig.
5, the 𝑀 × 𝑀 similarities are derived from ℳ , with the
                                                                                                                                      7
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

                                                      High                                    TABLE I
                                                      Medium
                                                      Low                  STATISTICS OF APA TASK IN THE SPEECHOCEAN762
                                                                                     Automatic Pronunciation Assessment
                                                                                                          Score       # of Counts
                                                                         Granularity      Aspect
                                                                                                        Interval    Train      Test
                                                                          Phoneme        Accuracy         [0, 2]   47,076 47,369
                                                                                         Accuracy
                                                                           Word            Stress        [0, 10]   15,849 15,967
 Fig. 6. Phoneme statistics of Speechocean762, including                                   Total
 occurrence count and corresponding mispronunciation rate.                               Accuracy
                                                                                        Completeness
                                                                          Utterance       Fluency        [0, 10]    2,500     2,500
              IV. PHONEME-SPECIFIC VARIATION
                                                                                          Prosody
    To balance the distribution of predicted phonemes while                                Total
accounting for pronunciation difficulties, the logits of phoneme
predictions generated by a phoneme predictor are perturbed                                  TABLE II
with randomly sampled Gaussian noise, where the radius is                 STATISTICS OF MDD TASK IN THE SPEECHOCEAN762
determined by the phoneme-dependent variance. To this end,
the proposed training scheme, phoneme-specific variation                          Mispronunciation Detection and Diagnosis
(PhnVar), consists of two factors, i.e., a data quantity factor and                                                    Counts
                                                                            Type              Description
a pronunciation difficulty factor. The data quantity factor                                                        Train    Test
assigns smaller variance to majority phoneme categories and                               The uttered phoneme
larger variance to minority ones, while the pronunciation                Correctness aligns with the canonical 45,088 45,959
                                                                                                phoneme
difficulty factor modulates feature areas based on
                                                                                        A canonical phoneme is
mispronunciation rates. Formally, we revisit Eq. (5) and express          Deletion
                                                                                                 omitted
                                                                                                                    450     396
the probability of the 𝑛-th canonical phoneme being predicted                           A canonical phoneme is
as a diagnostic phoneme 𝑘, derived from the softmax function:            Substitution                               914     593
                                                                                        mispronounced to others
                           exp(𝑔𝑘𝑛 )                                        Non-        The uttered phoneme not
                  𝑝𝑘𝑛 = 𝑀                .                   (24)        categorical       exists in the CMU        488     332
                        ∑𝑖=1 exp(𝑔𝑖𝑛 )
                                                                            Error       pronunciation dictionary
Here, 𝑔𝑘𝑛 is the logit of the 𝑘-th phoneme in logit vector 𝐠𝑛 =                         A canonical phoneme is
                                                                          Accented
                                                 𝑝                                     pronounced correctly but     136       89
(𝑔1𝑛 , 𝑔2𝑛 , … , 𝑔𝑀
                  𝑛
                    ), generated by Lineardiag (𝐡𝑁 ), where 𝑀 is            Error
                                                                                          with a strong accent
the number of phoneme categories. We then augment logits
with phoneme-specific variance, defined as the weighted p of
the data quantity factor QF𝑘 and the pronunciation difficulty
factor DF𝑘 for phoneme 𝑘, with coefficients 𝛼 and 𝛽:                                    IV. EXPERIMENTAL SETUPS
                               𝛼 × log(QF𝑘 ) + 𝛽 × log(DF𝑘 )          A. Experimental Data and Evaluation Metrics
  𝑔𝑛̂𝑘 = 𝑔𝑘𝑛 + 𝛿(σ) × exp (                                  ),
                                          𝛼+𝛽                         Dataset. A series of experiments were conducted on the
                                                                      Speechocean762 dataset, a publicly available dataset
                                                         (25)         specifically designed for research on computer-assisted
where 𝛿(σ) stands for a Gaussian distribution with a zero mean        language learning [26]. This dataset contains 5,000 English-
and the standard deviation σ. Both 𝛼 and 𝛽 are set to 1 in our        speaking recordings spoken by 250 Mandarin L2 learners. The
experiments. The data quantity factor is defined as normalized        training and test sets are of equal size, and each of them has
inverse phoneme frequency operated in the logarithmic scale:          2,500 utterances. For the APA task, pronunciation proficiency
                  𝑐𝑘                  ∑𝑀    𝑞𝑖                        scores were evaluated at multiple linguistic granularities with
         QF𝑘 =          ; 𝑐𝑘 = log 𝑖=1 ,                 (26)         various pronunciation aspects, as the statistics of the APA task
                max 𝑐𝑖                   𝑞𝑘
                    𝑖                                                 are summarized in Table I. For the MDD task, the phoneme
where 𝑞# is the number of instances in phoneme category 𝑘.            labels follow the definitions in the CMU pronunciation
The pronounce difficulty factor is expressed as normalized            dictionary, which includes a set of 39 canonical phonemes. In
mispronunciation rate:                                                Speechocean762, mispronunciation labels were manually
                  𝑑𝑘                𝑚𝑝𝑘                               assigned to phoneme segments with accuracy scores below 0.5
        DF𝑘 =          ; 𝑑𝑘 =              ,        (27)              and were categorized into four types: deletion, substitution,
                𝑚𝑎𝑥 𝑑𝑖           𝑚𝑝𝑘 + 𝑐𝑝𝑘
                    𝑖                                                 non-categorical error, and accented error. Table II summarizes
where 𝑚𝑝𝑘 and 𝑐𝑝𝑘 are the number of mispronounced and                 the phoneme segment statistics for the MDD task.
correctly pronounced instances for phoneme category 𝑘 ,               Evaluation Metrics. The primary evaluation metric for APA
respectively.                                                         adopts Pearson correlation coefficient (PCC), which measures
                                                                      the linear correlation between predicted scores and ground-truth
                                                                                                                                    8
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <




                    (a) MuFFIN                            (b) MuFFIN + ℒ$%                         (c) MuFFIN +ℒ!" + ℒ#

  Fig. 7. Visualization of phoneme representations from MuFFIN trained with the phonemic characteristic and the ordinal terms
  (i.e., ℒ𝑝𝑐 and ℒ𝑜 ). For each data point, color indicates the phoneme-level accuracy score, and shape denotes the corresponding
  phoneme category. We show the phoneme representations H$ of (a) the vanilla MuFFIN model, (b) MuFFIN+ℒ𝑝𝑐 , and (c)
  MuFFIN+ℒ𝑝𝑐 + ℒ𝑜 .




                    (a) MuFFIN                            (b) MuFFIN + ℒ!"#                 (c) Localized Zoomed-in Counterpart

  Fig. 8. Visualization of phoneme representations, with the blue and orange points denoting feature representations H$ and E$ ,
  generated from the phoneme encoder of MuFFIN and the phoneme-level prompt encoder, respectively. The feature
  representations are displayed for (a) the vanilla MuFFIN model, (b) MuFFIN + ℒ𝑐𝑜𝑛 , and (c) a localized zoomed-in
  counterpart.

scores. In accordance with prior studies, mean square error         EEng , 1 for duration value EDur , and 3,072 for SSL-based
(MSE) is reported for phoneme-level accuracy. On the other          features ESSL .
hand, for MDD tasks, the evaluation metrics follow the scoring      Training Configuration. For the training configuration, we
rubrics in [9]. Specifically, the mispronunciation detection        followed to the settings reported in [24][25], where each
subtask is evaluated using recall (RE), precision (PR), and F1-     experiment consisted of 5 independent trials, and each trial runs
score (F1), while the mispronunciation diagnosis subtask is         for 100 epochs with different random seeds. In each trial, the
assessed with diagnostic error rate (DER), false rejection rate     model was trained with an Adam optimizer with an initial
(FRR), false acceptance rate (FAR), and phoneme error rate          learning rate of 1e-3 and a batch size of 25. A learning rate
(PER).                                                              scheduler was used to decay the learning rate by a factor of 0.1
B. Implementation Details                                           after the overall loss did not decrease for 10 consecutive epochs.
                                                                    Furthermore, our models were initialized with a pretrained
Feature Extraction. For the pronunciation feature extraction,       model following the pretraining strategies described in [41].
the GOP features, the energy, and the duration statistics are       The reported experimental results were averaged over the 5
adopted in line with our previous studies [24][25]. The             trials, with evaluation based on the minimum phoneme-level
extraction of SSL-based features follows the processing flow        MSE.
suggested in [46], where features are extracted from the outputs    Model Configuration. The phoneme-level, word-level, and the
of pretrained acoustic models, including Wav2vec2.0 [54],           utterance-level encoder (viz. PhnEnc(⋅) , WordEnc(⋅) ,
WavLM [55], and HuBERT [56]. The SSL-based and energy
                                                                    UttEnc(⋅)) consisted of 3, 2, and 1 convolution-augmented
features are extracted at the frame level and then aggregated
                                                                    Branchformer blocks, respectively [25]. Within each encoder
into phoneme-level representations based on timestamps of
                                                                    block, the self-attention branch was implemented with a single-
phoneme segments derived from forced-aligning the learner’s
                                                                    head attention layer, followed by two feed-forward layers. Both
speech to the reference text. The extracted phoneme-level
                                                                    the self-attention and feed-forward layers had a hidden
proficiency features amount to 3,164 dimensions, comprising
                                                                    dimension of 24. Meanwhile, the convolutional branch
84 dimensions for GOP features EGOP , 7 for energy statistic
                                                                    consisted of a depth-wise convolutional layer with a 1 × 3
                                                                                                                                     9
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <




                   (a) MuFFIN                          (b) MuFFIN + PhnVar w/o DF                     (c) MuFFIN + PhnVar

 Fig. 9. Visualization of phoneme-level logits and decision boundaries from the diagnosis predictor of MuFFIN models trained
 with the proposed phoneme-specific variation (PhnVar) scheme. We show the phoneme logits of diagnosis predictor for (a) the
 vanilla MuFFIN model, (b) MuFFIN + PhnVar w/o DF (i.e., the variant of PhnVar without accounting for the pronunciation
 difficulty factor), and (c) MuFFIN + PhnVar.
kernel and a point-wise convolutional layer with a 1 × 1 kernel,     pronunciation cues, while also integrating phonological
both of which had 24 channels. To aggregate word-level and           features to enhance phoneme-level textual information [46]. As
utterance-level features, the attention pooling modules were         for hierarchical models, HierGAT devises a language-hierarchy
composed of a depth-wise convolutional layer and a single-           aware model with a series of graph attention neural networks
head self-attention layer, where the convolutional layer had 24      and further strengthening the relatedness among the aspects
channels with a kernel size of 1 × 3 and the attention layer had     with aspect attention mechanisms [58]. 3MH, a previous state-
a hidden dimension of 24. Furthermore, we set the hidden             of-the-art method for APA, employs 3M as the backbone model
dimension of the projection layers (viz. Linear𝑝 (⋅), Linear𝑤 (⋅),   and introduces sup-phoneme modeling to capture finer
and Linear𝑢 (⋅)) to 24. During the training phase, the tunable       articulation traits within the language hierarchy between the
                                                                     phoneme and word levels [48]. Gradformer (GFR) decouples
parameters of ℒ𝑝 , ℒ𝑤 , and ℒ𝑢 in Eq. (14) were set to 3, 1, and
                                                                     the language hierarchy into two sub-levels, i.e., lower
1, respectively, while the temperature factor 𝜏 in Eqs (20) and      (phoneme and word), and higher (utterance). A Conformer
(21) was set to 1.                                                   encoder models aspects at the lower linguistic level, while a
Mispronunciation Detection and Diagnosis via MuFFIN. To              Transformer decoder processes a sequence of learnable aspect
detect mispronunciation segments, MuFFIN follows a                   vectors and interacts with the encoder outputs to assess aspect
pronunciation scoring-based paradigm, where the outputs of the       at the utterance-level [42]. 3) Multi-faceted pronunciation
phoneme-level error detector serve as indicators of                  assessment: Ryu2023 introduces a unified model architecture
mispronounced segments. The phoneme segments are                     that jointly optimizes both phoneme recognition and
identified as mispronounced if the corresponding indicators          pronunciation assessments by independently stacking a CTC-
exceed a predefined threshold. Subsequently, the detected            based phoneme recognizer and a set of regressors on top of a
mispronunciation segments are fed into the phoneme-level             pretrained acoustic model [59]. JAM advances 3M by
predictor to generate diagnostic results. To ensure consistency      integrating a phoneme classifier to predict diagnostic phonemes
between the detector and predictor, we mask the canonical            based on input canonical phonemes and further boosts the MDD
phonemes (i.e., the phonetic transcription of the text prompt)       performance by exploiting electromagnetic articulography
during the softmax computation of the predictor.                     (EMA) features to capture the articulatory movements of L2
C. Compared Methods                                                  learners [60].
We compare the proposed model (MuFFIN) with three
categories of pronunciation assessment models. 1) Single-                              V. EXPERIMENTAL RESULT
aspect pronunciation assessment: Lin2021 is a hierarchical           A. Qualitative Analysis
APA model which takes phoneme-level surface features as              At the outset of experiments, we first analyze the phoneme
inputs and assesses accuracy scores at utterance-level [57].         statistics of Speechocean762 to reveal the intricate relationships
Kim2022 relies on layer-wise contextual representations              between data quantity and pronunciation difficulty for each
extracted from a pretrained acoustic model to measure oral           phoneme. Following this, we examine the effectiveness of the
skills in terms of fluency or prosody at the utterance-level [30].   proposed phoneme-level regularizers through a series of
2) Multi-aspect and multi-granular pronunciation assessment:         qualitative visualizations.
For the assessment models with parallel neural structures,
GOPT and LSTM are prominent models, both of which                    Phoneme Statistics of Speechocean762. In Fig 6, the
consume a sequence of GOP features and generate a set of             occurrence counts of phoneme segments (blue bars) paired with
proficiency scores at the phoneme-, word-, and utterance-level       their respective mispronunciation rates (orange points) are
simultaneously [45]. 3M augments the input features of GOPT
with SSL-based features to capture supra-segmental
                                                                                                                                                                10
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

                                                TABLE III
        PERFORMANCE EVALUATION OF MUFFIN AND COMPARATIVE METHODS ON THE APA TASK OF SPEECHOCEAN762
                        Phoneme-level
                                                     Word-level Aspects                                      Utterance-level Aspects
       Model                Acc.
                        MSE↓ PCC↑                Acc.↑       Stress↑      Total↑         Acc.↑       Comp.↑            Fluency↑    Prosody↑        Total↑
   Lin2021 [57]          -        -                -            -           -              -           -                   -           -           0.720
   Kim2022 [30]          -        -                -            -           -              -           -                 0.780       0.770           -

     LSTM [45]          0.089       0.591        0.514        0.294        0.531         0.720           0.076          0.745        0.747          0.741
                        (0.000)     (0.003)      (0.003)      (0.012)      (0.004)       (0.002)         (0.086)        (0.002)      (0.005)        (0.002)

     GOPT [45]          0.085       0.612        0.533        0.291        0.549         0.714           0.155          0.753        0.760          0.742
                        (0.001)     (0.003)      (0.004)      (0.030)      (0.002)       (0.004)         (0.039)        (0.008)      (0.006)        (0.005)

      3M [45]           0.078       0.656        0.598        0.289        0.617         0.760           0.325          0.828        0.827          0.796
                        (0.001)     (0.005)      (0.005)      (0.033)      (0.005)       (0.004)         (0.141)        (0.006)      (0.008)        (0.004)

     GFR [42]           0.079       0.646        0.598        0.334        0.614         0.732           0.318          0.769        0.767          0.756
                        (0.001)     (0.004)      (0.006)      (0.013)      (0.006)       (0.005)         (0.139)        (0.006)      (0.004)        (0.003)

   HierGAT [58]         0.073       0.683        0.648        0.327        0.663         0.798           0.531          0.840        0.833          0.821
                        (0.001)     (0.004)      (0.003)      (0.011)      (0.002)       (0.002)         (0.047)        (0.002)      (0.002)        (0.002)

     3MH [48]           0.071       0.693        0.682        0.361        0.694         0.782           0.374          0.843        0.836          0.811
                        (0.001)     (0.004)      (0.005)      (0.098)      (0.007)       (0.003)         (0.115)        (0.003)      (0.004)        (0.004)
   Ryu2023 [59]           -           -            -            -            -           0.719             -            0.775        0.773          0.743
     JAM [60]           0.076       0.664        0.622        0.241        0.638         0.773           0.205          0.831        0.829          0.805
                        (0.002)     (0.001)      (0.012)      (0.034)      (0.005)       (0.007)         (0.080)        (0.004)      (0.004)        (0.004)

      MuFFIN            0.063       0.742        0.705        0.315        0.714         0.807           0.768          0.841        0.832          0.830
                         (0.002)     (0.006)    (0.004)       (0.033)     (0.004)       (0.003)      (0.049)      (0.004)          (0.004)      (0.002)
 * The reported results include the mean PCC scores and standard deviations, calculated over 5 independent experimental trials. Acc. and Comp. refer to the
 pronunciation aspects of accuracy and completeness, respectively. The proposed MuFFIN achieves higher PCC scores compared to 3MH across all metrics
 except utterance-fluency, with approximate randomization test (𝑝 < 0.001).

reported for the speechocean762 dataset, where the phonemes                                              TABLE IV
are sorted by mispronunciation rate and then categorized into                        PERFORMANCE EVALUATION OF MUFFIN WITH PHONEME-
three disjoint subsets: high (mispronunciation rate above 5.1%),                     LEVEL REGULARIZERS UNDER THE PHONEME-SPECIFIC
medium (mispronunciation rate between 5.1% and 3.4%), and                            VARIATION TRAINING SCHEME
low (mispronunciation rate below 3.4%) regions.
    In Fig 6, it is evident that the occurrence counts of phonemes                             MuFFIN
                                                                                                                        Phoneme
                                                                                                                                       Word-level Score
and their corresponding mispronunciation rates exhibit distinct                                                          Score
distributional patterns. For example, the high-occurrence                             PhnVar       ℒ𝑐𝑜𝑛       ℒ𝑝𝑐𝑜        Acc.      Acc.       Stress   Total
phonemes (e.g., /AH/, /T/, and /N/) are found within the                                  -          -             -      0.742     0.705      0.315    0.714
medium mispronunciation rate region. In contrast, some low-                              V           -             -      0.746     0.704      0.310    0.714
occurrence phonemes (e.g., /ZH/, /TH/, and /NG/) are often
                                                                                         V          V              -      0.749     0.707      0.314    0.718
associated with high mispronunciation rates. Building on this,
to mitigate the data imbalance issue facing the MDD task, the                            V           -             V      0.745     0.703      0.296    0.713
proposed phoneme-specific variance incorporates two novel                                V          V              V      0.747     0.708      0.341    0.718
regulation terms: a quantity factor and a pronunciation                              *Acc. refer to the pronunciation aspects of accuracy.
difficulty factor. The former balances the feature distributions
of phonemes, while the latter adjusts feature scatteredness                          From Fig. 7(a), it is observed that despite MuFFIN jointly
according to the mispronunciation rate.                                          optimizing both the phoneme recognition and the assessment
Qualitative Visualizations of Phoneme Representations for                        tasks, the resulting phoneme representations, however, are
the Phonemic Characteristic and the Ordinal Terms. In the                        inevitably grouped by phoneme-level accuracy scores,
second set of experiments, we graphically examining the                          inadequate to explicitly capturing the subtle distinctions
impacts of the phonemic characteristic term and the ordinal                      between phonemes in the feature space. When training MuFFIN
term (i.e., ℒ𝑝𝑐 and ℒ𝑜 ) based on the proposed APA model. As                     with ℒ𝑝𝑐 , as shown in Fig. 7(b), the phoneme-discriminative
                                                                                 features are obtained, where the representations disperse
depicted Fig. 7, we extract the phoneme representations H𝑝
                                                                                 according to their respective phoneme categories. However,
from the test set and visualize each data point pertaining to the
                                                                                 simply separating the feature representations would omit the
phoneme category (denoted by shape) and the corresponding
                                                                                 ordinal relationships, which might impede pronunciation
pronunciation accuracy score (represented by color).
                                                                                 assessment tasks. In response to this, the synergy of ℒ𝑝𝑐 and
                                                                                                                                      11
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

ℒ𝑜 serves as a remedy, which enables the phoneme                     B. Performance of Automatic Pronunciation Assessment
representations to reflect both categorical distinctions and         In this subsection, we turn to evaluating the performance of
ordinal relationships derived from their accuracy scores, as         MuFFIN on pronunciation assessments and compare it with
shown in Fig. 7(c). Specifically, integrating ℒ𝑜 leads to a          several state-of-the-art models to validate its effectiveness. The
stronger correlation between pairwise distances and phoneme-         corresponding results are presented in Table III. We begin by
level accuracy within each phoneme category, resulting in an         discussing the experimental results at the phoneme- and word-
outward dispersion in the feature space as accuracy decreases.       level assessments and then proceed to the utterance-level
Grounded on these observations, incorporating ℒ𝑝𝑐 and ℒ𝑜             assessments.
during the training process of MuFFIN substantially improves         Assessment Performance at Phoneme and Word Levels. We
the discriminability of phoneme representations and                  first evaluate the assessment performance at the phoneme and
simultaneously reflects the ordinal relationships of the             word levels in Table III, from which the following observations
predicted accuracy scores in the feature space.                      can be made. First, the proposed MuFFIN outperforms other
                                                                     APA models by a remarkable margin in most pronunciation
Qualitative Visualizations of Phoneme Representations for            assessment tasks, except for the word-level stress. Specifically,
the Proposed Contrastive Term. Subsequently, to                      MuFFIN stands out in the phoneme-level accuracy,
qualitatively assess whether the contrastive term ℒ𝑐𝑜𝑛 aligns        demonstrating PCC score improvements of 4.9% and 5.9% over
the speech-derived representations (colored in blue) with their      the prior-art models, 3MH and HierGAT, respectively. We
corresponding textual embeddings (colored in orange) for             attribute these performance gains to the proposed multi-faceted
phoneme segments, we visualized the representations H𝑝 and           phoneme-level pronunciation feedback module, which jointly
E𝑝 from MuFFIN on the test set in Fig 8. By comparing among          optimizes the APA and MDD tasks, thereby encouraging the
Fig. 8(a) and Fig. 8(b), we observe that the proposed ℒ𝑐𝑜𝑛           phoneme encoder to learn distinct phoneme identities when
effectively projects these two types of phoneme representations      evaluating the pronunciation scores. With respect to the word-
into a shared feature space, resulting in a more coherent            level assessments, MuFFIN generally performs well across
distribution. Going one step further, a zoomed-in view is            most pronunciation aspects. However, in word-level stress, our
presented in Fig. 8(c), which highlights that the contrastive term   model demonstrates comparable performance against GFR and
not only aligns the heterogeneous phoneme representations            HierGAT, while trailing behind 3MH. A possible reason for the
with the corresponding textual embeddings, but also preserves        inferior performance is that 3MH leverages sub-phoneme
the phoneme-specific characteristics across phoneme                  modeling to create a pseudo (augmented)-linguistic hierarchy
categories.                                                          between phoneme and word levels, facilitating better rendering
Qualitative Visualizations of Phoneme Logits for the                 of supra-segmental information for word-level assessments.
Proposed Phoneme-specific Variation. Finally, to                         Second, we turn to evaluate the performance of strong
qualitatively evaluate the effectiveness of the proposed PhnVar      baselines with parallel neural architectures (the second group in
training scheme, we visualize the phoneme logits and decision        Table III). Compared to LSTM and GOPT, 3M stands out as a
boundaries of the diagnosis predictor. As shown in Figure 9, we      promising method, with its superiority stems from effectively
compare the MuFFIN models trained with PhnVar and the                exploiting SSL-based features to mitigate the data scarcity issue
variant (i.e., PhnVar without accounting for the pronunciation       of L2 learners’ speech and simultaneously encapsulate long-
difficulty factor term). Furthermore, the visualized phonemes        range articulatory traits. By augmenting the inputs of 3M with
(/AH/, /T/, /IH/, and /K/) are uniformly sampled from the test       electromagnetic articulography features, JAM slight boosts the
set, with occurrence counts of 4.4K, 4K, 3K, and 1.3K, and           performance in most pronunciation assessment tasks. Finally,
mispronunciation rates of 4.80%, 3.55%, 5.46%, and 2.39%,            among the APA models with advanced neural architectures (the
respectively.                                                        third group in Table III), 3MH achieves the best performance,
     The observations from Fig. 9 are highlighted as follows.        benefiting from the synergy of hierarchical modeling
First, the logits of phonemes with higher occurrence counts tend     approaches and depth-wise convolution layers. However,
to occupy a larger portion of the feature space, while those with    compared to MuFFIN, 3MH is limited in functionality, as it
lower occurrence counts are compressed into a narrower region.       only qualifies pronunciation proficiency with various aspect
This is evidenced by the increasing size of feature regions for      scores, lacking phoneme-level diagnostic feedback.
phonemes /K/, /IH/, /T/, to /AH/, which is consistent with their     Assessment Performance at Utterance-level. For the
respective occurrence frequencies. Subsequently, in Fig. 9(b),       performance of utterance-level pronunciation assessment in
it is observed that training MuFFIN with the variant of PhnVar       Table III, MuFFIN achieves the highest performance across
(viz. PhnVar w/o DF) results in more uniformly distributed           most aspects. Compared to 3MH, MuFFIN enhances the PCC
feature regions, independent of phoneme occurrence counts.           scores by 2.5% in utterance-level accuracy, 2.9% in utterance-
However, adjusting the feature space solely factoring in data        level total, and achieves comparable performance in utterance-
quantity factor fails to capture the distribution of                 level fluency and prosody. MuFFIN also achieves substantial
mispronunciations. In light of this, our PhnVar additionally         improvements in the utterance-level completeness assessment,
takes the pronunciation difficulty factor into account. The          a metric reflecting the proportion of correctly pronounced
phoneme logits of MuFFIN trained with PhnVar are visualized          words in an utterance. This gain is attributed to the joint training
in Fig. 9(c), where the feature regions are partitioned by the
phoneme mispronunciation rates, with region sizes decreasing
in the order of /IH/, /AH/, /T/, and /K/.
                                                                                                                                           12
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

                                 TABLE V
  PERFORMANCE EVALUATION OF MUFFIN AND COMPARATIVE METHODS ON MDD TASK
                    Mispronunciation Detection           Mispronunciation Diagnosis
      Model                                                                                   PER (%)↓
                  RE (%)↑ PR (%)↑         F1 (%)↑    FAR (%)↓ FRR (%)↓ DER (%)↓
   Ryu2023 [59]     91.60      26.90       41.50          -             -           -            9.93
    JAM [60]        34.76      61.10       45.01        64.32         0.58        45.23          2.81
     MuFFIN         64.33      66.89       65.99        35.67         0.97        60.97          2.36
    w/ PhnVar       68.37      67.60       67.98        31.63         1.01        58.82          2.33
  * For the mispronunciation detection subtask, our best-performing model (MuFFIN+PhnVar) outperforms the   Fig. 10. Precision-recall curves
  base model (MuFFIN) with significantly better performance (𝑝 < 0.001).                                    for MuFFIN trained with PhnVar.
of the MDD task within the APA model, which consequently                     ordinal regularizer tends to either slightly enhance performance
enables MuFFIN to pinpoint mispronounced segments and                        or retain that of the vanilla MuFFIN model. In addition, training
identify corresponding phonemes in learners’ speech. By                      MuFFIN with ConPCO attains the best performance in the
leveraging phoneme-discriminative representations, MuFFIN                    word-level assessment tasks (as shown in the last row of Table
effectively propagates fine-grained information from phoneme-                IV).
to utterance-level assessments through a tailored, hierarchy-
                                                                             C. Performance of Mispronunciation Detection and
aware neural architecture.
                                                                                  Diagnosis
    Next, compared to other strong baseline models, Lin2021
trails behind several APA models trained on multiple                         In this subsection, we evaluate the performance of MDD for the
pronunciation aspects (the second group in Table III), revealing             multi-faceted pronunciation assessment models. As joint
that the single-aspect assessment models fail to leverage the                optimization of APA and MDD remains relatively
dependency relationships between aspects through multi-task                  underexplored in CAPT, only a limited number of relevant
learning, thereby leading to inferior performance. In subsequent             studies are available for comparison in the following
work, Kim2022 ventures into replacing conventional ASR-                      experiments. To the best of our knowledge, Ryu2023 is the first
driven features with SSL-based features, resulting in substantial            attempt to develop a multi-faceted pronunciation feedback
improvements and achieving comparable performance to 3M.                     model, while JAM represents the recent follow-up work. The
In comparison with multi-faceted pronunciation assessment                    main results of the MDD task are summarized in Table V. Next,
models (viz. JAM and Ryu2023), JAM demonstrates superior                     Tables VI delve deeper into the imbalance issues inherent in the
performance in most aspects of the utterance-level assessment.               MDD, demonstrating the effectiveness of the proposed
This improvement may stem from the novel use of fine-grained                 phoneme- variation training (PhnVar) scheme.
phoneme-level features, including GOP features, prosody                      Performance       Evaluation      of     MDD.      To     detect
statistics, and EMA features.                                                mispronunciations with the MuFFIN, we leverage the outputs
Effectiveness of Phoneme-specific Variation and ConPCO.                      of the phoneme-level error detector to identify
Lastly, we examine the effectiveness of the proposed training                mispronunciation segments. Specifically, we first collect the
scheme, phoneme-specific variation (PhnVar), and the                         detector outputs for all phoneme segments in the training set. A
contrastive phonemic ordinal regularizer (ConPCO) for the                    global threshold is then selected through grid search, with a
pronunciation assessment. In Table IV, we concentrate on the                 stride of 0.1 over a range [0.0, 1.0]. At the outset of MDD
assessment performance at both the phoneme and word levels,                  experiments, we present the precision-recall curves of MuFFIN
as we empirically found that the proposed regularizers do not                and the MuFFIN trained with PhnVar in Fig. 10. For our models
cause detrimental effects on the utterance-level pronunciation               (i.e., MuFFIN and MuFFIN+PhnVar) this global threshold is
assessments, with performance either slightly improved or at                 set to 0.4. In contrast to prior dictation-based methods (e.g.,
least on par with the vanilla MuFFIN model [24][25]. In                      Ryu2023 and JAM), which detect mispronunciations by
addition, ConPCO is decomposed into the contrastive term                     identifying discrepancies between recognized and canonical
(ℒ𝑐𝑜𝑛 ) and the phonemic ordinal regularizer (ℒ𝑝𝑐𝑜 = ℒ𝑝𝑐 +                   phonemes, MuFFIN adopts a scoring-based approach that
ℒ𝑜 ), both of which are then combined with PhnVar for training               detects mispronunciations via threshold tuning, facilitating
MuFFIN.                                                                      broader adaptability to diverse L2 learners.
    From Table IV, we can observe that the proposed training                     As shown in Table V, MuFFIN outperforms other methods
scheme, PhnVar, the proposed PhnVar training scheme yields a                 in the mispronunciation detection subtask, achieving
0.7% improvement in phoneme-level accuracy over the base                     outstanding performance in terms of F1-score and precision.
model. Subsequently, the incorporation of the phoneme-level                  Moreover, training MuFFIN with the phoneme-specific
regularizers (viz., ℒ𝑐𝑜𝑛 and ℒ𝑝𝑐𝑜 ) under the PhnVar training                variation (PhnVar) leads to notable improvements in all
regime benefits pronunciation assessments, as evidenced by the               evaluation metrics compared to the base model. This gain is
sustained or improved results at the phoneme- and word-level                 further illustrated in Fig. 10, where the orange line
assessment tasks. Furthermore, the contrastive term primarily                (MuFFIN+PhnVar) exceeds the blue line (MuFFIN) in area
boosts the performance in the aspects of phoneme-level                       under the precision–recall curve. Subsequently, in comparison
accuracy and word-level total score. In contrast, the phonemic               with other baseline methods, Ryu2023, on the basis of a CTC-
                                                                                                                                                                     13
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

                                                        TABLE VI
                                     PROBING THE IMBALANCE ISSUES IN MDD WITH MUFFIN

   Group      Metrics      Average PER (%)↓             Average Recall (%)↑                              Average Precision (%)↑              Average F1-score (%)↑
               Type      Many     Med.      Few      Many       Med.       Few                    Many                Med.        Few        Many        Med.    Few
                         1.19      2.27     10.93    70.19      75.53      68.56              50.84                   53.42       69.50      57.53       60.14 67.77
              MuFFIN
                        (0.31)    (0.14)   (25.79)   (7.76)    (15.61)    (22.75)            (12.31)                 (19.05)     (24.79)     (6.95)     (14.59) (21.43)
                         1.41      2.41      9.38    61.90      64.04      68.97              66.82                   66.27       72.70      63.86       62.71 69.05
              PhnVar
 Occurrence             (0.34)    (1.51)   (20.19)   (6.22)    (17.41)    (24.66)             (9.81)                 (16.03)     (24.14)     (5.80)     (10.64) (9.38)
   Count                 1.45      2.42     11.22    64.71      68.62      67.99              61.27                   58.87       61.83      63.02       59.23 62.38
              w/o DF
                        (0.34)    (1.63)   (25.83)   (8.43)    (23.62)    (24.77)            (12.99)                 (14.96)     (22.68)     (7.94)     (14.34) (19.55)
                         1.28      2.09      9.89    57.55      62.72      60.02              69.22                   64.04       76.22      62.58       61.23 65.22
              w/o QF
                        (0.39)    (1.36)   (21.73)   (6.39)    (22.32)    (23.27)             (9.90)                 (19.84)     (24.85)     (6.80)     (18.60) (20.65)
               Type      High     Med.      Low       High     Med.        Low                           High         Med.        Low        High        Med.    Low
                         10.62     1.69     2.08      77.43     69.12      67.73              61.70                   55.83       56.23   66.95 59.81 58.68
              MuFFIN
                        (25.82)   (1.01)   (2.41)    (12.19)    (8.91)    (23.77)            (16.77)                 (20.91)     (24.37) (10.94) (14.18) (20.39)
                          9.30     1.82     2.15      70.51     62.19      62.22              70.88                   67.97       66.95   69.10 64.22 62.30
              PhnVar
  Mispron.              (20.13)   (0.91)   (2.36)    (14.30)   (10.57)    (24.78)            (12.35)                 (16.37)     (23.08) (6.93) (10.94) (20.47)
   Rate                  10.86     1.80     2.42      71.33     73.52      59.16              66.10                   54.20       61.67   66.77 60.79 57.08
              w/o DF
                        (25.75)   (0.92)   (3.68)    (12.52)   (16.62)    (26.29)            (14.05)                 (14.12)     (21.04) (8.10) (10.93) (20.64)
                          9.27     1.70     2.29      66.23     55.39      58.67              71.94                   70.43       67.11   67.29 61.04 60.69
              w/o QF
                        (21.65)   (1.06)   (3.72)    (14.28)   (16.96)    (23.25)            (12.39)                 (21.93)     (23.18) (8.42) (17.27) (20.74)
*Performance gains in the comparison of PhnVar variants (w/o DF and w/o QF) are highlighted in bold, while improvements
over the vanilla MuFFIN achieved by training with PhnVar are marked with underlines. ‘Mispron. Rate’ denotes the
mispronunciation rate.
based phoneme recognizer, achieves the highest recall value but
                                                                                                                   (a) The Average PER Evaluation
has the downside of low precision for the mispronunciation
detection task. This limitation is consistent with the                                                    10.00%                               10.93%




                                                                                    Average PER (%)
shortcomings reported for dictation-based MDD models                                                       7.50%

[37][38], where model performance is intrinsically constrained                                             5.00%
by the phoneme recognition rate. Instead of a direct free-                                                 2.50%
phoneme recognition process, JAM builds upon 3M and detects                                                                         2.27%
                                                                                                           0.00%         1.19%
mispronunciations in learners’ speech by attaching a phoneme                                                             Many      Medium       Few
classifier to the phoneme-level encoder. The corresponding                                                           Phoneme Occurrence Region
result demonstrates promising performance in terms of                                                           (b) The Average Recall Evaluation
precision metric, though it struggles with the low recall rate.                                           80.00%




                                                                                    Average Recall (%)
Compared to JAM, our MuFFIN achieves superior performance                                                               77.43%
                                                                                                                                    69.12%
                                                                                                          60.00%                                67.73%
across all metrics in the mispronunciation detection subtask.
These findings collaboratively highlight the effectiveness of the                                         40.00%

proposed scoring-based approach for multi-faceted                                                         20.00%
pronunciation feedback.
                                                                                                           0.00%
    For mispronunciation diagnosis subtask, our methods                                                                  High       Medium         Low
achieve promising performance in terms of FAR and PER.                                                                Pronunciation Error Region
However, a trade-off appears to exist between recall and the             Fig. 11. The performance evaluations highlight data
metrics of FRR and DER. Specifically, compared to JAM,                   imbalance issues in MDD. Our bar charts demonstrate that
MuFFIN achieves higher recall rate and lower PER but exhibits            (a) the data quantity factor primarily affects average PER in
inferior performance in both FRR and DER. This result implies            phoneme subsets grouped by occurrence count, and (b) the
that our model detects a greater number of mispronounced                 pronunciation difficulty factor significantly influences
segments but comes at the cost of diagnostic accuracy. We                average recall in phoneme subsets grouped by
leave this issue as a direction for future research.                     mispronunciation rate.
Systematic Examination of the Data Imbalance Problem in
MDD. In the following section, we explore the data imbalance           F1-score for each subset, with their mean and standard
problem in MDD, which stems from two intertwined factors               deviation. As shown in Table VI, in the first group, phoneme
within the phoneme segments, i.e., data quantity and                   segments are categorized into many (occurrence count above
pronunciation difficulty. To disentangle these two factors, we         1.3K), medium (occurrence count between 1.3K and 0.6K), and
divide the phoneme segments into two groups and report the             few (occurrence count below 0.6K) shot regions based on their
corresponding phoneme error rate (PER), recall, precision, and         occurrence counts. Conversely, in second group, phoneme
                                                                                                                                                   14
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

segments are categorized into high (mispronunciation rate                                 TABLE VII
above 5.1%), medium (mispronunciation rate between 5.1%                  ABLATION STUDY ON THE VARIED GRANULARITIES IN
and 3.4%), and low (mispronunciation rate below 3.4%)                       TRAINING OBJECTIVES OF APA FOR MUFFIN
pronunciation error regions according to their mispronunciation
rates (cf. Fig. 6). Furthermore, this set of experiments is                                                 Pronunciation Aspects
                                                                                         Number
conducted to assess the upper-bound performance of MuFFIN,              Training                                (PCC Score)
                                                                                           of
with the aims of analyzing the imbalance issue of MDD and               Objective                       Phone      Word         Utt.
                                                                                         Params
                                                                                                       Accuracy Accuracy Accuracy
examining the effectiveness of the proposed phoneme-specific
variance (PhnVar). To this end, a held-out set of 500 utterances        Utt. Only         541K            -           -        0.782
is set aside from the training data, with the remaining 2,000           Word Only         248K            -           0.674             -
utterances used for model training. This held-out set is designed
                                                                        Phone Only        126K          0.715            -              -
to cover both correct and incorrect pronunciations of each
phoneme and is then used to determine phoneme-specific                   +Word            249K          0.724         0.688             -
thresholds by maximizing the area under the precision–recall               +Utt.          608K          0.726         0.687           0.807
curve. Based on these phoneme-specific thresholds, we then            *In this table, MuFFIN refers to training without MDD task. ‘Utt.’ denotes
evaluate the MDD performance on the Speechocean762 test set.          the utterance.
    To highlight the data imbalance issues, Fig. 11 first presents                                  TABLE VIII
two sets of bar charts based on the MuFFIN model, where Fig.            ABLATION STUDIES ON TRAINING GRANULARITIES FOR
11(a) displays the average PER across phoneme subsets                           MUFFIN WITH APA AND MDD OBJECTIVES
grouped by occurrence count and Fig. 11(b) shows the average
recall for those grouped by mispronunciation rate. In Fig. 11(a),                     APA Task                           MDD Task
                                                                        Training
                                                                                 Phone Word Utt.
we can observe that the average PER of MuFFIN increases                 Objective                                 F1 (%) RE (%) PR (%)
                                                                                  Acc. Acc. Acc.
significantly from many-shot to few-shot region. This
observation is consistent with findings from prior studies on           MDD Only   -     -     -                   62.71      65.67    60.33
data imbalance learning [22][23], which manifests that                  +Utt.            -         -     0.787     63.34      63.49    63.45
occurrence frequency figures prominently in phoneme                     +Word            -      0.681         -    64.46      66.27    62.86
recognition accuracy. A naïve training process for a phoneme
classifier based on empirical risk minimization inevitably              +Phone        0.717        -          -    66.26      69.06    63.77
biases the model toward majority phoneme categories, resulting           +Word        0.741     0.696         -    66.04      67.08    65.36
in inferior performance on minority categories. Apart from the             +Utt.      0.742     0.705    0.807     65.99      64.33    66.89
data quantity issue, Fig. 11(b) reveals that the pronunciation        ‘Utt.’ denotes the utterance and ‘Acc.’ denotes the accuracy aspect.
difficulty factor causes a steady decline in average recall as
phoneme subsets shift from high to low mispronunciation rates.       F1-score. Furthermore, compared to the vanilla MuFFIN model,
This suggests that the infrequently mispronounced phoneme            training with PhnVar boosts the performance significantly in
segments pose greater challenges for pronunciation error             precision and F1-score, with gains particularly evident for
detection. Drawing on these observations, the proposed PhnVar        phoneme subsets with medium and few occurrences, as well as
training scheme integrates two mathematical terms designed to        those with medium and low mispronunciation rates.
simultaneously account for data quantity and pronunciation
difficulty.                                                          D. Ablation Studies for objectives of APA and MDD
    We next investigate the efficacy of each component in            In this subsection, a series of ablation studies are carried out to
PhnVar for handling the imbalance issues of MDD. The                 analyze the effectiveness of various training objectives for
ablation studies in Table VI are conducted by excluding either       MuFFIN on both pronunciation accuracy and mispronunciation
the data quantity factor (w/o QF) or the pronunciation difficulty    detection performance.
factor (w/o DF) from the proposed PhnVar. As shown in the            Effectiveness      of    Multi-granularity     Pronunciation
table, we observe a consistent performance trend across these        Assessments in MuFFIN. In this set of experiments, we begin
two groups, where the MuFFIN model gains in recall by                by training MuFFIN without the MDD task, then progressively
considering the data quantity factor (w/o DF), while precision       incorporate pronunciation assessment tasks at different
benefits from incorporating the pronunciation difficulty factor      linguistic levels. Table VII reports on the PCC scores on
(w/o QF). Subsequently, by factoring in both factors, PhnVar         assessment of pronunciation accuracy for MuFFIN with various
achieves notable performance gains in F1-score. These results        training objectives, comprising the assessment tasks at
shed light on the fact that simply balancing the logits of           phoneme, word, utterance levels, as well as cross-granularity
phoneme predictions encourages the MDD model to detect a             combinations. From Table VII, we observe that MuFFIN
greater number of pronunciation errors. This, however, boosts        trained in a multi-granularity manner achieves superior results
the recall rate while resulting in a decrease in precision           in relation to any single-granularity assessment model
evaluation. A plausible explanation is that the mispronunciation     compared in this paper, which indicates a strong correlation
rates are not uniformly distributed across all phonemes. As a        among assessment tasks within the linguistic hierarchy of an
remedy, the proposed PhnVar takes both the factors of data           utterance. For instance, MuFFIN trained with multi-granularity
quantity and the pronunciation difficulty into account, striking     objectives, i.e., Phone+Word and Phone+Word+Utt.,
a balance between recall and precision to achieve an optimal         outperforms their single-granularity counterparts, i.e., Word
                                                                                                                                               15
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

Only and Utt. Only, with respective gains of 14% and 13%.           scenario and to some extent lacks explainability for the
Furthermore, a comparison of parameter sizes reveals that           provided assessment results. Such scripted-speech assessments
utterance-level assessment models (i.e., Utt. Only and              fail to reflect learners' speaking abilities in real-world
Phone+Word+Utt.) have substantially larger parameter sizes          communication. Furthermore, the experimental dataset solely
than the other assessment models of different granularity           contains the Mandarin learners, potentially hindering the
combinations (e.g., Phone Only, Word Only, and Phone+Word).         generalization abilities and applicability to learners with other
We attribute this to the residual connections between the mean      accents. In future work, we plan to examine the proposed
pooling feature ̅̅E̅ SSL and the utterance-level regressors.        method on spoken language assessment, where learners speak
Effectiveness of Joint MDD and APA Training for MuFFIN.             freely or respond to a given task or question [61]. In addition,
We next ablate the individual contributions of the training         the issues of explainable pronunciation feedback are also left as
objectives for both MDD and APA to the multi-faceted                a future extension.
pronunciation assessment models. Table VIII details the
performance of MuFFIN in jointly addressing the MDD and                                            REFERENCES
APA tasks, where the evaluation metrics are reported for both       [1]  L. Davis, and J. M. Norris, “Assessing second language speaking at ETS:
MDD and the PCC scores on pronunciation accuracy across                  Introduction,” Routledge, 2025. 1-18.
various granularities. A closer look at Table VIII, we have the     [2] A. Van Moere and R. Downey, “Technology and artificial intelligence in
                                                                         language assessment,” in Handbook of Second Language Assessment, D.
following observations. 1) Multi-faceted pronunciation models            Tsagari and Banerjee J., Eds., 2017, pp. 341–357.
(e.g., MDD+Utt., MDD+Word, and MDD+Phone) that                      [3] P. M R.-Revell, “Computer-assisted pronunciation training (CAPT):
integrate APA tasks consistently outperform the model trained            Current issues and future directions,” RELC Journal, vol. 52, pp. 189–
solely on MDD (viz. MDD Only) across all MDD evaluation                  205, 2021.
                                                                    [4] K. Evanini, and X. Wang, “Automated speech scoring for non-native
metrics, demonstrating the synergistic effect of jointly                 middle school students with multiple task types,” in Proc. Interspeech,
modeling MDD and APA. 2) Among these multi-faceted                       2013, pp. 2435–2439.
pronunciation models, the model trained with phoneme-level          [5] Y. K. Singla, A. Gupta, S. Bagga, C. Chen, B. Krishnamurthy, and R. R.
assessment and MDD tasks (MDD+Phone) yields the optimum                  Shah, “Speaker-conditioned hierarchical modelling for automated speech
                                                                         scoring,” in Proc. Int. Conf. Inf. Knowl. Manag., 2021, pp. 1681–1691.
performance in term of the recall metric. 3) Regarding the          [6] K. Evanini and X. Wang, “Automated speech scoring for Nonnative
performance of pronunciation assessment, observations from               middle school students with multiple task types,” in Proc. Interspeech,
Tables VII and VIII suggest that the integration of MDD tasks            2013, pp. 2435–2439.
maintains or slightly improves pronunciation accuracy. Finally,     [7] K. Evanini, M. C. Hauck, and K. Hakuta, “Approaches to automated
                                                                         scoring of speaking for K–12 English language proficiency assessments,”
the primary improvement for the performance of pronunciation             in ETS Research Report Series, pp. 1–11, 2017.
assessment stems from the incorporation of diverse assessment       [8] D. Korzekwa, J. Lorenzo-Trueba, T. Drugman, and B. Kostek, “Computer
tasks at various linguistic levels.                                      assisted pronunciation training—Speech synthesis is almost all you need,”
                                                                         Speech Commun., vol. 142, pp. 22–33, 2022.
                                                                    [9] K. Li, X. Qian, and H. Meng, “Mispronunciation detection and diagnosis
                       VI. CONCLUSION                                    in L2 English speech using multi-distribution deep neural networks,”
In this paper, we have proposed a novel multi-faceted                    IEEE/ACM Trans. Audio Speech Lang. Process., vol. 25, no. 1, pp. 193–
                                                                         207, 2017.
pronunciation feedback model dubbed MuFFIN which is                 [10] S. Bannò, B. Balusu, M. Gales, K. Knill, and K. Kyriakopoulos, “View-
designed to qualify learners' pronunciation from multiple                specific assessment of L2 spoken English,” in Proc. Interspeech, 2022,
perspectives, including pronunciation aspects across various             pp. 4471–4475.
linguistic levels, as well as mispronunciation detection and        [11] N. F. Chen, and H. Li, “Computer-assisted pronunciation training: From
                                                                         pronunciation scoring towards spoken language learning,” in Proc. Asia-
diagnosis at phoneme-level. A novel contrastive phonemic                 Pacific Signal Inf. Process. Assoc. Annu. Summit Conf., 2016, pp. 1–7.
ordinal regularizer has been put forward to empower MuFFIN          [12] S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and
to generate more phoneme-discriminative features while                   assessment for interactive language learning,” Speech Commun., vol. 30,
accounting for the ordinal nature of phoneme-level accuracy              pp. 95–108, 2000.
                                                                    [13] S. Sudhakara, M. K. Ramanathi, C. Yarra, and P. K. Ghosh, “An improved
scores. Furthermore, to tackle the intricate data imbalance              goodness of pronunciation (GOP) measure for pronunciation evaluation
problem of MDD, we present a simple yet effective training               with DNN-HMM system considering hmm transition probabilities,” in
scheme that perturbs the outputs of a phoneme classifier with            Proc. Interspeech, 2019, pp. 954–958.
phoneme-specific variations. This approach effectively              [14] W. Hu, Y. Qian, F. K. Soong, and Y. Wang, “Improved mispronunciation
                                                                         detection with deep neural network trained acoustic models and transfer
balances the distribution of predicted phonemes while                    learning based logistic regression classifiers,” Speech Commun., vol. 67,
incorporating considerations of pronunciation difficulty. The            pp. 154–166, 2015.
practical utility of our method has been verified through           [15] L. Ferrer, H. Bratt, C. Richey, H. Franco, V. Abrash, and K. Precoda,
extensive experiments on speechocen762 benchmark dataset.                “Classification of lexical stress using spectral and prosodic features for
                                                                         computer-assisted language learning systems,” Speech Commun., vol. 69,
The proposed contrastive phonemic ordinal regularizer has                pp. 31–45, 2015.
been thoroughly examined through a series of graphical              [16] E. Coutinho et al., “Assessing the prosody of non-native speakers of
visualizations. Moreover, this study is the first attempt to             English: Measures and feature sets,” in Proc. Lang. Resour. Eval. Conf.,
address the data imbalance in MDD from the perspectives of               2016, pp. 1328–1332.
                                                                    [17] C. Cucchiarini et al., “Quantitative assessment of second language
data quantity and pronunciation difficulty. The empirical results        learners’ fluency by means of automatic speech recognition technology,”
demonstrate that our model outperforms some state-of-the-art             J. Acoust. Soc. of Am., 2000.
methods in both APA and MDD tasks.                                  [18] C. Zhu, T. Kunihara, D. Saito, N. Minematsu, and N. Nakanishi,
Limitations and Future Work. The proposed method is                      “Automatic prediction of intelligibility of words and phonemes produced
constrained by its dependence on the “read-aloud” learning
                                                                                                                                                               16
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

     orally by Japanese learners of English,” in Proc. IEEE Spoken Lang.                diagnosis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process.,
     Technol. Workshop, 2022, pp. 1029–1036.                                            2023, pp. 1–5.
[19] N. F. Chen, D. Wee, R. Tong, B. Ma, and H. Li, “Large-scale                   [41] B.-C. Yan, J.-T. Li, Y.-C. Wang, H.-W. Wang, T.-H. Lo, Y.-C. Hsu, W.-
     characterization of non-native mandarin Chinese spoken by speakers of              C. Chao, and B. Chen. “An effective pronunciation assessment approach
     European origin: Analysis on icall,” Speech Commun., vol. 84, pp. 46–56.           leveraging hierarchical transformers and pre-training strategies,” in Proc.
     2016                                                                               Annu. Meet. Assoc. Comput. Linguist., 2024, pp. 1737-1747.
[20] W. Li, N. F. Chen, S. M. Siniscalchi, and C.-H. Lee, “Improving               [42] H.-C. Pei, H. Fang, X. Luo and X.-S. Xu, “Gradformer: A framework for
     mispronunciation detection of mandarin tones for non-native learners               multi-aspect multi-granularity pronunciation assessment,” IEEE Trans.
     with soft-target tone labels and blstm-based deep tone models,”                    Audio Speech Lang. Process., vol. 32, pp. 554–563, 2024.
     IEEE/ACM Trans. Audio Speech Lang. Process., vol. 27, pp. 2012–2024,          [43] P. Muller, F. De Wet, C. Van Der Walt, and T. Niesler, “Automatically
     2019.                                                                              assessing the oral proficiency of proficient L2 speakers,” in Proc.
[21] K. Fu, J. Lin, D. Ke, Y. Xie, J. Zhang, B. Lin, “A full text-dependent end         Workshop Speech Lang. Technol. Educ., 2009, pp. 29–32.
     to end mispronunciation detection and diagnosis with easy data                [44] H. Franco, H. Bratt, R. Rossier, V. Rao Gadde, E., Shriberg, V. Abrash,
     augmentation techniques,” 2021, arXiv preprint arXiv:2104.08428.                   and K. Precoda, “Eduspeak: A speech recognition and pronunciation
[22] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, amd S. Kumar,           scoring toolkit for computer-aided language learning applications,” Lang.
     “Long-tail learning via logit adjustment,” in Proc. Int. Conf. on Learn.           Test., vol. 27, no. 3, pp. 401–418, 2010.
     Representations, 2021.                                                        [45] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass, “Transformer-based
[23] Y. Wang, J. Fei, H. Wang, W. Li, T. Bao, L. Wu, R. Zhao, Y. Shen,                  multi-aspect multigranularity non-native English speaker pronunciation
     “Balancing logit variation for long-tailed semantic segmentation,” in Proc.        assessment,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process.,
     IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 19561–19573.              2022, pp. 7262–7266.
[24] B.-C. Yan, H.-W. Wang, Y.-C. Wang, J.-T. Li, C.-H. Lin, and B. Chen,          [46] F.-A. Chao, T. H. Lo, T. I. Wu, Y. T. Sung, and Berlin Chen, “3M: An
     “Preserving phonemic distinctions for ordinal regression: a novel loss             effective multi-view, multigranularity, and multi-aspect modelling
     function for automatic pronunciation assessment,” in Proc. IEEE Autom.             approach to English pronunciation assessment,” in Proc. Asia-Pac. Signal
     Speech Recognit. Underst. Workshop, 2023, pp. 1–7.                                 Inf. Process. Assoc. Annu. Summit Conf., 2022, pp. 575–582.
[25] B.-C. Yan, H.-W. Wang, Y.-C. Wang, J.-T. Li, W.-C. Chao, B. Chen              [47] H. Do, Y. Kim, and G. G. Lee, “Hierarchical pronunciation assessment
     “ConPCO: Preserving phoneme characteristics for automatic                          with multi-aspect attention,” in Proc. IEEE Int. Conf. Acoust., Speech,
     pronunciation assessment leveraging contrastive ordinal regularization,”           Signal Process., 2023, pp. 1–5.
     in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2025, pp. 1–5.     [48] F.-A. Chao, T.-H. Lo, T.-I. Wu, Y.-T. Sung, B. Chen, “A hierarchical
[26] J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li, D. Povey,           context-aware modeling approach for multi-aspect and multi-granular
     and Y. Wang, “Speechocean762: An open-source non- native English                   pronunciation assessment,” in Proc. Interspeech, 2023, pp. 974–978.
     speech corpus for pronunciation assessment,” in Proc. Interspeech, pp.        [49] C. Zhu, T. Kunihara, D. Saito, N. Minematsu, N. Nakanishi, “Automatic
     3710–3714, 2021.                                                                   prediction of intelligibility of words and phonemes produced orally by
[27] E. B. Page, “Statistical and linguistic strategies in the computer grading         Japanese learners of English,” in Proc. IEEE Spoken Lang. Technol.
     of essays,” in Proc. Conf. Comput. Linguistics, 1967, pp. 1–13.                    Workshop, 2022, pp. 1029–1036.
[28] M. Wu, K. Li, W.-K. Leung, and H. Meng. “Transformer based end-to-            [50] Y. Shen, A. Yasukagawa, D. Saito, N. Minematsu, and K. Saito,
     end mispronunciation detection and diagnosis,” in Proc. Interspeech,               “Optimized prediction of fluency of L2 English based on interpretable
     2021, pp. 3954–3958.                                                               network using quantity of phonation and quality of pronunciation,” in
[29] S. Bannò and M. Matassoni. “Proficiency assessment of L2 spoken                    Proc. IEEE Spoken Lang. Technol. Workshop, 2021, pp. 698–704.
     English using wav2vec 2.0,” in Proc. IEEE Spoken Lang. Technol.               [51] Y. Peng, S. Dalmia, I. Lane, and S. Watanabe, “Branchformer: Parallel
     Workshop, 2022, pp. 1088–1095.                                                     mlp-attention architectures to capture local and global context for speech
[30] E. Kim, J.-J. Jeon, H. Seo, and H. Kim, “Automatic pronunciation                   recognition and understanding,” in Proc. Int. Conf. Mach. Learn., 2022,
     assessment using self-supervised speech representation learning,” in Proc.         pp. 17627–17643.
     Interspeech, 2022, pp. 1411–1415.                                             [52] A. Radford et al., “Learning transferable visual models from natural
[31] B.-C. Yan, H.-W. Wang, Y.-C. Wang, and B.Chen, “Effective graph-                   language supervision,” in Proc. Int. Conf. Mach. Learn., 2021, pp. 8748–
     based modeling of articulation traits for mispronunciation detection and           8763.
     diagnosis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process.,        [53] B. Elizalde, S. Deshmukh, M. A. Ismail, and H. Wang, “Clap: Learning
     2023, pp. 1–5.                                                                     audio concepts from natural language supervision,” in Proc. IEEE Int.
[32] J. Shi, N. Huo, and Q. Jin, “Context-aware goodness of pronunciation for           Conf. Acoust., Speech, Signal Process.,2023, pp. 1–5.
     computer-assisted pronunciation training,” in Proc. Interspeech, 2020, pp.    [54] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A
     3057–3061.                                                                         framework for self-supervised learning of speech representations,” in
[33] Q.-T. Truong, T. Kato, and S. Yamamoto, “Automatic assessment of L2                Proc. Adv. Neural Inf. Process. Syst., 2020, pp. 12449–12460.
     English word prosody using weighted distances of F0 and intensity             [55] S. Chen et al., “Wavlm: Large-scale self-supervised pre-training for full
     contours,” in Proc. Interspeech, 2018, pp. 2186–2190.                              stack speech processing,” IEEE J. Sel. Topics Signal Process, vol. 16, pp.
[34] C. Graham and F. Nolan, “Articulation rate as a metric in spoken language          1505–1518, 2022.
     assessment,” in Proc. Interspeech, 2019, pp. 3564–3568.                       [56] W.-N. Hsu et al., “HuBERT: Self-supervised speech representation
[35] S. Mao, Z. Wu, R. Li, X. Li, H. Meng, and L. Cai, “Applying multitask              learning by masked prediction of hidden units,” IEEE/ACM Trans. Audio,
     learning to acoustic-phonemic model for mispronunciation detection and             Speech, Lang. Process., pp. 3451–3460, 2021.
     diagnosis in L2 English speech,” in Proc. IEEE Int. Conf. on Acoust.,         [57] B. Lin, and L. Wang, “Deep feature transfer learning for automatic
     Speech, Signal Process., 2018, pp. 6254–6258.                                      pronunciation assessment,” in Proc. Interspeech, 2021. pp. 4438–4442.
[36] B. Lin, L. Wang, X. Feng, and J. Zhang, “Automatic scoring at multi-          [58] B.-C. Yan and B. Chen, “An effective hierarchical graph attention
     granularity for L2 pronunciation.,” in Proc. Interspeech. Assoc., 2020, pp.        network modeling approach for pronunciation assessment,” IEEE/ACM
     3022–3026.                                                                         Trans. Audio Speech Lang. Process., vol. 32, pp. 3974–3985, 2024.
[37] W.-K. Leung, X. Liu and H. Meng, “CNN-RNN-CTC based end-to-end                [59] H. Ryu, S. Kim, and M. Chung, “A joint model for pronunciation
     mispronunciation detection and diagnosis,” in Proc. IEEE Int. Conf. on             assessment and mispronunciation detection and diagnosis with multi-task
     Acoust., Speech, Signal Process., 2018, pp. 8132–8136.                             learning,” in Proc. Interspeech, 2023, pp. 959–963.
[38] B.-C. Yan, M.-C. Wu, H.-T. Hung, and B. Chen, “An end-to-end                  [60] Y.-Y. He, B.-C. Yan, T.-H. Lo, M.-S. Lin, Y.-C. Hsu, B. Chen, "JAM: A
     mispronunciation detection system for L2 English speech leveraging                 unified neural architecture for joint multi-granularity pronunciation
     novel anti-phone modeling,” in Proc. Interspeech, 2020, pp. 3032–3036.             assessment and phone-level mispronunciation detection and diagnosis
[39] B.-C. Yan, H.-W. Wang, and B. Chen, “Peppanet: Effective                           Towards a Comprehensive CAPT System," in Proc. Asia-Pac. Signal Inf.
     mispronunciation detection and diagnosis leveraging phonetic,                      Process. Assoc. Annu. Summit Conf., 2024,
     phonological, and acoustic cues,” in Proc. IEEE Spoken Lang. Technol.         [61] J. Park and S. Choi, “Addressing cold start problem for end-to-end
     Workshop, 2023, pp. 1045–1051.                                                     automatic speech scoring,” in Proc. Interspeech, 2023, pp. 994–998.
[40] B.-C. Yan, H.-W. Wang, Y.-C. Wang, and B.Chen, “Effective graph-
     based modeling of articulation traits for mispronunciation detection and
