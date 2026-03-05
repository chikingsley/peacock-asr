# > REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

1
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
MuFFIN: Multifaceted Pronunciation Feedback
Model with Interactive Hierarchical Neural Modeling
Bi-Cheng Yan, Student Member, IEEE, Ming-Kang Tsai, and Berlin Chen, Member, IEEE
Abstract—Computer-assisted pronunciation training (CAPT)
manages to facilitate second-language (L2) learners to practice
pronunciation skills by offering timely and instructive feedback.
To examine pronunciation proficiency from multiple facets,
existing methods for CAPT broadly fall into two categories:
mispronunciation detection and diagnosis (MDD) as well as
automatic pronunciation assessment (APA). The former aims to
pinpoint phonetic pronunciation errors and provide diagnostic
feedback, while the latter seeks instead to quantify pronunciation
proficiency pertaining to various aspects. Despite the natural
complementarity between MDD and APA, researchers and
practitioners, however, often treat them as independent tasks with
disparate modeling paradigms. In light of this, we in this paper
first introduce MuFFIN, a Multi-Faceted pronunciation Feedback
model with an Interactive hierarchical Neural architecture, to
jointly address the tasks of MDD and APA. To better capture the
nuanced distinctions between phonemes in the feature space, a
novel phoneme-contrastive ordinal regularization mechanism is
thenputforwardtooptimizetheproposedmodeltogeneratemore
phoneme-discriminative features while factoring in the ordinality
of the aspect scores. In addition, to address the intricate data
imbalance problem in MDD, we design a simple yet effective
training objective, which is specifically tailored to perturb the
outputs of a phoneme classifier with the phoneme-specific
variations, so as to better render the distribution of predicted
phonemes meanwhile considering their mispronunciation
characteristics. A series of experiments conducted on the
Speechocean762 benchmark dataset demonstrates the efficacy of
our method in relation to several cutting-edge baselines, showing
state-of-the-art performance on both the APA and MDD tasks.
Index Terms—Computer-assisted pronunciation training,
automatic pronunciation assessment, mispronunciation detection
and diagnosis, multi-aspect and multi-granular pronunciation
assessments, contrastive learning.
I.INTRODUCTION
UELED by the amplified demand for foreign language
acquisition,researchoncomputer-assistedpronunciation
training (CAPT) has aroused significant attention amidst
the tide of globalization, figuring prominently in the field of
computer-assisted language learning (CALL) [1][2]. To bridge
the gap between insufficient supplies and pressing needs from
language teachers and learners, CAPT systems have emerged
as appealing learning tools ubiquitously, shifting the
conventional pedagogical paradigm from teacher-led to selfdirected learning. Beyond their critical roles in education and
language learning, CAPT systems also serve as a handy
referenceforprofessionals(e.g.,interviewersandexaminers)in
high-stakes assessments, with the goals of reducing the
workload [3][4], alleviating the burdens of recruiting new
human experts, and achieving consistent and objective
assessment results [5][6][7].
A de-facto archetype system for CAPT is normally
instantiated in a read-aloud scenario, where an L2 learner is
provided with a reference text and instructed to pronounce it
correctly. By taking the learner’s speech paired with the
reference text as input, CAPT systems are anticipated to assess
the learner’s oral competence from multiple facets, providing
detailed and potentially diagnostic performance feedback with
a near-instant turnaround. To this end, mispronunciation
detection and diagnosis (MDD) and automatic pronunciation
assessment (APA) are two active strands of research in
developing pronunciation feedback modules for CAPT. The
former seeks to pinpoint phonetic pronunciation errors and
provides L2 learners with the corresponding diagnostic
feedback [8][9]. The latter, in contrast, concentrates more on
assessing the learner’s pronunciation quality through multifaceted pronunciation scores, reflecting his/her proficiency
pertainingtospecificaspectsorsomeextentofspokenlanguage
usage [10][11]. One time-tested approach for MDD is goodness
of pronunciation (GOP) and its derivatives [12][13], which
calculate the ratio between the likelihoods of the canonical and
most likely pronounced phonemes. Phoneme-level erroneous
pronunciationsaresubsequentlydetectedifthelikelihoodratios
of certain phoneme segments fail below predetermined
thresholds. On a separate front, the models of iconic APA
methods are typically trained to mimic human ratings based on
surface features (viz. a set of hand-crafted features). These
models either employ a classifier to predict a holistic score
representing learners’ oral proficiency [10] or use regressors to
estimate continuous analytic scores for specific pronunciation
aspects, such as phoneme-level accuracy [14], word-level
lexical stress [15], and utterance-level pronunciation quality
[16][17].
In spite of the complementary nature of MDD and APA,
most existing efforts treat them as independent tasks, thereby
developing two disparate feedback modules for use in CAPT.
However, some prior studies reveal that an L2 English learner
tends to have lower utterance-level assessment scores of
intelligibility and fluency [18] whenever his or her utterances
frequently contain phoneme-level pronunciation errors
[19][20]. In the view of this, we in this paper first propose a
novel CAPT modeling paradigm, dubbed MuFFIN, which is a
F
2
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
Multi-Faceted pronunciation Feedback model with an
Interactive hierarchical Neural structure. MuFFIN unifies the
individual feedback modules of MDD and APA into a
streamlined, hierarchical neural architecture through a multitask learning scheme. Building on a language hierarchy aware
neural architecture with the tailor-made convolutionaugmented Branchformer blocks, MuFFIN can effectively
capture interactions across the linguistic granularities (i.e.,
phoneme, word, and utterance) and preserve fine-grained
articulatory cues at different linguistic units. Next, to render the
subtle differences between phonemes in the feature space, we
introduce a novel phoneme-contrastive ordinal regularizer to
facilitate the proposed model in generating more phonemediscriminative features. This training regime leverages
contrastive learning to better align the phoneme representations
of a scoring model with the textual embeddings of their
corresponding canonical phonemes, while also accounting for
the ordinal relationships among the regression targets (i.e.,
phoneme-level accuracy scores). Furthermore, a simple yet
effective training objective, phoneme-specific variation, is
explored to ease the data imbalance problem incurred by MDD
[21]. Data imbalance is a long-standing problem in MDD,
where phoneme distributions are often skewed between correct
and incorrect pronunciation instances. As illustrated in Fig. 1,
we demonstrate the distributions of vowels with correct and
incorrect pronunciations in the training set of the
Speechocean762 dataset (see Section IV that follows), which
are further categorized into many-shot, medium-shot, and fewshot regions based on their occurrence counts. It is evident that
the occurrence of correct pronunciations significantly exceeds
that of mispronounced ones. Compounding this issue, correct
and incorrect pronunciations exhibit unique long-tailed trends,
respectively. For instance, the vowels /EH/ and /EY/ are
categorized as medium-shot in correct pronunciations but as
many-shot in mispronunciations. Similarly, the vowel /UW/ is
found in the many-shot region of correct pronunciations but
shifts to the few-shot region of mispronunciations.
Typically, a naïve training process of a phoneme-level
pronunciation classifier for MDD is susceptible to an
undesirable bias toward correct pronunciations due to their
higher occurrence frequency, which therefore dominate the
entire training process [22]. As a remedy, our proposed strategy
for modeling phoneme-specific variations is built around the
hypothesis that logits of phoneme categories with higher
occurrence counts (viz. majority phoneme categories) may
occupy a larger portion of the feature space, whereas those of
with lower occurrence counts (viz. minority phoneme
categories) are compressed into a narrower region [23], as
depicted in Fig. 2(a). The proposed training strategy augments
the logits of phoneme predictions with randomly sampled
Gaussian noise, where the radius is determined by the proposed
phoneme-specific variation. To address the intricate data
imbalanceproblemofMDD,themodelingofphoneme-specific
variations comprises two complementary factors: a quantity
factor and a pronunciation difficulty factor. The former assigns
smaller variances to majority phoneme categories and larger
variances to minority phoneme categories. In contrast, the latter
modulates feature areas based on the mispronunciation rates of
phonemes. By doing so, as shown in Fig. 2(b), the synergy of
the two factors not only balances the feature distributions of
disparate phonemes but also adjusts the regions corresponding
to pronunciation difficulties. In summary, this paper presents a
continuation of our previous work described in [24] and [25]
with a significant extension of novel technical contents,
experiments,andanalysis,whosemaincontributionsareatleast
four-fold:
• We present MuFFIN, a multi-faceted pronunciation
feedback model that jointly addresses the tasks of MDD and
APA through an interactive hierarchical neural framework.
Fig. 2. The motivation of the proposed phoneme-specific
variation. In the feature space, each point represents a data
instance predicted by a phoneme classifier, with different
colors indicating distinct categories. (a) The naïve training
process tends to bias toward majority phoneme categories,
leading to the compression of minority phoneme categories
into a narrow area. (b) By applying the proposed phonemespecific variation training strategy, the feature spaces of
minority phoneme categories expand, achieving a more
balanced feature distribution while incorporating
pronunciation difficulty to modulate feature areas.
Phoneme
Category UH AH EH
Feature
Region Many Medium Few
(a) Naïve Training Process (b) Training with Phoneme-specific Variation
Fig. 1. Data imbalance problem of MDD exists in
Speechocean762 dataset ( Many-shot, Medium-shot,
and Few-shot), where the frequencies of correct
pronunciations are notably higher than those of
mispronunciations. Moreover, the correct and incorrect
pronunciationsexhibittwodistinctlong-taileddistributions.
3
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
This model signifies a paradigm shift from separate
modeling of APA and MDD to a unified assessment
approach, opening up a new avenue in CAPT.
• A contrastive phonetic ordinal regularizer is proposed to
align the speech-derived phoneme representations with the
corresponding phoneme-level textual embeddings, while
organically engaging the ordinality of pronunciation
accuracy scores. A series of graphical examinations are
conducted through the lens of the ordinality and phoneme
properties.
• To the best of our knowledge, this is the first attempt to
address data imbalance issues in MDD by incorporating
phoneme-specific variations into the training process. Our
method highlights that the data imbalance problem in MDD
stems from two intertwined and equally crucial factors, viz.
the quantity and the pronunciation difficulty of the training
data. Our empirical findings reveal that addressing the dataimbalance problem of MDD by solely considering the data
quantity factor primarily enhances the recall metric but
sacrifices the precision metric. This analysis has led us to
propose a training strategy that incorporates a pronunciation
difficulty factor, achieving a better balance between recall
and precision metrics compared to strategies that consider
either the quantity factor or the pronunciation difficulty
factor individually.
• Extensive sets of experiments carried out on the
Speechocean762 benchmark dataset [26] confirm the
effectiveness of our proposed methods, which improves the
performance of state-of-the-art ones on both the APA and
MDD tasks.
II.RELATED WORK
Computer-assisted pronunciation training (CAPT) is a
subfield of computer-assisted language learning (CALL),
whose research and development can trace back to pioneering
efforts in the 1960s [27] and have gained significant attention
recently due to the unprecedented advancements in speech and
language technologies [28][29][30]. According to the
diagnostic feedback of CAPT, research endeavors typically fall
into phoneme-level mispronunciation detection and diagnosis
(MDD) as well as automatic pronunciation assessment (APA),
both mostly developed under read-aloud learning scenarios.
A. Mispronunciation Detection and Diagnosis
Mispronunciation detection and diagnosis (MDD) manages
to detect erroneous pronunciation at phoneme segments, and in
turn provide L2 learners with the corresponding diagnostic
feedback [31][32]. Common approaches to MDD can be
grouped into three categories: pronunciation scoring-based,
dictation-based, and prompt-based methods. Pronunciation
scoring-based methods typically exploit various types of
confidence measurements to evaluate pronunciation quality via
a well-trained ASR system (e.g., hybrid DNN-HMM ASR
system). Frequently-used measurements include, but are not
limited to, phoneme durations [33][34], likelihood ratios [13],
phoneme posterior probabilities [35], and their combinations
[36]. Given an input utterance and its corresponding canonical
phoneme sequence (viz. phoneme-level text prompt),
pronunciation scoring-based methods first gauge the
pronunciation scores for each phoneme in the canonical
phoneme sequence. Mispronounced phoneme segments are
then detected when their scores fall below predetermined
thresholds, signifying a deviation from the expected
pronunciation. However, pronunciation scoring-based methods
are untenable to provide diagnostic feedback for the detected
mispronounced phoneme segments. As a remedy, dictationbased methods strive to formulate MDD as a phoneme
recognition task by employing a phoneme recognizer to dictate
themostlikelyphonemesequenceutteredbyanL2learner.The
erroneous pronunciation portions can be easily identified by
comparingthedictationresultwiththe correspondingcanonical
phoneme sequence. For instance, Leung et al. ventured into
employing a CTC-based phoneme recognizer for L2 English
learners, showing comparative performance with pronunciation
scoring-based methods in the mispronunciation detection
subtask, where the performance gains mainly contributed from
the accurate diagnosis of mispronunciations in unvoiced
phoneme segments [37]. Yan et al. exploited the hybrid CTCAttention ASR model as the dictation model and sought to
capture deviant (non-categorical) phoneme productions uttered
by accented L2 learners with anti-phone modeling [38]. Both of
the above-mentioned methods rely on precise alignments to
identify mispronounced segments; however, in practical
applications, alignment errors might arise when comparing the
canonical phoneme sequence to accented or disfluent speech
produced by L2 learners. In response, prompt-based methods
leverage an attention mechanism to derive the soft alignment
between the canonical phoneme sequence and the learner’s
input speech in an end-to-end manner, offering a promising
approach to reduce alignment errors. As one of the first
attempts, PeppaNet aligns canonical phonemes with the
learner’s speech via a Transformer decoder, where any
discrepancies are captured in the matching degree vectors
through end-to-end neural modeling [39]. Among other things,
MDDGCN introduces a graph-based prompt encoder for
canonical phonemes, aiming to improve diagnosis accuracy by
regularizing the relationships between canonical and actually
pronounced phonemes through a pre-defined phonetic graph
[40].
B. Automatic Pronunciation Assessment
Automatic Pronunciation Assessment (APA) quantifies an
L2 learner’s pronunciation proficiency in a target language by
providing either analytic scores (viz. continuous numerical
values) for specific pronunciation aspects [41][42] or a holistic
assessment (viz. discrete categorical values) to reflect overall
spoken competence [10]. Early efforts in APA predominantly
focused on single-aspect assessment, typically by constructing
individual scoring modules to predict proficiency scores at
specific linguistic levels with various sets of hand-crafted
features. These hand-crafted features, extracted from the
learner's input speech or its corresponding ASR-generated
4
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
transcript, may include acoustic features, confidence scores of
recognized linguistic units, time-alignment information, and
statistical measures [43][44]. To scrutinize learners'
pronunciation comprehensively, recent advances in APA have
advocated multi-aspect and multi-granular pronunciation
assessment, leveraging unified scoring models that evaluate
pronunciation proficiency across multiple linguistic levels (viz.
phoneme, word, and utterance) with diverse aspects (e.g.,
accuracy, fluency, and completeness). Drawing on this research
tend, Gong et al. proposed a parallel pronunciation modeling
architecture dubbed GOPT, which took GOP features as input
and adopted a Transformer encoder as the backbone model to
jointly model multiple pronunciation aspects across various
linguistic granularities [45]. Following this school of thought,
3M extended GOPT by augmenting the model’s input
embeddingswithprosodicfeaturesandself-supervisedlearning
(SSL)-based features, aiming to achieve multi-view, multigranularity, and multi-aspect pronunciation modeling [46].
Despite their decent performance, the hierarchical structure of
an utterance is largely set aside. To capture the language
hierarchyofanutterance,Doetal.proposedahierarchicalAPA
model and explored a novel multi-trait attention layer to
strengthen the connection between scoring aspects [47]. Chao
et al. introduced sub-phoneme modeling and employed a depthwise separable convolution layer to construct a hierarchical
APA model, facilitating better modeling of local context cues
at the sub-word level [48]. Apart from the above, Gradformer
leveraged a granularity-decoupled Transformer network that
first separates the granularity of an utterance into lower-level
(phoneme- and word-level) ones and higher-level (utterancelevel) one. A Conformer encoder in turn jointly models
pronunciation aspects at the lower levels, while a Transformer
decoder processes a set of trainable aspect vectors and interacts
with the encoder outputs for utterance-level pronunciation
assessment [42].
III.MULTI-FACETED PRONUNCIATION FEEDBACK MODEL WITH
AN INTERACTIVE HIERARCHICAL NEURAL ARCHITECTURE
The overall architecture of the proposed MuFFIN is
schematically depicted in Fig. 3(a), which contains three main
components: phoneme-level modeling, word-level modeling,
and utterance-level modeling. The encoder at each different
linguistic level adopts a novel convolution-augmented
Branchformer block [25], as shown in Fig. 3(b), which consists
of two branches with one branch designed to capture suprasegmentalpronunciationcueswithmulti-headattention(MHA)
layers while the other tailored to capture fine-grained
pronunciation cues with a series of convolution layers.
Furthermore, as illustrated in Fig. 4, a novel phoneme-level
pronunciation feedback module is devised to assess phonemelevel accuracy and perform mispronunciation detection and
diagnosis.
A. Problem Formulation
Given an input utterance U, consisting of a time sequence
Fig. 3. The proposed multi-faceted pronunciation feedback model with an interactive hierarchical neural architecture
(MuFFIN). (a) The overall model architecture processes input audio and text prompts, hierarchically representing the learner’s
pronunciation to generate assessment scores across various aspects. (b) The proposed convolution-augmented Branchformer
block functions as the backbone of MuFFIN, operating in encoders at different linguistic levels. The pronunciation aspects of
accuracy, fluency, completeness, and prosody are denoted as Acc, Flu, Comp, and Pros, respectively.
LayerNorm LayerNorm
Self-Attention
Dropout
Feed-forward
Dropout
Weighted Comb
LayerNorm
Pointwise Conv
GLUActivation
Depthwise Conv
LayerNorm
ReLUActivation
Pointwise Conv
Dropout
(a) Multi-faceted Pronunciation Feedback Model with an Interactive Hierarchical NeuralArchitecture (MuFFIN)
(b) Convolutionaugmented Branchformer
L K
Word 1 (Call)
Text Prompt (T)
AO T IH
Word 2 (It)
R B
Word 3 (Bear)
EH
InputAudio (X)
0 1 2 3 4 5 6 7
Feature Extraction
Phoneme Encoder
Phoneme-level Pronunciation Feedback Module
Phone
[K]
Feedback
Phone
[L]
Feedback
Phone
[IH]
Feedback
Phone
[B]
Feedback
Phone
[R]
Feedback
Word-level Regressor
Word
[Call]
Acc/Stress/Total
Word
[It]
Acc/Stress/Total
Word
[Bear]
Acc/Stress/Total
CNN CNN CNN
Weighted Merge
Cat & Proj
Phone-level and Word-level Features (X!,H!,H $")
……
……
Utterance Encoder
……
……
……
Atten Pooling
……
Utterance-level Regressor
Utterance
Acc/Flu/Comp/Pros/Total
…
… … …
Phoneme-level Textual Embedding (E!
)
Condensed acoustic features (X"
)
Atten Pooling Atten Pooling Atten Pooling
Cat & Proj Cat & Proj Cat & Proj
Word Encoder
Phone-level Feature Representations (X!
,H!
)
CNNCNN
…
……
Fig. 4. The proposed phoneme-level pronunciation
feedback module, which takes the output of phoneme
encoder and simultaneously generate phoneme-level error
states, diagnostic phonemes, and accuracy scores.
LayerNorm
Feed-forward
LayerNorm
Feed-forward
Softmax
LayerNorm
Feed-forward
Sigmoid
2,0,0,…,2,2
Error Detector Diagnosis Predictor Accuracy Regressor
0, 1, 1,…, 0, 0
Error states
K,AA, O,…, EH, R
Diagnostic Feedback Accuracy Scores
5
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
of audio signals X produced by an L2 learner, and a reference
text prompt T with 𝑀 words, which is converted into 𝑁
canonical phonemes based on a pronunciation dictionary1
, the
proposed multi-faceted pronunciation feedback model aims to
estimate proficiency scores at various linguistic granularities,
while pinpointing phoneme-level pronunciation errors for the
canonical phoneme sequence. Formally, let G = {𝑝,𝑤,𝑢} be a
set of linguistic granularities, where 𝑝,𝑤,𝑢 stands for the
phoneme-, word-, and utterance-level, respectively. For each
granularity 𝑔 ∈ 𝐺, our model aims to predict a set of aspect
score sequences A𝑔
= {𝐚1
𝑔
,𝐚2
𝑔
,…,𝐚𝑁𝑔
𝑔
} , where 𝑁𝑔 is the
number of pronunciation aspects at granularity 𝑔 . In the
meantime, for the canonical phoneme sequence 𝐪 =
(𝑞1,𝑞2,…,𝑞𝑁), the proposed model seeks to detect an error
state sequence 𝐞 = (𝑒1,𝑒2,…,𝑒𝑁) and generate a phonetic
diagnosis sequence 𝐲 = (𝑦1,𝑦2,…,𝑦𝑁). Both 𝑒𝑛 and 𝑦𝑛 are
phoneme-level pronunciation feedback for 𝑞𝑛, where 𝑒𝑛 = 1
denotes a mispronounced phoneme segment and 𝑒𝑛 = 0 for a
correct one, while 𝑦𝑛 specifies the phoneme produced by the
learner.
B. Interactive Hierarchical Neural Modeling
Phoneme-level Modeling. For an input utterance, we first
extract various pronunciation features to portray the
pronunciationqualityoftheL2learneratphoneme-level,which
are then concatenated and projected to obtain a sequence of
condensed acoustic features X𝑝
. The feature extraction process
is formulated as:
X𝑝
= Linear𝑝([EGOP
;EDur
;EEng
;ESSL
]), (1)
where Linear𝑝(⋅) is a single feedforward layer, EGOP
is
goodness of pronunciation (GOP)-based features including log
phoneme posterior (LPP) and log posterior ratio (LPR) [12][14].
EDur
and EEng
are prosodic features related to duration and
energy statistics [49][50], while ESSL
are self-supervised
learning (SSL) based features [46]. We then add phoneme-level
textual embeddings E𝑝
to X𝑝
, followed by a phoneme encoder
to obtain aspect representations H𝑝
= (𝐡1
𝑝
,𝐡2
𝑝
,…,𝐡𝑁
𝑝
):
H0
𝑝
= X𝑝
+ E𝑝
, (2)
H𝑝
= PhnEnc(H0
𝑝
). (3)
Here, E𝑝
is generated by passing 𝐪 into a phoneme-level
prompt encoder which comprises a phoneme and position
embedding layer. PhnEnc(⋅) is composed of a stack of 3
convolution-augmented Branchformer blocks.
Afterward, the pronunciation feedback module builds on
H𝑝
to estimate the multi-faceted pronunciation feedback,
comprising three components: an error detector, a diagnosis
predictor, and an accuracy score regressor. The error detector is
a binary labeling model which predicts the error state 𝑒̂𝑛 ,
indicating whether the 𝑛-th phoneme of 𝐪 is identified as a
mispronunciation:
P𝑑𝑒𝑡(𝑒̂𝑛|𝐪,X) = Sigmoid(Linear𝑑𝑒𝑡(𝐡𝑛
𝑝
)), (4)
1
CMU dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
where Linear𝑑𝑒𝑡(⋅) is a linear layer followed by layer
normalization. The diagnostic predictor performs a sequential
multi-class labeling process to derive the probability
distribution of diagnostic feedback for the 𝑛 -th canonical
phoneme as:
P𝑑𝑖𝑎𝑔(𝑦̂𝑛
∣𝐪,𝑋) = Softmax(Linear𝑑𝑖𝑎𝑔(𝐡𝑁
𝑝
)), (5)
where Linear𝑑𝑖𝑎𝑔(⋅) usedtoconverthiddendimensionsintothe
size of pronunciation dictionary. Finally, the phoneme-level
accuracy score is estimated by an accuracy score regressor.
Word-level Modeling. For the word-level assessments, a
word-level attention pooling is introduced to produce a word
representation vector from its constituent phonemes,
instantiated with a 1-D depth-wise convolution layer followed
by an MHA layer and an average operation. The word-level
input representations X𝑤
are computed by individually passing
X𝑝
and H𝑝
into the word-level attention pooling and
subsequently packing them together with a linear projection:
X ̂𝑤
,H ̂𝑤
= AttPool𝑤1
(X𝑝
),AttPool𝑤2
(H𝑝
), (6)
X𝑤
= Linear𝑤([X ̂𝑤
;H ̂𝑤
]). (7)
Next, the word-level textual embeddings E𝑤
are added to X𝑤
,
and a word encoder is employed to generate word-level
contextualized representations H𝑤
:
H0
𝑤
= X𝑤
+ E𝑤
, (8)
H𝑤
= WordEnc(H0
𝑤
), (9)
where E𝑤
are obtained by mapping the text prompt T into the
corresponding embedding sequence via a word and position
embedding layer, and WordEnc(⋅) consists of 2 convolutionaugmented Branchformer blocks. Finally, three distinct 1-D
depth-wiseconvolutionlayersareperformedon H𝑤
togenerate
word-level aspect representations (i.e., H𝑤1,H𝑤2 , and H𝑤3 ),
which are then transformed into the pronunciation score
sequences by the corresponding word-level regressors.
Utterance-levelModeling.Fortheutterance-levelassessments,
we first extract an utterance-level SSL-based feature E ̅̅̅̅SSL
by
applying average pooling over the time dimension of framelevel SSL-based features. Next, we merge H𝑤1,H𝑤2, and H𝑤3
with a weighted combination to obtain word-level
representations H ̅̅̅̅̅𝑤
. A sequence of utterance-level input
representations H0
𝑢
isobtainedbyfirstapplying1-Ddepth-wise
convolution layers to X𝑝
, H𝑝
, and H ̅̅̅̅̅𝑤
, followed by
concatenation and linear projection. Consequently, an utterance
encoder is exploited to generate contextualized representations
H𝑢
:
H ̅̅̅̅̅𝑤
= Merge(H𝑤1,H𝑤2,H𝑤3), (10)
H0
𝑢
= Linear𝑢([DC1(X𝑝
);DC2(H𝑝
);DC3(H ̅̅̅̅̅𝑤
)]), (11)
H𝑢
= UttEnc(H0
𝑢
), (12)
where Merge(⋅) is a weighted average operation [51],
UttEnc(⋅) is a single convolution-augmented Branchformer
block, and DC1(⋅),DC2(⋅),DC3(⋅) are distinct 1-D depthwise
6
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
convolution layers, each with a kernel size of 3. Afterward, five
separate attention pooling modules are built on top of H!
to
generate utterance-level aspect representation vectors. These
features are then combined with E ̅̅̅̅SSL
via the residual
connections and converted into utterance-level aspect scores
through the respective regressors.
Training Objective. The training objective of the proposed
model is calculated from the losses of APA and MDD:
ℒ𝑀𝑢𝐹𝐹𝐼𝑁 = ℒ𝐴𝑃𝐴 + ℒ𝑀𝐷𝐷. (13)
The APAloss is a weighted sumof the mean square error (MSE)
losses gathered from different granularity levels:
ℒ𝐴𝑃𝐴 = ∑
ℒ𝑝𝑗𝑝
𝑁𝑝 𝑗𝑝
+ ∑
ℒ𝑤𝑗𝑤
𝑁𝑤 𝑗𝑤
+ ∑
ℒ𝑢𝑗𝑢
𝑁𝑢 𝑗𝑢
, (14)
where ℒ𝑝𝑗𝑝 , ℒ𝑤𝑗𝑤 , and ℒ𝑢𝑗𝑢 are phoneme-level, word-level,
and utterance-level losses for disparate aspects, and 𝑁𝑝, 𝑁𝑤,
𝑁𝑢 mark the numbers of aspects at each granularity. On a
separate front, the training objective of MDD comes with the
tasks of mispronunciation detection ℒ𝑑𝑒𝑡 and diagnosis ℒ𝑑𝑖𝑎𝑔:
ℒ𝑀𝐷𝐷 = ℒ𝑑𝑒𝑡 + ℒ𝑑𝑖𝑎𝑔, (15)
ℒ𝑑𝑒𝑡 = −∑logP𝑑𝑒𝑡(𝑒̂𝑛 = 𝑒𝑛|𝐪,X)
𝑁
𝑛=1
, (16)
ℒ𝑑𝑖𝑎𝑔 = −∑logP𝑑𝑖𝑎𝑔(𝑦̂𝑛
= 𝑦𝑛
∣𝐪,X)
𝑁
𝑛=1
, (17)
where ℒ𝑑𝑒𝑡 and ℒ𝑑𝑖𝑎𝑔 represent the negative log-likelihood
used for training the detector and the predictor, respectively.
IV.CONTRASTIVE PHONEMIC ORDINAL REGULARIZER
To generate more phoneme-discriminative features for the
multi-faceted pronunciation assessment model, we proposed
contrastive phonemic ordinal regularizer (ConPCO), which
consists of three mathematical terms: the contrastive term ℒ𝑐𝑜𝑛,
the phonemic characteristic term ℒ𝑝𝑐, and the ordinal term ℒ𝑜.
ℒ𝑐𝑜𝑛 aims to simultaneously project the phoneme
representations generated from a pronunciation assessment
model and the embeddings of phoneme-level text prompt into a
joint feature space. ℒ𝑝𝑐 and ℒ𝑜 adjust the distances between
inter-andintra-phonemecategories,wheretheformerenhances
inter-phoneme discrepancy, and the latter improves intraphoneme compactness with ordinal relationship. The proposed
ConPCO regularizer is formulated as:
ℒ𝐶𝑜𝑛𝑃𝐶𝑂 = ℒ𝑐𝑜𝑛 + ℒ𝑝𝑐 + ℒ𝑜. (18)
Contrastive Term. Let H𝑝
= (𝐡1
𝑝
,𝐡2
𝑝
,…,𝐡𝑁
𝑝
) stand for the
phoneme representation sequence of an utterance generated by
aphonemeencoderinapronunciationscoringmodel,and E𝑝
=
(𝐞1
𝑝
,𝐞2
𝑝
,…,𝐞𝑁
𝑝
) denote the textual embedding of canonical
phonemesgeneratedbyaphoneme-levelpromptencoder.Next,
a set of paired phoneme representations ℳ = {(𝐳𝑖
𝑝
,𝐳𝑖
𝑡
),𝑖 =
1,…,𝑀} is obtained by first applying separate linear
projections to H𝑝
and E𝑝
, and then calculating the centroid
vectors for each phoneme category. Next, as illustrated in Fig.
5, the 𝑀 × 𝑀 similarities are derived from ℳ, with the
contrastive term ℒ𝑐𝑜𝑛 aiming to maximize the similarity
between paired phoneme representations while minimizing the
similarity of unpaired ones [52][53]. The contrastive term ℒ𝑐𝑜𝑛
includes two losses, with a temperature hyper-parameter 𝜏 that
controls the strength of penalties on negative samples:
ℒ𝑐𝑜𝑛 = ℒ𝑝2𝑡 + ℒ𝑡2𝑝, (19)
ℒ𝑝2𝑡 = −
1
𝑀
∑log
exp(𝜙(𝐳𝑖
𝑝
,𝐳𝑖
𝑡
)/𝜏)
∑ exp(𝜙(𝐳𝑖
𝑝
,𝐳𝑗
𝑡
)/𝜏)
𝑀
𝑗=1
𝑀
𝑖=1
, (20)
ℒ𝑡2𝑝 = −
1
𝑀
∑log
exp(𝜙(𝐳𝑖
𝑡
,𝐳𝑖
𝑝
)/𝜏)
∑ exp(𝜙(𝐳𝑖
𝑡,𝐳𝑗
𝑝
)/𝜏)
𝑀
𝑗=1
𝑀
𝑖=1
, (21)
where 𝜙(𝐳𝑖
𝑝
,𝐳𝑗
𝑡
) is dot product between ℓ2 normalized vectors
𝐳𝑖
𝑝
and 𝐳𝑗
𝑡
(cosine similarity). During training, ℳ is
constructed from each batch, where we empirically sample the
data instances with the highest proficiency score to compute
centroid vectors.
Phonemic Characteristic Term. The phonemic characteristic
term ℒ𝑝𝑐 preserve the phonemic proximity information by
minimize the negative distances between centroid vectors 𝐳𝑖
𝑝
:
ℒ𝑝𝑐 = −
1
𝑀(𝑀 − 1)
∑∑∥𝐳𝑖
𝑝
− 𝐳𝑗
𝑝
∥2
𝑖≠𝑗
𝑀
𝑖=1
, (22)
where ℒ𝑝𝑐 is equivalent to maximizing the distances between
phoneme categories during the optimization process.
Ordinal Term. To reflect ordinal relationships of regression
targets in the feature space, the ordinal term ℒ𝑜 is defined to
minimize the distance between the feature representations 𝐡𝑖
𝑝
and their corresponding phoneme centroid vectors 𝐳𝑖
𝑝
with
relative differences of proficiency score:
ℒ𝑜 =
1
𝑁
∑𝑤𝑖‖𝐡𝑖
𝑝
− 𝐳𝑖
𝑝
‖2
𝑁
𝑖=1
, (23)
where 𝑤𝑖 = |𝐶 − 𝑦𝑖
𝑝
| is a compactness weight for each 𝐡𝑖
𝑝
,
reflecting the ordinal behaviors within the label space, with 𝑦𝑖
𝑝
denoting the corresponding phoneme-level accuracy score. The
tunable constant 𝐶 is set to be 3, representing the highest
accuracy score plus a small margin.
Fig. 5. The visualization of the calculation process for the
contrastive term ℒ𝑐𝑜𝑛.
L K AO IH …
!!
"
… !#
"
!$
"
!%
"
!%
&
!$
&
!#
&
...
!!
&
!!
"
.!#
$
… !!
"
.!%
$
!!
"
.!&
$
!!
"
.!!
$
!&
".!#
$
… !&
".!%
$
!&
".!&
$
!&
".!!
$
!%
".!#
$
… !%
".!%
$
!%
".!&
$
!%
".!!
$
…
…
…
...
!#
"
.!#
$
… !#
"
.!%
$
!#
"
.!&
$
!#
"
.!!
$
…
…
Phoneme Encoder
Prompt Encoder
Text Prompt T: Call It Bear
Audio Signals X
Centroid Vector Calculation
Centroid Vector Calculation
7
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
IV.PHONEME-SPECIFIC VARIATION
To balance the distribution of predicted phonemes while
accounting for pronunciation difficulties, the logits of phoneme
predictions generated by a phoneme predictor are perturbed
with randomly sampled Gaussian noise, where the radius is
determined by the phoneme-dependent variance. To this end,
the proposed training scheme, phoneme-specific variation
(PhnVar), consists of two factors, i.e., a data quantity factor and
a pronunciation difficulty factor. The data quantity factor
assigns smaller variance to majority phoneme categories and
larger variance to minority ones, while the pronunciation
difficulty factor modulates feature areas based on
mispronunciationrates.Formally,werevisitEq.(5)andexpress
the probability of the 𝑛-th canonical phoneme being predicted
as a diagnostic phoneme 𝑘, derived from the softmax function:
𝑝𝑘
𝑛
=
exp(𝑔𝑘
𝑛
)
∑ exp(𝑔𝑖
𝑛)
𝑀
𝑖=1
. (24)
Here, 𝑔𝑘
𝑛
is the logit of the 𝑘-th phoneme in logit vector 𝐠𝑛
=
(𝑔1
𝑛
,𝑔2
𝑛
,…,𝑔𝑀
𝑛
), generated by Lineardiag(𝐡𝑁
𝑝
), where 𝑀 is
the number of phoneme categories. We then augment logits
with phoneme-specific variance, defined as the weighted p of
the data quantity factor QF𝑘 and the pronunciation difficulty
factor DF𝑘 for phoneme 𝑘, with coefficients 𝛼 and 𝛽:
𝑔̂𝑘
𝑛
= 𝑔𝑘
𝑛
+ 𝛿(σ) × exp(
𝛼 × log(QF𝑘) + 𝛽 × log(DF𝑘)
𝛼 + 𝛽
),
(25)
where 𝛿(σ) stands for a Gaussian distribution with a zero mean
and the standard deviation σ. Both 𝛼 and 𝛽 are set to 1 in our
experiments. The data quantity factor is defined as normalized
inverse phoneme frequency operated in the logarithmic scale:
QF𝑘
=
𝑐𝑘
max
𝑖
𝑐𝑖
; 𝑐𝑘 = log
∑ 𝑞𝑖
𝑀
𝑖=1
𝑞𝑘
, (26)
where 𝑞# is the number of instances in phoneme category 𝑘.
The pronounce difficulty factor is expressed as normalized
mispronunciation rate:
DF𝑘 =
𝑑𝑘
𝑚𝑎𝑥
𝑖
𝑑𝑖
; 𝑑𝑘 =
𝑚𝑝𝑘
𝑚𝑝𝑘 + 𝑐𝑝𝑘
, (27)
where 𝑚𝑝𝑘 and 𝑐𝑝𝑘 are the number of mispronounced and
correctly pronounced instances for phoneme category 𝑘 ,
respectively.
IV.EXPERIMENTAL SETUPS
A. Experimental Data and Evaluation Metrics
Dataset. A series of experiments were conducted on the
Speechocean762 dataset, a publicly available dataset
specifically designed for research on computer-assisted
language learning [26]. This dataset contains 5,000 Englishspeaking recordings spoken by 250 Mandarin L2 learners. The
training and test sets are of equal size, and each of them has
2,500 utterances. For the APA task, pronunciation proficiency
scores were evaluated at multiple linguistic granularities with
various pronunciation aspects, as the statistics of the APA task
are summarized in Table I. For the MDD task, the phoneme
labels follow the definitions in the CMU pronunciation
dictionary, which includes a set of 39 canonical phonemes. In
Speechocean762, mispronunciation labels were manually
assigned to phoneme segments with accuracy scores below 0.5
and were categorized into four types: deletion, substitution,
non-categorical error, and accented error. Table II summarizes
the phoneme segment statistics for the MDD task.
Evaluation Metrics. The primary evaluation metric for APA
adopts Pearson correlation coefficient (PCC), which measures
thelinearcorrelationbetweenpredictedscoresandground-truth
TABLE I
STATISTICS OF APATASK IN THE SPEECHOCEAN762
Automatic Pronunciation Assessment
Granularity Aspect
Score
Interval
# of Counts
Train Test
Phoneme Accuracy [0, 2] 47,076 47,369
Word
Accuracy
[0, 10] 15,849 15,967 Stress
Total
Utterance
Accuracy
[0, 10] 2,500 2,500
Completeness
Fluency
Prosody
Total
TABLE II
STATISTICS OF MDDTASK IN THE SPEECHOCEAN762
Mispronunciation Detection and Diagnosis
Type Description
Counts
Train Test
Correctness
The uttered phoneme
aligns with the canonical
phoneme
45,088 45,959
Deletion
A canonical phoneme is
omitted
450 396
Substitution
A canonical phoneme is
mispronounced to others
914 593
Noncategorical
Error
The uttered phoneme not
exists in the CMU
pronunciation dictionary
488 332
Accented
Error
A canonical phoneme is
pronounced correctly but
with a strong accent
136 89
Fig. 6. Phoneme statistics of Speechocean762, including
occurrence count and corresponding mispronunciation rate.
High
Medium
Low
8
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
scores. In accordance with prior studies, mean square error
(MSE) is reported for phoneme-level accuracy. On the other
hand, for MDD tasks, the evaluation metrics follow the scoring
rubrics in [9]. Specifically, the mispronunciation detection
subtask is evaluated using recall (RE), precision (PR), and F1-
score (F1), while the mispronunciation diagnosis subtask is
assessed with diagnostic error rate (DER), false rejection rate
(FRR), false acceptance rate (FAR), and phoneme error rate
(PER).
B. Implementation Details
Feature Extraction. For the pronunciation feature extraction,
the GOP features, the energy, and the duration statistics are
adopted in line with our previous studies [24][25]. The
extraction of SSL-based features follows the processing flow
suggested in [46], where features are extracted from the outputs
of pretrained acoustic models, including Wav2vec2.0 [54],
WavLM [55], and HuBERT [56]. The SSL-based and energy
features are extracted at the frame level and then aggregated
into phoneme-level representations based on timestamps of
phoneme segments derived from forced-aligning the learner’s
speech to the reference text. The extracted phoneme-level
proficiency features amount to 3,164 dimensions, comprising
84 dimensions for GOP features EGOP
, 7 for energy statistic
EEng
, 1 for duration value EDur
, and 3,072 for SSL-based
features ESSL
.
Training Configuration. For the training configuration, we
followed to the settings reported in [24][25], where each
experiment consisted of 5 independent trials, and each trial runs
for 100 epochs with different random seeds. In each trial, the
model was trained with an Adam optimizer with an initial
learning rate of 1e-3 and a batch size of 25. A learning rate
scheduler was used to decay the learning rate by a factor of 0.1
aftertheoveralllossdidnotdecreasefor10consecutiveepochs.
Furthermore, our models were initialized with a pretrained
model following the pretraining strategies described in [41].
The reported experimental results were averaged over the 5
trials, with evaluation based on the minimum phoneme-level
MSE.
ModelConfiguration.Thephoneme-level,word-level,andthe
utterance-level encoder (viz. PhnEnc(⋅) , WordEnc(⋅) ,
UttEnc(⋅)) consisted of 3, 2, and 1 convolution-augmented
Branchformer blocks, respectively [25]. Within each encoder
block, the self-attention branch was implemented with a singlehead attention layer, followed by two feed-forward layers. Both
the self-attention and feed-forward layers had a hidden
dimension of 24. Meanwhile, the convolutional branch
consisted of a depth-wise convolutional layer with a 1 × 3
Fig. 7. Visualization of phoneme representations from MuFFIN trained with the phonemic characteristic and the ordinal terms
(i.e., ℒ𝑝𝑐 and ℒ𝑜). For each data point, color indicates the phoneme-level accuracy score, and shape denotes the corresponding
phoneme category. We show the phoneme representations H$
of (a) the vanilla MuFFIN model, (b) MuFFIN+ℒ𝑝𝑐, and (c)
MuFFIN+ℒ𝑝𝑐 + ℒ𝑜.
Fig. 8. Visualization of phoneme representations, with the blue and orange points denoting feature representations H$
and E$
,
generated from the phoneme encoder of MuFFIN and the phoneme-level prompt encoder, respectively. The feature
representations are displayed for (a) the vanilla MuFFIN model, (b) MuFFIN + ℒ𝑐𝑜𝑛 , and (c) a localized zoomed-in
counterpart.
(a) MuFFIN (c) MuFFIN +ℒ!" + ℒ# (b) MuFFIN + ℒ$%
(b) MuFFIN + ℒ!"# (a) MuFFIN (c) Localized Zoomed-in Counterpart
9
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
kernel and a point-wise convolutional layer with a 1 × 1 kernel,
both of which had 24 channels. To aggregate word-level and
utterance-level features, the attention pooling modules were
composed of a depth-wise convolutional layer and a singlehead self-attention layer, where the convolutional layer had 24
channels with a kernel size of 1 × 3 and the attention layer had
a hidden dimension of 24. Furthermore, we set the hidden
dimension of the projection layers (viz. Linear𝑝(⋅), Linear𝑤(⋅),
and Linear𝑢(⋅)) to 24. During the training phase, the tunable
parameters of ℒ𝑝, ℒ𝑤, and ℒ𝑢 in Eq. (14) were set to 3, 1, and
1, respectively, while the temperature factor 𝜏 in Eqs (20) and
(21) was set to 1.
Mispronunciation Detection and Diagnosis via MuFFIN. To
detect mispronunciation segments, MuFFIN follows a
pronunciationscoring-basedparadigm,wheretheoutputsofthe
phoneme-level error detector serve as indicators of
mispronounced segments. The phoneme segments are
identified as mispronounced if the corresponding indicators
exceed a predefined threshold. Subsequently, the detected
mispronunciation segments are fed into the phoneme-level
predictor to generate diagnostic results. To ensure consistency
between the detector and predictor, we mask the canonical
phonemes (i.e., the phonetic transcription of the text prompt)
during the softmax computation of the predictor.
## C Compared Methods
We compare the proposed model (MuFFIN) with three
categories of pronunciation assessment models. 1) Singleaspect pronunciation assessment: Lin2021 is a hierarchical
APA model which takes phoneme-level surface features as
inputs and assesses accuracy scores at utterance-level [57].
Kim2022 relies on layer-wise contextual representations
extracted from a pretrained acoustic model to measure oral
skills in terms of fluency or prosody at the utterance-level [30].
2) Multi-aspect and multi-granular pronunciation assessment:
For the assessment models with parallel neural structures,
GOPT and LSTM are prominent models, both of which
consume a sequence of GOP features and generate a set of
proficiency scores at the phoneme-, word-, and utterance-level
simultaneously [45]. 3M augments the input features of GOPT
with SSL-based features to capture supra-segmental
pronunciation cues, while also integrating phonological
features to enhance phoneme-level textual information [46]. As
for hierarchical models, HierGAT devises a language-hierarchy
aware model with a series of graph attention neural networks
and further strengthening the relatedness among the aspects
with aspect attention mechanisms [58]. 3MH, a previous stateof-the-art methodfor APA, employs 3Mas the backbonemodel
and introduces sup-phoneme modeling to capture finer
articulation traits within the language hierarchy between the
phoneme and word levels [48]. Gradformer (GFR) decouples
the language hierarchy into two sub-levels, i.e., lower
(phoneme and word), and higher (utterance). A Conformer
encoder models aspects at the lower linguistic level, while a
Transformer decoder processes a sequence of learnable aspect
vectors and interacts with the encoder outputs to assess aspect
at the utterance-level [42]. 3) Multi-faceted pronunciation
assessment: Ryu2023 introduces a unified model architecture
that jointly optimizes both phoneme recognition and
pronunciation assessments by independently stacking a CTCbased phoneme recognizer and a set of regressors on top of a
pretrained acoustic model [59]. JAM advances 3M by
integratingaphonemeclassifier topredict diagnosticphonemes
basedoninputcanonicalphonemesandfurtherbooststheMDD
performance by exploiting electromagnetic articulography
(EMA) features to capture the articulatory movements of L2
learners [60].
V.EXPERIMENTAL RESULT
A. Qualitative Analysis
At the outset of experiments, we first analyze the phoneme
statisticsofSpeechocean762torevealtheintricaterelationships
between data quantity and pronunciation difficulty for each
phoneme. Following this, we examine the effectiveness of the
proposed phoneme-level regularizers through a series of
qualitative visualizations.
Phoneme Statistics of Speechocean762. In Fig 6, the
occurrence counts of phoneme segments (blue bars) paired with
their respective mispronunciation rates (orange points) are
Fig. 9. Visualization of phoneme-level logits and decision boundaries from the diagnosis predictor of MuFFIN models trained
with the proposed phoneme-specific variation (PhnVar) scheme. We show the phoneme logits of diagnosis predictor for (a) the
vanilla MuFFIN model, (b) MuFFIN + PhnVar w/o DF (i.e., the variant of PhnVar without accounting for the pronunciation
difficulty factor), and (c) MuFFIN + PhnVar.
(b) MuFFIN + PhnVar w/o DF (a) MuFFIN (c) MuFFIN + PhnVar
10
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
reported for the speechocean762 dataset, where the phonemes
are sorted by mispronunciation rate and then categorized into
threedisjointsubsets:high(mispronunciationrateabove5.1%),
medium (mispronunciation rate between 5.1% and 3.4%), and
low (mispronunciation rate below 3.4%) regions.
InFig6,itisevidentthattheoccurrencecountsofphonemes
and their corresponding mispronunciation rates exhibit distinct
distributional patterns. For example, the high-occurrence
phonemes (e.g., /AH/, /T/, and /N/) are found within the
medium mispronunciation rate region. In contrast, some lowoccurrence phonemes (e.g., /ZH/, /TH/, and /NG/) are often
associated with high mispronunciation rates. Building on this,
to mitigate the data imbalance issue facing the MDD task, the
proposed phoneme-specific variance incorporates two novel
regulation terms: a quantity factor and a pronunciation
difficulty factor. The former balances the feature distributions
of phonemes, while the latter adjusts feature scatteredness
according to the mispronunciation rate.
Qualitative Visualizations of Phoneme Representations for
the Phonemic Characteristic and the Ordinal Terms. In the
second set of experiments, we graphically examining the
impacts of the phonemic characteristic term and the ordinal
term (i.e., ℒ𝑝𝑐 and ℒ𝑜) based on the proposed APA model. As
depicted Fig. 7, we extract the phoneme representations H𝑝
from the test set and visualize each data point pertaining to the
phoneme category (denoted by shape) and the corresponding
pronunciation accuracy score (represented by color).
From Fig. 7(a), it is observed that despite MuFFIN jointly
optimizing both the phoneme recognition and the assessment
tasks, the resulting phoneme representations, however, are
inevitably grouped by phoneme-level accuracy scores,
inadequate to explicitly capturing the subtle distinctions
betweenphonemesinthefeaturespace.WhentrainingMuFFIN
with ℒ𝑝𝑐, as shown in Fig. 7(b), the phoneme-discriminative
features are obtained, where the representations disperse
according to their respective phoneme categories. However,
simply separating the feature representations would omit the
ordinal relationships, which might impede pronunciation
assessment tasks. In response to this, the synergy of ℒ𝑝𝑐 and
TABLE III
PERFORMANCE EVALUATION OF MUFFINAND COMPARATIVE METHODS ON THE APATASK OF SPEECHOCEAN762
Model
Phoneme-level
Acc.
Word-level Aspects Utterance-level Aspects
MSE↓ PCC↑ Acc.↑ Stress↑ Total↑ Acc.↑ Comp.↑ Fluency↑ Prosody↑ Total↑
Lin2021 [57] - - - - - - - - - 0.720
Kim2022 [30] - - - - - - - 0.780 0.770 -
LSTM [45] 0.089
(0.000)
0.591
(0.003)
0.514
(0.003)
0.294
(0.012)
0.531
(0.004)
0.720
(0.002)
0.076
(0.086)
0.745
(0.002)
0.747
(0.005)
0.741
(0.002)
GOPT [45] 0.085
(0.001)
0.612
(0.003)
0.533
(0.004)
0.291
(0.030)
0.549
(0.002)
0.714
(0.004)
0.155
(0.039)
0.753
(0.008)
0.760
(0.006)
0.742
(0.005)
3M [45] 0.078
(0.001)
0.656
(0.005)
0.598
(0.005)
0.289
(0.033)
0.617
(0.005)
0.760
(0.004)
0.325
(0.141)
0.828
(0.006)
0.827
(0.008)
0.796
(0.004)
GFR [42] 0.079
(0.001)
0.646
(0.004)
0.598
(0.006)
0.334
(0.013)
0.614
(0.006)
0.732
(0.005)
0.318
(0.139)
0.769
(0.006)
0.767
(0.004)
0.756
(0.003)
HierGAT [58] 0.073
(0.001)
0.683
(0.004)
0.648
(0.003)
0.327
(0.011)
0.663
(0.002)
0.798
(0.002)
0.531
(0.047)
0.840
(0.002)
0.833
(0.002)
0.821
(0.002)
3MH [48] 0.071
(0.001)
0.693
(0.004)
0.682
(0.005)
0.361
(0.098)
0.694
(0.007)
0.782
(0.003)
0.374
(0.115)
0.843
(0.003)
0.836
(0.004)
0.811
(0.004)
Ryu2023 [59] - - - - - 0.719 - 0.775 0.773 0.743
JAM [60] 0.076
(0.002)
0.664
(0.001)
0.622
(0.012)
0.241
(0.034)
0.638
(0.005)
0.773
(0.007)
0.205
(0.080)
0.831
(0.004)
0.829
(0.004)
0.805
(0.004)
MuFFIN 0.063
(0.002)
0.742
(0.006)
0.705
(0.004)
0.315
(0.033)
0.714
(0.004)
0.807
(0.003)
0.768
(0.049)
0.841
(0.004)
0.832
(0.004)
0.830
(0.002)
* The reported results include the mean PCC scores and standard deviations, calculated over 5 independent experimental trials. Acc. and Comp. refer to the
pronunciation aspects of accuracy and completeness, respectively. The proposed MuFFIN achieves higher PCC scores compared to 3MH across all metrics
except utterance-fluency, with approximate randomization test (𝑝 < 0.001).
TABLE IV
PERFORMANCE EVALUATION OF MUFFINWITH PHONEMELEVEL REGULARIZERS UNDER THE PHONEME-SPECIFIC
VARIATION TRAINING SCHEME
MuFFIN
Phoneme
Score
Word-level Score
PhnVar ℒ𝑐𝑜𝑛 ℒ𝑝𝑐𝑜 Acc. Acc. Stress Total
- - - 0.742 0.705 0.315 0.714
V - - 0.746 0.704 0.310 0.714
V V - 0.749 0.707 0.314 0.718
V - V 0.745 0.703 0.296 0.713
V V V 0.747 0.708 0.341 0.718
*Acc. refer to the pronunciation aspects of accuracy.
11
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
ℒ𝑜 serves as a remedy, which enables the phoneme
representations to reflect both categorical distinctions and
ordinal relationships derived from their accuracy scores, as
shown in Fig. 7(c). Specifically, integrating ℒ𝑜 leads to a
stronger correlation between pairwise distances and phonemelevel accuracy within each phoneme category, resulting in an
outward dispersion in the feature space as accuracy decreases.
Grounded on these observations, incorporating ℒ𝑝𝑐 and ℒ𝑜
during the training process of MuFFIN substantially improves
the discriminability of phoneme representations and
simultaneously reflects the ordinal relationships of the
predicted accuracy scores in the feature space.
Qualitative Visualizations of Phoneme Representations for
the Proposed Contrastive Term. Subsequently, to
qualitatively assess whether the contrastive term ℒ𝑐𝑜𝑛 aligns
the speech-derived representations (colored in blue) with their
corresponding textual embeddings (colored in orange) for
phoneme segments, we visualized the representations H𝑝
and
E𝑝
from MuFFIN on the test set in Fig 8. By comparing among
Fig. 8(a) and Fig. 8(b), we observe that the proposed ℒ𝑐𝑜𝑛
effectively projects these two types of phoneme representations
into a shared feature space, resulting in a more coherent
distribution. Going one step further, a zoomed-in view is
presentedinFig.8(c),whichhighlightsthatthecontrastiveterm
not only aligns the heterogeneous phoneme representations
with the corresponding textual embeddings, but also preserves
the phoneme-specific characteristics across phoneme
categories.
Qualitative Visualizations of Phoneme Logits for the
Proposed Phoneme-specific Variation. Finally, to
qualitatively evaluate the effectiveness of the proposed PhnVar
training scheme, we visualize the phoneme logits and decision
boundaries of the diagnosis predictor. As shown in Figure 9, we
compare the MuFFIN models trained with PhnVar and the
variant (i.e., PhnVar without accounting for the pronunciation
difficulty factor term). Furthermore, the visualized phonemes
(/AH/, /T/, /IH/, and /K/) are uniformly sampled from the test
set, with occurrence counts of 4.4K, 4K, 3K, and 1.3K, and
mispronunciation rates of 4.80%, 3.55%, 5.46%, and 2.39%,
respectively.
The observations from Fig. 9 are highlighted as follows.
First,thelogitsofphonemeswithhigheroccurrencecountstend
to occupy a larger portion of the feature space, while those with
loweroccurrencecountsarecompressedintoanarrowerregion.
This is evidenced by the increasing size of feature regions for
phonemes /K/, /IH/, /T/, to /AH/, which is consistent with their
respective occurrence frequencies. Subsequently, in Fig. 9(b),
it is observed that training MuFFIN with the variant of PhnVar
(viz. PhnVar w/o DF) results in more uniformly distributed
feature regions, independent of phoneme occurrence counts.
However, adjusting the feature space solely factoring in data
quantity factor fails to capture the distribution of
mispronunciations. In light of this, our PhnVar additionally
takes the pronunciation difficulty factor into account. The
phoneme logits of MuFFIN trained with PhnVar are visualized
in Fig. 9(c), where the feature regions are partitioned by the
phoneme mispronunciation rates, with region sizes decreasing
in the order of /IH/, /AH/, /T/, and /K/.
B. Performance of Automatic Pronunciation Assessment
In this subsection, we turn to evaluating the performance of
MuFFIN on pronunciation assessments and compare it with
several state-of-the-art models to validate its effectiveness. The
corresponding results are presented in Table III. We begin by
discussing the experimental results at the phoneme- and wordlevel assessments and then proceed to the utterance-level
assessments.
Assessment Performance at Phoneme and Word Levels. We
first evaluate the assessment performance at the phoneme and
word levels in Table III, from which the following observations
can be made. First, the proposed MuFFIN outperforms other
APA models by a remarkable margin in most pronunciation
assessment tasks, except for the word-level stress. Specifically,
MuFFIN stands out in the phoneme-level accuracy,
demonstratingPCCscoreimprovementsof4.9%and5.9%over
the prior-art models, 3MH and HierGAT, respectively. We
attribute these performance gains to the proposed multi-faceted
phoneme-level pronunciation feedback module, which jointly
optimizes the APA and MDD tasks, thereby encouraging the
phoneme encoder to learn distinct phoneme identities when
evaluating the pronunciation scores. With respect to the wordlevel assessments, MuFFIN generally performs well across
most pronunciation aspects. However, in word-level stress, our
model demonstrates comparable performance against GFR and
HierGAT, while trailing behind 3MH. A possible reason for the
inferior performance is that 3MH leverages sub-phoneme
modeling to create a pseudo (augmented)-linguistic hierarchy
between phoneme and word levels, facilitating better rendering
of supra-segmental information for word-level assessments.
Second, we turn to evaluate the performance of strong
baselines with parallel neural architectures (the second group in
Table III). Compared to LSTM and GOPT, 3M stands out as a
promising method, with its superiority stems from effectively
exploitingSSL-basedfeaturestomitigatethedatascarcityissue
of L2 learners’ speech and simultaneously encapsulate longrange articulatory traits. By augmenting the inputs of 3M with
electromagnetic articulography features, JAM slight boosts the
performance in most pronunciation assessment tasks. Finally,
among the APAmodels with advanced neural architectures (the
third group in Table III), 3MH achieves the best performance,
benefiting from the synergy of hierarchical modeling
approaches and depth-wise convolution layers. However,
compared to MuFFIN, 3MH is limited in functionality, as it
only qualifies pronunciation proficiency with various aspect
scores, lacking phoneme-level diagnostic feedback.
Assessment Performance at Utterance-level. For the
performance of utterance-level pronunciation assessment in
Table III, MuFFIN achieves the highest performance across
most aspects. Compared to 3MH, MuFFIN enhances the PCC
scores by 2.5% in utterance-level accuracy, 2.9% in utterancelevel total, and achieves comparable performance in utterancelevel fluency and prosody. MuFFIN also achieves substantial
improvements in the utterance-level completeness assessment,
a metric reflecting the proportion of correctly pronounced
wordsinanutterance.Thisgainisattributedtothejointtraining
12
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
of the MDD task within the APA model, which consequently
enables MuFFIN to pinpoint mispronounced segments and
identify corresponding phonemes in learners’ speech. By
leveraging phoneme-discriminative representations, MuFFIN
effectively propagates fine-grained information from phonemeto utterance-level assessments through a tailored, hierarchyaware neural architecture.
Next, compared to other strong baseline models, Lin2021
trails behind several APA models trained on multiple
pronunciation aspects (the second group in Table III), revealing
that the single-aspect assessment models fail to leverage the
dependency relationships between aspects through multi-task
learning,therebyleadingtoinferiorperformance.Insubsequent
work, Kim2022 ventures into replacing conventional ASRdrivenfeatureswithSSL-basedfeatures,resultinginsubstantial
improvements and achieving comparable performance to 3M.
In comparison with multi-faceted pronunciation assessment
models (viz. JAM and Ryu2023), JAM demonstrates superior
performance in most aspects of the utterance-level assessment.
This improvement may stem from the novel use of fine-grained
phoneme-level features, including GOP features, prosody
statistics, and EMA features.
Effectiveness of Phoneme-specific Variation and ConPCO.
Lastly, we examine the effectiveness of the proposed training
scheme, phoneme-specific variation (PhnVar), and the
contrastive phonemic ordinal regularizer (ConPCO) for the
pronunciation assessment. In Table IV, we concentrate on the
assessment performance at both the phoneme and word levels,
as we empirically found that the proposed regularizers do not
cause detrimental effects on the utterance-level pronunciation
assessments, with performance either slightly improved or at
least on par with the vanilla MuFFIN model [24][25]. In
addition, ConPCO is decomposed into the contrastive term
(ℒ𝑐𝑜𝑛) and the phonemic ordinal regularizer (ℒ𝑝𝑐𝑜 = ℒ𝑝𝑐 +
ℒ𝑜), both of which are then combined with PhnVar for training
MuFFIN.
From Table IV, we can observe that the proposed training
scheme, PhnVar, the proposed PhnVar training scheme yields a
0.7% improvement in phoneme-level accuracy over the base
model. Subsequently, the incorporation of the phoneme-level
regularizers (viz., ℒ𝑐𝑜𝑛 and ℒ𝑝𝑐𝑜) under the PhnVar training
regime benefits pronunciation assessments, as evidenced by the
sustained or improved results at the phoneme- and word-level
assessment tasks. Furthermore, the contrastive term primarily
boosts the performance in the aspects of phoneme-level
accuracy and word-level total score. In contrast, the phonemic
ordinal regularizer tends to either slightly enhance performance
orretainthat of thevanillaMuFFINmodel. Inaddition,training
MuFFIN with ConPCO attains the best performance in the
word-level assessment tasks (as shown in the last row of Table
IV).
## C Performance of Mispronunciation Detection and
Diagnosis
In this subsection, we evaluate the performance of MDD for the
multi-faceted pronunciation assessment models. As joint
optimization of APA and MDD remains relatively
underexplored in CAPT, only a limited number of relevant
studies are available for comparison in the following
experiments. To the best of our knowledge, Ryu2023 is the first
attempt to develop a multi-faceted pronunciation feedback
model, while JAM represents the recent follow-up work. The
mainresults of the MDDtaskare summarizedinTable V. Next,
Tables VI delve deeper into the imbalance issues inherent in the
MDD, demonstrating the effectiveness of the proposed
phoneme- variation training (PhnVar) scheme.
Performance Evaluation of MDD. To detect
mispronunciations with the MuFFIN, we leverage the outputs
of the phoneme-level error detector to identify
mispronunciation segments. Specifically, we first collect the
detector outputs for all phoneme segments in the training set. A
global threshold is then selected through grid search, with a
stride of 0.1 over a range [0.0,1.0]. At the outset of MDD
experiments, we present the precision-recall curves of MuFFIN
andtheMuFFINtrainedwithPhnVarinFig.10.Forourmodels
(i.e., MuFFIN and MuFFIN+PhnVar) this global threshold is
set to 0.4. In contrast to prior dictation-based methods (e.g.,
Ryu2023 and JAM), which detect mispronunciations by
identifying discrepancies between recognized and canonical
phonemes, MuFFIN adopts a scoring-based approach that
detects mispronunciations via threshold tuning, facilitating
broader adaptability to diverse L2 learners.
As shown in Table V, MuFFIN outperforms other methods
in the mispronunciation detection subtask, achieving
outstanding performance in terms of F1-score and precision.
Moreover, training MuFFIN with the phoneme-specific
variation (PhnVar) leads to notable improvements in all
evaluation metrics compared to the base model. This gain is
further illustrated in Fig. 10, where the orange line
(MuFFIN+PhnVar) exceeds the blue line (MuFFIN) in area
under the precision–recall curve. Subsequently, in comparison
with other baseline methods, Ryu2023, on the basis of a CTCTABLE V
PERFORMANCE EVALUATION OF MUFFINAND COMPARATIVE METHODS ON MDDTASK
Model
Mispronunciation Detection Mispronunciation Diagnosis
PER (%)↓
RE (%)↑ PR (%)↑ F1 (%)↑ FAR (%)↓ FRR (%)↓ DER (%)↓
Ryu2023 [59] 91.60 26.90 41.50 - - - 9.93
JAM [60] 34.76 61.10 45.01 64.32 0.58 45.23 2.81
MuFFIN 64.33 66.89 65.99 35.67 0.97 60.97 2.36
w/ PhnVar 68.37 67.60 67.98 31.63 1.01 58.82 2.33
* For the mispronunciation detection subtask, our best-performing model (MuFFIN+PhnVar) outperforms the
base model (MuFFIN) with significantly better performance (𝑝 < 0.001).
Fig. 10. Precision-recall curves
for MuFFIN trained with PhnVar.
13
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
basedphonemerecognizer,achievesthehighestrecallvaluebut
has the downside of low precision for the mispronunciation
detection task. This limitation is consistent with the
shortcomings reported for dictation-based MDD models
[37][38], where model performance is intrinsically constrained
by the phoneme recognition rate. Instead of a direct freephoneme recognition process, JAM builds upon 3M and detects
mispronunciations in learners’ speech by attaching a phoneme
classifier to the phoneme-level encoder. The corresponding
result demonstrates promising performance in terms of
precision metric, though it struggles with the low recall rate.
ComparedtoJAM,ourMuFFINachievessuperiorperformance
across all metrics in the mispronunciation detection subtask.
Thesefindingscollaborativelyhighlight theeffectiveness ofthe
proposed scoring-based approach for multi-faceted
pronunciation feedback.
For mispronunciation diagnosis subtask, our methods
achieve promising performance in terms of FAR and PER.
However, a trade-off appears to exist between recall and the
metrics of FRR and DER. Specifically, compared to JAM,
MuFFIN achieves higher recall rate and lower PER but exhibits
inferior performance in both FRR and DER. This result implies
that our model detects a greater number of mispronounced
segments but comes at the cost of diagnostic accuracy. We
leave this issue as a direction for future research.
Systematic Examination of the Data Imbalance Problem in
MDD. In the following section, we explore the data imbalance
problem in MDD, which stems from two intertwined factors
within the phoneme segments, i.e., data quantity and
pronunciation difficulty. To disentangle these two factors, we
divide the phoneme segments into two groups and report the
corresponding phoneme error rate (PER), recall, precision, and
F1-score for each subset, with their mean and standard
deviation. As shown in Table VI, in the first group, phoneme
segments are categorized into many (occurrence count above
1.3K), medium (occurrence count between 1.3K and 0.6K), and
few (occurrence count below 0.6K) shot regions based on their
occurrence counts. Conversely, in second group, phoneme
Fig. 11. The performance evaluations highlight data
imbalance issues in MDD. Our bar charts demonstrate that
(a) the data quantity factor primarily affects average PER in
phoneme subsets grouped by occurrence count, and (b) the
pronunciation difficulty factor significantly influences
average recall in phoneme subsets grouped by
mispronunciation rate.
1.19%
2.27%
10.93%
0.00%
2.50%
5.00%
7.50%
10.00%
Many Medium Few
77.43%
69.12% 67.73%
0.00%
20.00%
40.00%
60.00%
80.00%
High Medium Low
Average PER (%) Average Recall (%)
(a) TheAverage PER Evaluation
Phoneme Occurrence Region
(b) TheAverage Recall Evaluation
Pronunciation Error Region
TABLE VI
PROBING THE IMBALANCE ISSUES IN MDDWITH MUFFIN
Group Metrics Average PER (%)↓ Average Recall (%)↑ Average Precision (%)↑ Average F1-score (%)↑
Type Many Med. Few Many Med. Few Many Med. Few Many Med. Few
Occurrence
Count
MuFFIN
1.19
(0.31)
2.27
(0.14)
10.93
(25.79)
70.19
(7.76)
75.53
(15.61)
68.56
(22.75)
50.84
(12.31)
53.42
(19.05)
69.50
(24.79)
57.53
(6.95)
60.14
(14.59)
67.77
(21.43)
PhnVar
1.41
(0.34)
2.41
(1.51)
9.38
(20.19)
61.90
(6.22)
64.04
(17.41)
68.97
(24.66)
66.82
(9.81)
66.27
(16.03)
72.70
(24.14)
63.86
(5.80)
62.71
(10.64)
69.05
(9.38)
w/o DF
1.45
(0.34)
2.42
(1.63)
11.22
(25.83)
64.71
(8.43)
68.62
(23.62)
67.99
(24.77)
61.27
(12.99)
58.87
(14.96)
61.83
(22.68)
63.02
(7.94)
59.23
(14.34)
62.38
(19.55)
w/o QF
1.28
(0.39)
2.09
(1.36)
9.89
(21.73)
57.55
(6.39)
62.72
(22.32)
60.02
(23.27)
69.22
(9.90)
64.04
(19.84)
76.22
(24.85)
62.58
(6.80)
61.23
(18.60)
65.22
(20.65)
Type High Med. Low High Med. Low High Med. Low High Med. Low
Mispron.
Rate
MuFFIN
10.62
(25.82)
1.69
(1.01)
2.08
(2.41)
77.43
(12.19)
69.12
(8.91)
67.73
(23.77)
61.70
(16.77)
55.83
(20.91)
56.23
(24.37)
66.95
(10.94)
59.81
(14.18)
58.68
(20.39)
PhnVar
9.30
(20.13)
1.82
(0.91)
2.15
(2.36)
70.51
(14.30)
62.19
(10.57)
62.22
(24.78)
70.88
(12.35)
67.97
(16.37)
66.95
(23.08)
69.10
(6.93)
64.22
(10.94)
62.30
(20.47)
w/o DF
10.86
(25.75)
1.80
(0.92)
2.42
(3.68)
71.33
(12.52)
73.52
(16.62)
59.16
(26.29)
66.10
(14.05)
54.20
(14.12)
61.67
(21.04)
66.77
(8.10)
60.79
(10.93)
57.08
(20.64)
w/o QF
9.27
(21.65)
1.70
(1.06)
2.29
(3.72)
66.23
(14.28)
55.39
(16.96)
58.67
(23.25)
71.94
(12.39)
70.43
(21.93)
67.11
(23.18)
67.29
(8.42)
61.04
(17.27)
60.69
(20.74)
*Performance gains in the comparison of PhnVar variants (w/o DF and w/o QF) are highlighted in bold, while improvements
over the vanilla MuFFIN achieved by training with PhnVar are marked with underlines. ‘Mispron. Rate’ denotes the
mispronunciation rate.
14
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
segments are categorized into high (mispronunciation rate
above 5.1%), medium (mispronunciation rate between 5.1%
and 3.4%), and low (mispronunciation rate below 3.4%)
pronunciationerrorregionsaccordingtotheirmispronunciation
rates (cf. Fig. 6). Furthermore, this set of experiments is
conducted to assess the upper-bound performance of MuFFIN,
with the aims of analyzing the imbalance issue of MDD and
examining the effectiveness of the proposed phoneme-specific
variance (PhnVar). To this end, a held-out set of 500 utterances
is set aside from the training data, with the remaining 2,000
utterances usedfor model training. Thisheld-out set is designed
to cover both correct and incorrect pronunciations of each
phoneme and is then used to determine phoneme-specific
thresholds by maximizing the area under the precision–recall
curve. Based on these phoneme-specific thresholds, we then
evaluatetheMDDperformanceontheSpeechocean762testset.
To highlight the data imbalance issues, Fig. 11 first presents
two sets of bar charts based on the MuFFIN model, where Fig.
11(a) displays the average PER across phoneme subsets
grouped by occurrence count and Fig. 11(b) shows the average
recall for those grouped by mispronunciation rate. In Fig. 11(a),
we can observe that the average PER of MuFFIN increases
significantly from many-shot to few-shot region. This
observation is consistent with findings from prior studies on
data imbalance learning [22][23], which manifests that
occurrence frequency figures prominently in phoneme
recognition accuracy. A naïve training process for a phoneme
classifier based on empirical risk minimization inevitably
biasesthemodeltowardmajorityphonemecategories,resulting
in inferior performance on minority categories. Apart from the
data quantity issue, Fig. 11(b) reveals that the pronunciation
difficulty factor causes a steady decline in average recall as
phoneme subsets shift from high to low mispronunciation rates.
This suggests that the infrequently mispronounced phoneme
segments pose greater challenges for pronunciation error
detection. Drawing on these observations, the proposed PhnVar
training scheme integrates two mathematical terms designed to
simultaneously account for data quantity and pronunciation
difficulty.
We next investigate the efficacy of each component in
PhnVar for handling the imbalance issues of MDD. The
ablation studies in Table VI are conducted by excluding either
thedataquantityfactor(w/o QF)orthepronunciationdifficulty
factor (w/o DF) from the proposed PhnVar. As shown in the
table, we observe a consistent performance trend across these
two groups, where the MuFFIN model gains in recall by
considering the data quantity factor (w/o DF), while precision
benefits from incorporating the pronunciation difficulty factor
(w/o QF). Subsequently, by factoring in both factors, PhnVar
achieves notable performance gains in F1-score. These results
shed light on the fact that simply balancing the logits of
phoneme predictions encourages the MDD model to detect a
greater number of pronunciation errors. This, however, boosts
the recall rate while resulting in a decrease in precision
evaluation.Aplausibleexplanationisthatthemispronunciation
rates are not uniformly distributed across all phonemes. As a
remedy, the proposed PhnVar takes both the factors of data
quantity and the pronunciation difficulty into account, striking
a balance between recall and precision to achieve an optimal
F1-score. Furthermore, compared to the vanilla MuFFIN model,
training with PhnVar boosts the performance significantly in
precision and F1-score, with gains particularly evident for
phoneme subsets with medium and few occurrences, as well as
those with medium and low mispronunciation rates.
D. Ablation Studies for objectives of APA and MDD
In this subsection, a series of ablation studies are carried out to
analyze the effectiveness of various training objectives for
MuFFIN on both pronunciation accuracy and mispronunciation
detection performance.
Effectiveness of Multi-granularity Pronunciation
Assessments in MuFFIN. In this set of experiments, we begin
by training MuFFIN without the MDD task, then progressively
incorporate pronunciation assessment tasks at different
linguistic levels. Table VII reports on the PCC scores on
assessmentofpronunciationaccuracyforMuFFINwithvarious
training objectives, comprising the assessment tasks at
phoneme, word, utterance levels, as well as cross-granularity
combinations. From Table VII, we observe that MuFFIN
trained in a multi-granularity manner achieves superior results
in relation to any single-granularity assessment model
compared in this paper, which indicates a strong correlation
among assessment tasks within the linguistic hierarchy of an
utterance. For instance, MuFFIN trained with multi-granularity
objectives, i.e., Phone+Word and Phone+Word+Utt.,
outperforms their single-granularity counterparts, i.e., Word
TABLE VII
ABLATION STUDY ON THE VARIED GRANULARITIES IN
TRAINING OBJECTIVES OF APAFOR MUFFIN
Training
Objective
Number
of
Params
Pronunciation Aspects
(PCC Score)
Phone
Accuracy
Word
Accuracy
Utt.
Accuracy
Utt. Only 541K - - 0.782
Word Only 248K - 0.674 -
Phone Only 126K 0.715 - -
+Word 249K 0.724 0.688 -
+Utt. 608K 0.726 0.687 0.807
*In this table,MuFFIN refers to training withoutMDD task.‘Utt.’denotes
the utterance.
TABLE VIII
ABLATION STUDIES ON TRAINING GRANULARITIES FOR
MUFFINWITH APAAND MDDOBJECTIVES
Training
Objective
APA Task MDD Task
Phone
Acc.
Word
Acc.
Utt.
Acc.
F1 (%) RE (%) PR (%)
MDD Only - - - 62.71 65.67 60.33
+Utt. - - 0.787 63.34 63.49 63.45
+Word - 0.681 - 64.46 66.27 62.86
+Phone 0.717 - - 66.26 69.06 63.77
+Word 0.741 0.696 - 66.04 67.08 65.36
+Utt. 0.742 0.705 0.807 65.99 64.33 66.89
‘Utt.’ denotes the utterance and ‘Acc.’ denotes the accuracy aspect.
15
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
Only and Utt. Only, with respective gains of 14% and 13%.
Furthermore, a comparison of parameter sizes reveals that
utterance-level assessment models (i.e., Utt. Only and
Phone+Word+Utt.) have substantially larger parameter sizes
than the other assessment models of different granularity
combinations(e.g.,PhoneOnly,WordOnly,andPhone+Word).
We attribute this to the residual connections between the mean
pooling feature E ̅̅̅̅SSL
and the utterance-level regressors.
Effectiveness of Joint MDD and APA Training for MuFFIN.
We next ablate the individual contributions of the training
objectives for both MDD and APA to the multi-faceted
pronunciation assessment models. Table VIII details the
performance of MuFFIN in jointly addressing the MDD and
APA tasks, where the evaluation metrics are reported for both
MDD and the PCC scores on pronunciation accuracy across
various granularities. A closer look at Table VIII, we have the
following observations. 1) Multi-faceted pronunciation models
(e.g., MDD+Utt., MDD+Word, and MDD+Phone) that
integrate APA tasks consistently outperform the model trained
solely on MDD (viz. MDD Only) across all MDD evaluation
metrics, demonstrating the synergistic effect of jointly
modeling MDD and APA. 2) Among these multi-faceted
pronunciation models, the model trained with phoneme-level
assessment and MDD tasks (MDD+Phone) yields the optimum
performance in term of the recall metric. 3) Regarding the
performance of pronunciation assessment, observations from
Tables VII and VIII suggest that the integration of MDD tasks
maintains or slightly improves pronunciation accuracy. Finally,
the primary improvement for the performance of pronunciation
assessment stems from the incorporation of diverse assessment
tasks at various linguistic levels.
VI.CONCLUSION
In this paper, we have proposed a novel multi-faceted
pronunciation feedback model dubbed MuFFIN which is
designed to qualify learners' pronunciation from multiple
perspectives, including pronunciation aspects across various
linguistic levels, as well as mispronunciation detection and
diagnosis at phoneme-level. A novel contrastive phonemic
ordinal regularizer has been put forward to empower MuFFIN
to generate more phoneme-discriminative features while
accounting for the ordinal nature of phoneme-level accuracy
scores. Furthermore, to tackle the intricate data imbalance
problem of MDD, we present a simple yet effective training
scheme that perturbs the outputs of a phoneme classifier with
phoneme-specific variations. This approach effectively
balances the distribution of predicted phonemes while
incorporating considerations of pronunciation difficulty. The
practical utility of our method has been verified through
extensive experiments on speechocen762 benchmark dataset.
The proposed contrastive phonemic ordinal regularizer has
been thoroughly examined through a series of graphical
visualizations. Moreover, this study is the first attempt to
address the data imbalance in MDD from the perspectives of
dataquantityandpronunciationdifficulty.Theempiricalresults
demonstrate that our model outperforms some state-of-the-art
methods in both APA and MDD tasks.
Limitations and Future Work. The proposed method is
constrained by its dependence on the “read-aloud” learning
scenario and to some extent lacks explainability for the
provided assessment results. Such scripted-speech assessments
fail to reflect learners' speaking abilities in real-world
communication. Furthermore, the experimental dataset solely
contains the Mandarin learners, potentially hindering the
generalization abilities and applicability to learners with other
accents. In future work, we plan to examine the proposed
method on spoken language assessment, where learners speak
freely or respond to a given task or question [61]. In addition,
the issues of explainable pronunciation feedback are also left as
a future extension.
## REFERENCES
[1] L. Davis, and J. M. Norris, “Assessing second language speaking at ETS:
Introduction,” Routledge, 2025. 1-18.
[2] A. Van Moere and R. Downey, “Technology and artificial intelligence in
language assessment,” in Handbook of Second Language Assessment, D.
Tsagari and Banerjee J., Eds., 2017, pp. 341–357.
[3] P. M R.-Revell, “Computer-assisted pronunciation training (CAPT):
Current issues and future directions,” RELC Journal, vol. 52, pp. 189–
205, 2021.
[4] K. Evanini, and X. Wang, “Automated speech scoring for non-native
middle school students with multiple task types,” in Proc. Interspeech,
2013, pp. 2435–2439.
[5] Y. K. Singla, A. Gupta, S. Bagga, C. Chen, B. Krishnamurthy, and R. R.
Shah, “Speaker-conditioned hierarchical modelling for automated speech
scoring,” in Proc. Int. Conf. Inf. Knowl. Manag., 2021, pp. 1681–1691.
[6] K. Evanini and X. Wang, “Automated speech scoring for Nonnative
middle school students with multiple task types,” in Proc. Interspeech,
2013, pp. 2435–2439.
[7] K. Evanini, M. C. Hauck, and K. Hakuta, “Approaches to automated
scoring of speaking for K–12 English language proficiency assessments,”
in ETS Research Report Series, pp. 1–11, 2017.
[8] D. Korzekwa, J. Lorenzo-Trueba, T. Drugman, and B. Kostek, “Computer
assistedpronunciationtraining—Speechsynthesisisalmostallyouneed,”
Speech Commun., vol. 142, pp. 22–33, 2022.
[9] K. Li, X. Qian, and H. Meng, “Mispronunciation detection and diagnosis
in L2 English speech using multi-distribution deep neural networks,”
IEEE/ACM Trans. Audio Speech Lang. Process., vol. 25, no. 1, pp. 193–
207, 2017.
[10] S. Bannò, B. Balusu, M. Gales, K. Knill, and K. Kyriakopoulos, “Viewspecific assessment of L2 spoken English,” in Proc. Interspeech, 2022,
pp. 4471–4475.
[11] N. F. Chen, and H. Li, “Computer-assisted pronunciation training: From
pronunciation scoring towards spoken language learning,” in Proc. AsiaPacific Signal Inf. Process. Assoc. Annu. Summit Conf., 2016, pp. 1–7.
[12] S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and
assessment for interactive language learning,” Speech Commun., vol. 30,
pp. 95–108, 2000.
[13] S.Sudhakara,M.K.Ramanathi,C.Yarra,andP.K.Ghosh,“Animproved
goodness of pronunciation (GOP) measure for pronunciation evaluation
with DNN-HMM system considering hmm transition probabilities,” in
Proc. Interspeech, 2019, pp. 954–958.
[14] W. Hu, Y. Qian, F. K. Soong, and Y. Wang, “Improved mispronunciation
detection with deep neural network trained acoustic models and transfer
learning based logistic regression classifiers,” Speech Commun., vol. 67,
pp. 154–166, 2015.
[15] L. Ferrer, H. Bratt, C. Richey, H. Franco, V. Abrash, and K. Precoda,
“Classification of lexical stress using spectral and prosodic features for
computer-assisted language learning systems,” Speech Commun., vol. 69,
pp. 31–45, 2015.
[16] E. Coutinho et al., “Assessing the prosody of non-native speakers of
English: Measures and feature sets,” in Proc. Lang. Resour. Eval. Conf.,
2016, pp. 1328–1332.
[17] C. Cucchiarini et al., “Quantitative assessment of second language
learners’ fluency by means of automatic speech recognition technology,”
J. Acoust. Soc. of Am., 2000.
[18] C. Zhu, T. Kunihara, D. Saito, N. Minematsu, and N. Nakanishi,
“Automatic prediction of intelligibility of words and phonemes produced
16
> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <
orally by Japanese learners of English,” in Proc. IEEE Spoken Lang.
Technol. Workshop, 2022, pp. 1029–1036.
[19] N. F. Chen, D. Wee, R. Tong, B. Ma, and H. Li, “Large-scale
characterization of non-native mandarin Chinese spoken by speakers of
European origin: Analysis on icall,” Speech Commun., vol. 84, pp. 46–56.
2016
[20] W. Li, N. F. Chen, S. M. Siniscalchi, and C.-H. Lee, “Improving
mispronunciation detection of mandarin tones for non-native learners
with soft-target tone labels and blstm-based deep tone models,”
IEEE/ACM Trans. Audio Speech Lang. Process., vol. 27, pp. 2012–2024,
2019.
[21] K. Fu, J. Lin, D. Ke, Y. Xie, J. Zhang, B. Lin, “A full text-dependent end
to end mispronunciation detection and diagnosis with easy data
augmentation techniques,” 2021, arXiv preprint arXiv:2104.08428.
[22] A.K.Menon,S.Jayasumana,A.S.Rawat,H.Jain,A.Veit,amdS.Kumar,
“Long-tail learning via logit adjustment,” in Proc. Int. Conf. on Learn.
Representations, 2021.
[23] Y. Wang, J. Fei, H. Wang, W. Li, T. Bao, L. Wu, R. Zhao, Y. Shen,
“Balancing logit variation for long-tailed semantic segmentation,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 19561–19573.
[24] B.-C. Yan, H.-W. Wang, Y.-C. Wang, J.-T. Li, C.-H. Lin, and B. Chen,
“Preserving phonemic distinctions for ordinal regression: a novel loss
function for automatic pronunciation assessment,” in Proc. IEEE Autom.
Speech Recognit. Underst. Workshop, 2023, pp. 1–7.
[25] B.-C. Yan, H.-W. Wang, Y.-C. Wang, J.-T. Li, W.-C. Chao, B. Chen
“ConPCO: Preserving phoneme characteristics for automatic
pronunciation assessment leveraging contrastive ordinal regularization,”
in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2025, pp. 1–5.
[26] J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li, D. Povey,
and Y. Wang, “Speechocean762: An open-source non- native English
speech corpus for pronunciation assessment,” in Proc. Interspeech, pp.
3710–3714, 2021.
[27] E. B. Page, “Statistical and linguistic strategies in the computer grading
of essays,” in Proc. Conf. Comput. Linguistics, 1967, pp. 1–13.
[28] M. Wu, K. Li, W.-K. Leung, and H. Meng. “Transformer based end-toend mispronunciation detection and diagnosis,” in Proc. Interspeech,
2021, pp. 3954–3958.
[29] S. Bannò and M. Matassoni. “Proficiency assessment of L2 spoken
English using wav2vec 2.0,” in Proc. IEEE Spoken Lang. Technol.
Workshop, 2022, pp. 1088–1095.
[30] E. Kim, J.-J. Jeon, H. Seo, and H. Kim, “Automatic pronunciation
assessment using self-supervised speech representation learning,” in Proc.
Interspeech, 2022, pp. 1411–1415.
[31] B.-C. Yan, H.-W. Wang, Y.-C. Wang, and B.Chen, “Effective graphbased modeling of articulation traits for mispronunciation detection and
diagnosis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process.,
2023, pp. 1–5.
[32] J. Shi, N. Huo, and Q. Jin, “Context-aware goodness of pronunciation for
computer-assisted pronunciation training,” in Proc. Interspeech, 2020, pp.
3057–3061.
[33] Q.-T. Truong, T. Kato, and S. Yamamoto, “Automatic assessment of L2
English word prosody using weighted distances of F0 and intensity
contours,” in Proc. Interspeech, 2018, pp. 2186–2190.
[34] C. Graham and F. Nolan, “Articulation rate as a metric in spoken language
assessment,” in Proc. Interspeech, 2019, pp. 3564–3568.
[35] S. Mao, Z. Wu, R. Li, X. Li, H. Meng, and L. Cai, “Applying multitask
learning to acoustic-phonemic model for mispronunciation detection and
diagnosis in L2 English speech,” in Proc. IEEE Int. Conf. on Acoust.,
Speech, Signal Process., 2018, pp. 6254–6258.
[36] B. Lin, L. Wang, X. Feng, and J. Zhang, “Automatic scoring at multigranularity for L2 pronunciation.,” in Proc. Interspeech. Assoc., 2020, pp.
3022–3026.
[37] W.-K. Leung, X. Liu and H. Meng, “CNN-RNN-CTC based end-to-end
mispronunciation detection and diagnosis,” in Proc. IEEE Int. Conf. on
Acoust., Speech, Signal Process., 2018, pp. 8132–8136.
[38] B.-C. Yan, M.-C. Wu, H.-T. Hung, and B. Chen, “An end-to-end
mispronunciation detection system for L2 English speech leveraging
novel anti-phone modeling,” in Proc. Interspeech, 2020, pp. 3032–3036.
[39] B.-C. Yan, H.-W. Wang, and B. Chen, “Peppanet: Effective
mispronunciation detection and diagnosis leveraging phonetic,
phonological, and acoustic cues,” in Proc. IEEE Spoken Lang. Technol.
Workshop, 2023, pp. 1045–1051.
[40] B.-C. Yan, H.-W. Wang, Y.-C. Wang, and B.Chen, “Effective graphbased modeling of articulation traits for mispronunciation detection and
diagnosis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process.,
2023, pp. 1–5.
[41] B.-C. Yan, J.-T. Li, Y.-C. Wang, H.-W. Wang, T.-H. Lo, Y.-C. Hsu, W.-
C. Chao, and B. Chen. “An effective pronunciation assessment approach
leveraging hierarchical transformers and pre-training strategies,” in Proc.
Annu. Meet. Assoc. Comput. Linguist., 2024, pp. 1737-1747.
[42] H.-C. Pei, H. Fang, X. Luo and X.-S. Xu, “Gradformer: A framework for
multi-aspect multi-granularity pronunciation assessment,” IEEE Trans.
Audio Speech Lang. Process., vol. 32, pp. 554–563, 2024.
[43] P. Muller, F. De Wet, C. Van Der Walt, and T. Niesler, “Automatically
assessing the oral proficiency of proficient L2 speakers,” in Proc.
Workshop Speech Lang. Technol. Educ., 2009, pp. 29–32.
[44] H. Franco, H. Bratt, R. Rossier, V. Rao Gadde, E., Shriberg, V. Abrash,
and K. Precoda, “Eduspeak: A speech recognition and pronunciation
scoring toolkit for computer-aided language learning applications,” Lang.
Test., vol. 27, no. 3, pp. 401–418, 2010.
[45] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass, “Transformer-based
multi-aspect multigranularity non-native English speaker pronunciation
assessment,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process.,
2022, pp. 7262–7266.
[46] F.-A. Chao, T. H. Lo, T. I. Wu, Y. T. Sung, and Berlin Chen, “3M: An
effective multi-view, multigranularity, and multi-aspect modelling
approach to English pronunciation assessment,” in Proc. Asia-Pac. Signal
Inf. Process. Assoc. Annu. Summit Conf., 2022, pp. 575–582.
[47] H. Do, Y. Kim, and G. G. Lee, “Hierarchical pronunciation assessment
with multi-aspect attention,” in Proc. IEEE Int. Conf. Acoust., Speech,
Signal Process., 2023, pp. 1–5.
[48] F.-A. Chao, T.-H. Lo, T.-I. Wu, Y.-T. Sung, B. Chen, “A hierarchical
context-aware modeling approach for multi-aspect and multi-granular
pronunciation assessment,” in Proc. Interspeech, 2023, pp. 974–978.
[49] C. Zhu, T. Kunihara, D. Saito, N. Minematsu, N. Nakanishi, “Automatic
prediction of intelligibility of words and phonemes produced orally by
Japanese learners of English,” in Proc. IEEE Spoken Lang. Technol.
Workshop, 2022, pp. 1029–1036.
[50] Y. Shen, A. Yasukagawa, D. Saito, N. Minematsu, and K. Saito,
“Optimized prediction of fluency of L2 English based on interpretable
network using quantity of phonation and quality of pronunciation,” in
Proc. IEEE Spoken Lang. Technol. Workshop, 2021, pp. 698–704.
[51] Y. Peng, S. Dalmia, I. Lane, and S. Watanabe, “Branchformer: Parallel
mlp-attention architectures to capture local and global context for speech
recognition and understanding,” in Proc. Int. Conf. Mach. Learn., 2022,
pp. 17627–17643.
[52] A. Radford et al., “Learning transferable visual models from natural
language supervision,” in Proc. Int. Conf. Mach. Learn., 2021, pp. 8748–
8763.
[53] B. Elizalde, S. Deshmukh, M. A. Ismail, and H. Wang, “Clap: Learning
audio concepts from natural language supervision,” in Proc. IEEE Int.
Conf. Acoust., Speech, Signal Process.,2023, pp. 1–5.
[54] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A
framework for self-supervised learning of speech representations,” in
Proc. Adv. Neural Inf. Process. Syst., 2020, pp. 12449–12460.
[55] S. Chen et al., “Wavlm: Large-scale self-supervised pre-training for full
stack speech processing,” IEEE J. Sel. Topics Signal Process, vol. 16, pp.
1505–1518, 2022.
[56] W.-N. Hsu et al., “HuBERT: Self-supervised speech representation
learning by masked prediction of hidden units,” IEEE/ACM Trans. Audio,
Speech, Lang. Process., pp. 3451–3460, 2021.
[57] B. Lin, and L. Wang, “Deep feature transfer learning for automatic
pronunciation assessment,” in Proc. Interspeech, 2021. pp. 4438–4442.
[58] B.-C. Yan and B. Chen, “An effective hierarchical graph attention
network modeling approach for pronunciation assessment,” IEEE/ACM
Trans. Audio Speech Lang. Process., vol. 32, pp. 3974–3985, 2024.
[59] H. Ryu, S. Kim, and M. Chung, “A joint model for pronunciation
assessment and mispronunciation detection and diagnosis with multi-task
learning,” in Proc. Interspeech, 2023, pp. 959–963.
[60] Y.-Y. He, B.-C. Yan, T.-H. Lo, M.-S. Lin, Y.-C. Hsu, B. Chen, "JAM: A
unified neural architecture for joint multi-granularity pronunciation
assessment and phone-level mispronunciation detection and diagnosis
Towards a Comprehensive CAPT System," in Proc. Asia-Pac. Signal Inf.
Process. Assoc. Annu. Summit Conf., 2024,
[61] J. Park and S. Choi, “Addressing cold start problem for end-to-end
automatic speech scoring,” in Proc. Interspeech, 2023, pp. 994–998.
