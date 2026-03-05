# Prior efforts in building computer-assisted

## Abstract
Prior efforts in building computer-assisted
pronunciation training (CAPT) systems
often treat automatic pronunciation
assessment (APA) and mispronunciation
detection and diagnosis (MDD) as separate
fronts: the former aims to provide multiple
pronunciation aspect scores across diverse
linguistic levels, while the latter focuses
instead on pinpointing the precise phonetic
pronunciation errors made by non-native
language learners. However, it is generally
expected that a full-fledged CAPT system
should perform both functionalities
simultaneously and efficiently. In response
tothissurgingdemand,weinthisworkfirst
propose HMamba, a novel CAPT
approach that seamlessly integrates APA
and MDD tasks in parallel. In addition, we
introduce a novel loss function, decoupled
cross-entropy loss (deXent), specifically
tailored for MDD to facilitate bettersupervised learning for detecting
mispronounced phones, thereby enhancing
overall performance. A comprehensive set
of empirical results on the speechocean762
benchmark dataset demonstrates the
effectiveness of our approach on APA.
Notably, our proposed approach also yields
a considerable improvement in MDD
performance over a strong baseline,
achieving an F1-score of 63.85%. Our
codes are made available at
https://github.com/Fuann/hma
mba
## 1 Introduction
In this era of globalization and technologization,
computer-assisted pronunciation training (CAPT)
systems have emerged as an appealing alternative
to meet the pressing need for second language (L2)
learning.Incomparisonwithtraditionalcurriculum
learning, CAPT offers advantages in both time
efficiencyandcost-effectiveness.Morecritically,it
shifts the conventional pedagogical paradigm from
teacher-directed to self-directed learning, thereby
providing a stress-free environment for L2 learners
(Eskenazi et al., 2009). In addition, CAPT
applications have achieved marked success in
various commercial sectors and testing services,
such as the APPs of Duolingo (McCarthy et al.,
2021) and the SpeechRater (Zechner et al., 2009)
developed by Educational Testing Service (ETS).
Typically, a de-facto archetype system for CAPT
encompasses a “reading-aloud” scenario, where a
non-native speaker is given a text prompt and
instructed to pronounce it correctly. In this context,
previous literature broadly divides applications of
CAPT into two categories: automatic
pronunciation assessment (APA) and
Towards Efficient and Multifaceted Computer-assisted Pronunciation
Training Leveraging Hierarchical Selective State Space Model
and Decoupled Cross-entropy Loss
Fu-An Chao, Berlin Chen
Department of Computer Science and Information Engineering
National Taiwan Normal University
{fuann, berlin}@ntnu.edu.tw
Figure 1: A running example depicts the evaluation
differences between APA and MDD systems in the
reading-aloud scenario.
mispronunciation detection and diagnosis (MDD),
with each category dedicated to specific facets of
pronunciation training. APA aims to evaluate the
spoken proficiency of L2 learners by providing
fine-grained feedback on various aspect
assessments (e.g., accuracy and fluency) across
multiple linguistic levels (e.g., word and utterance
level) (Kheir et al., 2023). To assess L2 learners’
spoken proficiency,APAsystems typically employ
scoringmodels thatareeitherjointlytrained (Gong
et al., 2022; Chao et al., 2022) or leverage multiple
regressors in an ensemble paradigm (Bannò et al.,
2022a; Bannò and Matassoni, 2022b) to generate
scores across various aspects. As such, users can
receive multi-aspect assessment scores predicted
by an APA system, as illustrated in the example
shown in Figure 1. In contrast to APA, MDD
focuses more on non-native speakers’ phonetic
pronunciation errors (Chen and Li, 2016). These
errors usually have clear-cut distinctions between
correct and incorrect ones, and can be easily
quantified through deletions, substitutions, and
insertions. For instance, a number of MDD models
are designed to capitalize on classifier-based
modeling (Truong et al., 2004; Strik et al., 2009;
Harrison et al. 2009), enabling precise
identification of the exact positions where
pronunciation errors occur within an utterance.
This capability provides L2 learners with specific
feedback on discrepancies between intended
pronunciation and actual pronunciation.
Albeit the phonetic (segmental) errors are
crucial in the beginning stages of non-native
language learning, prosodic (suprasegmental)
errors may often cause a detrimental impact on the
perceptionoffluencyandleadtopoorintelligibility
(Chen and Li, 2016). This effect may be more
pronounced in learning stress-timed languages like
English, especially for a learner whose mother
tongue is a syllable-timed language, such as
Chinese (Ding and Xu, 2016). To tackle this
problem, APA can play a pivotal role by offering
prosodic or intonation assessment for L2 learners.
For example, Lin et al. (2021a) introduced rhythm
rubrics to predict the traits of sentence-level stress
in L2 English utterances, demonstrating a strong
correlation with the prosody scores assessed by the
human experts. In addition, Arias et al. (2010)
proposed text-independent systems for assessing
intonation and stress, focusing on measuring the
similaritybetweenatest-taker’sintonationorstress
curve and that of a reference response.
Onthesegrounds,itisevidentthatbothAPAand
MDD are indispensable ingredients of CAPT,
playing complementary roles in its success.
However, previous studies on APA and MDD
appear to have developed independently, with
limitedresearchexploringtheirsynergeticuse.Ryu
et al. (2023) proposed a joint model for APA and
MDD, leveraging knowledge transfer and multitask learning.Their findings indicate high negative
correlationsbetweenseveralassessmentscoresand
mispronunciations. This also suggests that the
human assessors may be influenced by phonetic
errors when evaluating overall proficiency scores
for various aspects, which to some extent gives
away the halo effect present in the human
annotations. While the corresponding results show
that jointly modeling both tasks can achieve better
performance than modeling each task in isolation,
only utterance-level holistic assessments are
consideredintheirexperiments.Inordertoprovide
morecomprehensiveandfine-grainedfeedbackfor
L2 learners, other granularities, such as the phone
level or the word level, should also be aptly
modeled.Recognizingthisimportance,wepropose
HMamba, a more effective approach, for
multifaceted CAPT. Being aware of the linguistic
hierarchy,HMambacancapturetheintrinsicmultilayered speech structure, delivering both coarse
and fine-grained pronunciation assessments while
offering more accurate diagnostic feedback of
mispronunciations.Inaddition, toaddresstheextra
computational costs introduced by multi-task
learning,HMambaleveragesaselectivestatespace
model (SSM) that can efficiently tackle both APA
and MDD tasks in parallel.The main contributions
of this paper can be summarized as follows:
• We introduce HMamba, a unified and
linguistically hierarchy-aware model that jointly
tacklesAPAand MDD tasks, achieving superior
overall performance compared to prior arts that
employ either single-task or multi-task models.
• We propose a novel loss function, decoupled
cross-entropy loss (termed deXent), which
effectively addresses the inherent issue of text
prompt-aware MDD methods. Notably, deXent
is feasible and well-suited for optimizing the
MDD performance, particularly in striking the
balance between precision and recall.
• To the best of our knowledge, this is the first
worktoadoptandextendMambaintheAPAand
MDD tasks for a more efficient and
comprehensive CAPT application.
## 2 Methodology
## 2.1 Problem Definition
Considering an input time sequence of speech
signal 𝐮 uttered by an L2 learner and a reference
text prompt 𝐩 that contains 𝑁 -length canonical
phone sequence 𝐩 = {𝑝0,𝑝1,…,𝑝𝑁−1}, we adopt
a set of feature extractors along with an aligner to
extract an acoustic feature sequence 𝐗 =
{𝐱0,𝐱1,…,𝐱𝑁−1} that aligned with 𝐩 from 𝐮.
Our model aims to address APA and MDD tasks
simultaneously but with separate processing flows:
First, we define 𝐺 as a set of linguistic
granularities, and for each granularity 𝑔 ∈ 𝐺 the
model manages to predict a set of aspect scores
𝐬𝑔
= {𝑠0
𝑔
,𝑠1
𝑔
,…,𝑠𝑀𝑔−1
𝑔
}, where 𝑀𝑔 refers to the
number of aspect scores of target granularity 𝑔. In
this work, 𝐺 = {𝑔𝑝ℎ𝑛
,𝑔𝑤𝑟𝑑
,𝑔𝑢𝑡𝑡
}, where we have
granularities of 𝑔𝑝ℎ𝑛
(phone level), 𝑔𝑤𝑟𝑑
(word
level), and 𝑔𝑢𝑡𝑡
(utterance level) for the APA task.
Meanwhile, the model also requires to detect error
states 𝐞 = {𝑒0,𝑒1,…,𝑒𝑁−1} with respect to 𝐩
and in turn generate the correct diagnostic output
𝐲 = {𝑦0,𝑦1,…,𝑦𝑁−1} , where 𝑦𝑛 denotes the
uttered phone of the learner corresponds to 𝑝𝑛.
## 2.2 HMamba
In this subsection, we shed light on the details of
the proposed model, HMamba, which is devised as
a hierarchical structure built upon the paradigm of
selective SSM. A schematic illustration of the
complete architecture is depicted in Figure 2.
Specifically, HMamba synthesizes the APA and
MDD modules, each of which contains multiple
regressors and a classifier, respectively. These
modules collectively generate the corresponding
aspect score sequence 𝐬𝑔
for each linguistic
granularity 𝑔, as well as the phonetic error states 𝐞
and diagnosis 𝐲. Furthermore, each classifier and
regressor is implemented with a simple feedforward network (FFN) and jointly optimized
through the training.
Figure 2: An overall architectural overview of HMamba, which consists of a bottom-up hierarchical modeling
structure with several Mamba blocks across three levels (viz. phone, word, and utterance levels) that can perform
multi-granular APA and MDD in parallel.
Acoustic Feature Extraction: In order to portray
the non-native speaker’s pronunciation quality,
previous studies on either APA or MDD generally
adopt a pre-trained acoustic model to extract
goodness of pronunciation (GOP)-based features
(Witt and Young, 2000; Hu et al., 2015; Shi et al.,
2020). However, these features merely offer the
segmental-level information that may not capture
prosodic errors of an L2 learner. Given this
limitation, we first utilize a pre-trained acoustic
model as an aligner to identify phone boundaries
(including silence), facilitating the extraction of
other prosodic features such as the phone duration
andstatisticsofrootmeansquaredenergy(Donget
al., 2024). To mitigate the low-resourced data
problem(Chao et al., 2022), we also consider other
self-supervised learning (SSL) features including
wav2vec 2.01
(Baevski et al., 2020), HuBERT2
(Hsuetal.,2021),andWavLM3
(Chenetal.,2022).
All these features are concatenated and
subsequently projected through a linear layer to
form a sequence of acoustic features 𝐗 . The
transformation of each time step 𝑡 is given by
𝐚𝑡 = [𝐚𝑡
𝑔𝑜𝑝
;𝐚𝑡
𝑑𝑢𝑟
;𝐚𝑡
𝑒𝑛𝑔
;𝐚𝑡
𝑤2𝑣
;𝐚𝑡
ℎ𝑏𝑡
;𝐚𝑡
𝑤𝑙𝑚
] (1)
𝐱𝑡 = 𝐖𝐚𝑡 + 𝐛 (2)
where 𝐖 and 𝐛 are trainable parameters. Notably,
a dropout rate of 10% is applied to all SSLfeatures
prior to the concatenation due to the discrepancy in
dimensionality between these and other features.
Phonological Feature Extraction: In addition to
acoustic cues, a common practice in CAPT is to
inject the phonological information by introducing
the reference text prompt features such as
canonical phoneme embeddings (Gong et al.,
2022), context-aware sup-phoneme embeddings
(Chao et al., 2023), and vowel/consonant
embeddings (Fu et al., 2021). In contrast to
previous studies (Gong et al., 2022; Chao et al.,
2022; Do et al., 2023a), we extract the canonical
phoneme embeddings 𝐄𝑝ℎ𝑛
from 𝐩 using a phone
embedding layer that includes the silence (SIL)
information which has been shown to be crucial
when evaluating ones’ spoken proficiency.
Additionally, an absolute positional embedding
𝐄𝑎𝑏𝑠
and a relative position embedding 𝐄𝑟𝑒𝑙
are
1
https://huggingface.co/facebook/wav2vec
2-large-xlsr-53
2
https://huggingface.co/facebook/hubertlarge-ll60k
extracted. Distinct from 𝐄𝑎𝑏𝑠
, 𝐄𝑟𝑒𝑙
denotes
relative positions of phones in a word using tokens
such as begin [B], internal [I], end [E], and
single-phone word [S] tokens. For special cases
of silence positions, we explicitly categorize them
as either long silence [LS] or short silence [SS].
Following the guideline suggested by ETS
(Evanini et al., 2015), positions with a silence
duration exceeding 0.495 seconds are assigned to
[LS]; otherwise, they are assigned to [SS].
Finally,alltheseembeddingfeaturesarepoint-wise
added to 𝐗 to obtain phone-level input features for
subsequent modeling:
𝐇𝑔0
𝑝ℎ𝑛
= 𝐗 + 𝐄𝑝ℎ𝑛
+ 𝐄𝑎𝑏𝑠
+ 𝐄𝑟𝑒𝑙
(3)
The details of the complete feature ablations (both
acoustic and phonological features) are shown in
Appendix B.
Mamba Blocks: To foster highly efficient multitask learning, we introduce selective SSMs instead
of attention-based models such as the Transformer
(Vaswani et al., 2017). Specifically, we adopt
Mamba (Gu and Dao, 2023) as our backbone
model structure in this work. Different from
previous SSM instantiations, Mamba features an
input-dependent selection mechanism and a
hardware-aware algorithm, allowing for efficient
input information filtering by dynamically
adjusting the SSM parameters based on the input
data. This also facilitate faster recurrent
computation of the model using scan. Nevertheless,
the vanilla Mamba conducts causal computations
in a unidirectional manner, which prevents it from
capturing global information as effectively as the
multi-head self-attention (MHSA) module
involved in Transformer. To address this problem,
we explore a bidirectional variant of Mamba as the
basic modeling block. In this approach, we replace
the MHSA module in the Transformer encoder
with a bidirectional Mamba layer, as depicted in
Figure 2. Specifically, for input 𝐇𝑔𝑖 to the Mamba
block at granularity level 𝑔, the output 𝐇𝑔𝑖+1 of the
block is:
𝐇′𝑔𝑖 = BiMamba(LayerNorm(𝐇𝑔𝑖 )) + 𝐇𝑔𝑖 (4)
𝐇𝑔𝑖+1 = FFN(LayerNorm(𝐇′𝑔𝑖 )) + 𝐇′𝑔𝑖 (5)
3
https://huggingface.co/microsoft/wavlmlarge
where BiMamba denotes the bidirectional Mamba
layer and FFN refers to the feed-forward module,
respectively. Notably, there are several studies
investigating the bidirectional processing of
Mamba (Liang et al., 2024; Zhang et al., 2024;
Jiang et al., 2024). In this work, we use a similar
structure as Jiang et al. (2024) to implement the
bidirectional Mamba layer. For input 𝐍𝑔𝑖 from the
output of layer normalization of 𝐇𝑔𝑖 to a
bidirectional Mamba layer, the corresponding
output 𝐌𝑔𝑖 is computed as follows:
𝐙𝑔𝑖 = Linear(𝐍𝑔𝑖) (6)
𝐒𝑔𝑖→
= Linear(𝐍𝑔𝑖), 𝐒𝑔𝑖←
= Flip(𝐒𝑔𝑖→
) (7)
!
𝐂𝑔𝑖
→
= Conv1D→
(𝐒𝑔𝑖
→
)
𝐂𝑔𝑖
←
= Conv1D←
(𝐒𝑔𝑖
←
)
(8)
{
𝐎𝑔𝑖→
= 𝜎(𝐙𝑔𝑖 ) ⨂ SSM→
(𝐂𝑔𝑖→
)
𝐎𝑔𝑖←
= 𝜎(𝐙𝑔𝑖 ) ⨂ SSM←
(𝐂𝑔𝑖←
)
(9)
𝐌𝑔𝑖 = Linear(
1
2
𝐎𝑔𝑖→
+
1
2
Flip(𝐎𝑔𝑖←
)) (10)
where 𝐒𝑔𝑖→
and 𝐒𝑔𝑖←
denote the forward and
backward sequence features, respectively.
Specifically, 𝐒𝑔𝑖←
is derived from 𝐒𝑔𝑖→
by a
flipping operation Flip(∙). Conv1D(∙), 𝜎(∙), and
SSM(∙) represents the 1-D convolution, activation
function,andselectiveSSMalgorithmdescribedin
Mamba (Gu and Dao, 2023), respectively.
Hierarchical Mamba: Since the speech signals
are typically distinguished by the complex
hierarchical composition, prior studies (Do et al.,
2023a; Chao et al., 2023) have suggested that
hierarchical modeling structures is more amenable
than parallel modeling structures (Gong et al.,
2022). To capture the linguistic hierarchy while
retaining the cross-aspect relations within the same
linguistic unit, we design and instantiate our model
with a hierarchical structure and introduce Mamba
blocks to model the dependencies at each
granularity level. More concretely, our approach
generates finer granularity scores at the lower
layers and coarser granularity scores at the higher
layers, as exhibited in Figure 2. In phone-level
modeling, we first use 𝐇𝑔0
𝑝ℎ𝑛
as the input into 𝐿𝑝-
layer Mamba blocks to obtain the phone-level
contextualized representations 𝐇
𝑔𝐿𝑝
𝑝ℎ𝑛
:
𝐇
𝑔𝐿𝑝
𝑝ℎ𝑛
= MambaBlock𝑝ℎ𝑛(𝐇𝑔0
𝑝ℎ𝑛
) (11)
Subsequently, 𝐇
𝑔𝐿𝑝
𝑝ℎ𝑛
are then propagated forward
into the APA module and the MDD module for
solving a regression and a sequence classification
problem, respectively. The APA module contains
one regressor that aims to predict the phone-level
aspect score 𝑠0
𝑔𝑝ℎ𝑛
(accuracy). On the other hand,
the MDD module comprises a classifier and a
softmax function that cooperatively learn a
distribution 𝑦̂𝑡 over the phoneme classes 𝐶 for
each time step 𝑡. The diagnosis 𝑦𝑡 can then be
identifiedbyapplyingtheargmaxfunctionto 𝑦̂𝑡.In
this work, we streamline the MDD task by treating
it as a process of free phone recognition (Li et al.,
2015). As such, we can directly detect the
corresponding error state 𝑒𝑡 by comparing 𝑦𝑡 with
𝑝𝑡 , eliminating the need for a separate detection
module. Meanwhile, the resulting 𝐇
𝑔𝐿𝑝
𝑝ℎ𝑛
is served
as 𝐇𝑔0
𝑤𝑟𝑑
for subsequent modeling.
In word-level modeling, 𝐿𝑤 -layer Mamba
blocks are first adopted and followed by a 1-D
convolutionlayertocapturethelocaldependencies
(Lee, 2016). The reason for utilizing the
convolution layer is that the convolution operation
canaccommodatedifferentrealizationsofthesame
underlying phone from various L2 speakers,
thereby mitigating the temporal variability. The
word-level representations 𝐇𝑔𝐿𝑤
𝑤𝑟𝑑
can be derived
as follows:
𝐇′𝑔𝐿𝑤
𝑤𝑟𝑑
= MambaBlock𝑤𝑟𝑑(𝐇𝑔0
𝑤𝑟𝑑
) (12)
𝐇𝑔𝐿𝑤
𝑤𝑟𝑑
= Conv1D𝑤𝑟𝑑(𝐇′𝑔𝐿𝑤
𝑤𝑟𝑑
) (13)
To obtain word-level aspect scores, we put
𝐇𝑔𝐿𝑤
𝑤𝑟𝑑
into the word-level APA module which
contains three regressors to predict the word-level
aspect scores 𝑠0
𝑔𝑤𝑟𝑑
,𝑠1
𝑔𝑤𝑟𝑑
,𝑠2
𝑔𝑤𝑟𝑑
(accuracy, stress,
and total scores), respectively.To facilitate training
efficiency, we propagate the word score to each of
itsphonesduringthetrainingstage.Intheinference
phase, we ensure consistency by averaging the
outputs corresponding to each word. In addition,
𝐇𝑔𝐿𝑤
𝑤𝑟𝑑
is viewed as 𝐇𝑔0
𝑢𝑡𝑡
for further modeling.
As for the utterance-level assessments, instead
of prepending the [CLS] tokens to learn the
utterance-level representation (Gong et al., 2022),
we explore pooling-based approaches to aggregate
the hidden information. To this end, we utilize an
attentionpoolinglayersimilartoPengetal.(2022).
Specifically, assuming that a 𝑑-dimensional input
sequence to the attention pooling layer is
𝐡0,𝐡1 …,𝐡𝑇−1 , the pooling output is 𝐡 =
∑ 𝛼𝑖𝐡𝑖
𝑇−1
𝑖=0
, where 𝛼𝑖 is calculated by
𝛼𝑖 =
exp (𝐰𝑇
𝐪𝑖/𝜏)
∑ exp (𝐰𝑇 𝐪𝑗/𝜏)
𝑇−1
𝑗=0
(14)
where 𝐰 is a learnable vector, 𝐪 is the
concatenated scores of [𝑠0
𝑔𝑝ℎ𝑛
,𝑠0
𝑔𝑤𝑟𝑑
,𝑠1
𝑔𝑤𝑟𝑑
,𝑠2
𝑔𝑤𝑟𝑑
],
and 𝜏 is a controllable temperature hyperparameter.
The whole process of utterance-level modeling can
then be formulated by
𝐇𝑔𝐿𝑢
𝑢𝑡𝑡
= MambaBlock𝑢𝑡𝑡(𝐇𝑔0
𝑢𝑡𝑡
) (15)
𝐡𝑔𝑢𝑡𝑡
= AttentionPooling𝑢𝑡𝑡(𝐇𝑔𝐿𝑢
𝑢𝑡𝑡
) (16)
After obtaining 𝐇𝑔𝐿𝑢
𝑢𝑡𝑡
from 𝐿𝑢 -layer Mamba
blocks, 𝐡𝑔𝑢𝑡𝑡
is derived through the attention
pooling layer to predict the utterance-level aspect
scores 𝑠0
𝑔𝑢𝑡𝑡
,𝑠1
𝑔𝑢𝑡𝑡
,𝑠2
𝑔𝑢𝑡𝑡
,𝑠3
𝑔𝑢𝑡𝑡
,𝑠4
𝑔𝑢𝑡𝑡
(accuracy,
completeness, fluency, prosody, and total scores)
via an utterance-levelAPAmodule which contains
five regressors corresponding to each score.
## 2.3 Optimization
Automatic Pronunciation Assessment Loss: In
the proposed model, each APA module is
optimized using Mean Square Error (MSE). The
loss for multi-aspect multi-granular assessment,
ℒ𝐴𝑃𝐴, is calculated by assigning weights to each
granularity level 𝑔:
ℒ𝐴𝑃𝐴 = ∑𝜔𝑔
𝑔∈𝐺
∙
1
𝑁𝑔
∑ ∙ ℒ𝑔𝑘
𝑁𝑔−1
𝑘=0
(17)
where 𝜔𝑔 and 𝑁𝑔 are the tunable hyperparameter
and number of aspect scores at granularity level 𝑔,
respectively. ℒ𝑔𝑘
refers to the MSE loss computed
for 𝑘-th aspect score at granularity level 𝑔.
MispronunciationDetectionandDiagnosisLoss:
To be in line with previous MDD studies, our
model incorporates canonical phoneme
embeddings to enhance text prompt-awareness.
Despite some performance improvements, the
mismatchbetweentheL2learner’srealizedphones
and canonical phones can still cause some
deteriorating effects. This discrepancy can
introduce inaccurate predictions that may
potentially affect the overall quality of phonetic
analysis. To mitigate this negative impact, we
devise a new loss function tailored for the MDD
task, as illustrated inFigure 3. Specifically, we first
decouple the original cross-entropy loss into two
separate losses, one for mispronunciations and the
other for correct pronunciations:
ℒ𝑋𝑒𝑛𝑡
𝑚𝑖𝑠
= − ∑ log(𝑦̂𝑡[𝑦𝑡])
𝑡∈ℳ
(18)
ℒ𝑋𝑒𝑛𝑡
ℎ𝑖𝑡
= − ∑ log(𝑦̂𝑡[𝑦𝑡])
𝑡∈ℋ
(19)
where ℳ and ℋ are mispronunciation and
correct pronunciation positions, respectively, and
𝑦̂𝑡[𝑦𝑡] is the predicted probability of the true label
𝑦𝑡 at time step 𝑡. After obtaining two decoupled
losses, the proposed decoupled cross-entropy loss
(deXent) is obtained by the following formulation:
ℒ𝑀𝐷𝐷 = ℒ𝑋𝑒𝑛𝑡
ℎ𝑖𝑡
+ (
𝜇ℎ
𝜇𝑚
)𝛼
ℒ𝑋𝑒𝑛𝑡
𝑚𝑖𝑠 (20)
where 𝜇𝑚
and 𝜇ℎ
denote the frequency of the
mispronunciations and correct pronunciations in
the training set, respectively, and 𝛼 controls the
weight magnitude. After that, we use ℒ𝑀𝐷𝐷 to
optimize the MDD module, and the overall loss
thus can be expressed by
ℒ = ℒ𝐴𝑃𝐴 + 𝛽 ∙ ℒ𝑀𝐷𝐷 (21)
where 𝛽 is a tunable hyperparameter.
According to Equation 20, the proposed loss
functioncanbeviewedasoneoftheloss-balancing
methods, such as focal loss (Lin et al., 2017) and
class-balanced loss (Cui et al., 2019), to tackle the
imbalance issue in MDD. However, in most endto-end MDD methods, where the labels are phones
instead of mispronunciations (0 or 1s), directly
applying the existing loss-balancing methods on
phones is implicit and can be sub-optimal when we
aim to detect potential mispronunciations. Hence,
we believe the proposed deXent provides a better
alternative to end-to-end MDD.
Figure 3: Difference between (a) the original crossentropy loss and (b) the decoupled cross-entropy loss,
given the text prompt “crime.”
## 3 Experimental Setup
## 3.1 Dataset
We conducted experiments on speechocean762, a
widely-used open-source dataset curated for APA
and MDD research (Zhang et al., 2021). The
dataset consists of 5,000 English-speaking
recordingsfrom250MandarinL2learners,divided
evenly into training and test sets. For theAPAtask,
pronunciation proficiency scores were assessed at
various linguistic granularities and across different
pronunciation aspects. Each score is evaluated by
five experts using standardized rubrics. For the
MDD task, the dataset provides an extra
mispronunciation transcription annotated using a
set of 46 phones. This set comprises 39 phones
from the CMU dictionary4
, 6 L2-specific phones,
and a [unk] token for unknown phones. Notably,
there are no insertion errors in the utterances, and a
[DEL] token is introduced to mark deletion errors
of L2 learners. Therefore, the realized phones can
be aligned with canonical phones in this dataset.
## 3.2 Training and Evaluation Details
Training:We optimized the model withAdam and
a tri-phase rate scheduler (Baevski et al., 2020),
where the learning rate was gradually increased
during the first 40% of steps, held constant for the
following 40%, and then linearly decayed for the
remaining steps. The initial learning rate was set to
2e-3 except for the utterance-level APA module,
4
http://www.speech.cs.cmu.edu/cgibin/cmudict
which was set to 9e-5. Other implementation
details are presented inAppendixA.
Evaluation: The evaluation metrics employed
include the Pearson Correlation Coefficient (PCC)
and Mean Square Error (MSE) for the APA task.
On the other hand, we used precision, recall, F1-
score, and phone error rate (PER) to evaluate the
MDD performance, so as to be in accordance with
prior studies. To ensure the validity of our
experimental results, we conducted 5 independent
trials for each experiment, running 20 epochs with
different seeds. The metrics for each task are
reported as the average of these trials.
## 3.3 Compared Baselines
For APA, we compare our proposed approach,
HMamba, with various cutting-edge baselines
which can be categorized into two families: singleaspect pronunciation assessment models or multigranular multi-aspect pronunciation assessment
models. The first group includes the Deep Feature
(Lin et al., 2021b), HuBERT Large (Kim et al.,
2022), and Joint-CAPT-L1 (Ryu et al., 2023). The
second group encompasses LSTM, GOPT (Gong
et al., 2022), 3M (Chao et al., 2022), HiPAMA(Do
et al., 2023a), and 3MH (Chao et al., 2023).As for
MDD,wecompareHMambawithJoint-CAPT-L1,
as to our knowledge it is the only attempt that
jointly addresses theAPAand MDD tasks with the
speechocean762 dataset.
Model Year
Phone Score Word Score (PCC) Utterance Score (PCC)
MSE↓ PCC↑ Accuracy↑ Stress↑ Total↑ Accuracy↑ Completeness↑ Fluency↑ Prosody↑ Total↑
Deep Feature 2021 - - - - - - - - - 0.720
HuBERT Large 2022 - - - - - - - 0.780 0.770 -
Joint-CAPT-L1 2023 - - - - - 0.719 - 0.775 0.773 0.743
LSTM 2022 0.089 0.591 0.514 0.294 0.531 0.720 0.076 0.745 0.747 0.741
GOPT 2022 0.085 0.612 0.533 0.291 0.549 0.714 0.155 0.753 0.760 0.742
3M 2022 0.078 0.656 0.598 0.289 0.617 0.760 0.325 0.828 0.827 0.796
HiPAMA 2023 0.084 0.616 0.575 0.320 0.591 0.730 0.276 0.749 0.751 0.754
3MH 2023 0.071 0.693 0.682 0.361 0.694 0.782 0.374 0.843 0.836 0.811
HMamba 2024 0.062 0.739 0.708 0.366 0.718 0.807 0.278 0.848 0.843 0.829
Table 1: APA performance evaluations of our model and all strong baselines on the speechocean762 test set.
## 4 Experimental Results and Discussion
## 4.1 APA Performance
Overall Results: In Table 1, we compare theAPA
performance of HMamba with other competitive
baselines, leading to several key observations.
Firstly, it is notable that our approach, HMamba,
consistently outperforms all other methods on
nearly all assessment tasks, particularly in terms of
accuracy scores at phone, word, and utterance
levels. This improvement stems from the joint
modeling paradigm of APA and MDD,
highlighting that pronunciation assessments can
also benefit from phonetic error discovery,
consistent with prior research findings (Ryu et al.,
2023). In addition, by adopting SSL features,
HMambaaswellasotherapproacheslikeHuBERT
Large, 3M, and 3MH, achieves significant
improvements over the other APA methods in
terms of utterance-level assessments. In
comparison to other hierarchical models such as
HiPAMA and 3MH, HMamba leverages an SSM
structure instead of the Transformer structure,
demonstratingsuperiorperformanceonavarietyof
assessment tasks (further analysis between Mamba
and Transformer are shown in Appendix C). In
assessing utterance completeness, while HMamba
falls behind 3M and 3MH, it is on par with
HiPAMA. According to Zhang et al. (2021), the
completeness refers to the percentage of the words
that are actually pronounced. This may imply that
our approach has a weaker ability to detect
incompletely articulated words. However, this
limitation could stem from the fact that HMamba
focuses more on phoneme accuracy and
mispronunciation detection, rather than purely
evaluating word-level completeness.
Effects of Hierarchical Structure: To inspect the
hierarchical structure that influences on the APA
performance of the proposed approach, we
conductedanexperimenttoreplacethehierarchical
structure with two other variants, resulting in two
different models: LMamba and PMamba,
respectively. LMamba has a similar structure to
HMamba but outputs all assessment scores in the
last layers regardless of their granular differences.
On the other hand, PMamba adopts the parallel
structuresuggestedbypreviousstudies(Gongetal.,
2022;Chaoetal.,2022)thatuseprepended[CLS]
tokens to predict utterance-level scores.According
to the results shown in Table 2, HMamba
outperforms PMamba and LMamba across all
assessment aspects, highlighting the advantages of
its hierarchical structure for the APA task. This
finding aligns with previous research (Chao et al.,
2023). In addition, the significant performance
gaps between HMamba and LMamba also suggest
that phone-level and word-level scores should be
predicted in lower layers.
Model
Phone Score Word Score (PCC) Utterance Score (PCC)
MSE↓ PCC↑ Accuracy↑ Stress↑ Total↑ Accuracy↑ Completeness↑ Fluency↑ Prosody↑ Total↑
LMamba 0.071 0.694 0.678 0.299 0.689 0.790 0.234 0.844 0.838 0.816
PMamba 0.068 0.707 0.689 0.320 0.700 0.784 0.142 0.843 0.832 0.817
HMamba 0.062 0.739 0.708 0.366 0.718 0.807 0.278 0.848 0.843 0.829
Table 2: Performance comparison between different modeling structures.
Model
Mispronunciations
PER ↓
Precision ↑ Recall ↑ F1 ↑
Joint-CAPT-L1 26.70% 91.40% 41.50% 9.93%
HMamba 64.35% 63.41% 63.85% 2.72%
Table 3: MDD performance evaluations of our model,
compared with a representative multi-task approach
(Ryu et al., 2023) on the speechocean762 test set.
Loss 𝛼
Mispronunciations
PER ↓
Precision ↑ Recall ↑ F1 ↑
Xent - 77.07% 38.60% 51.40% 2.53%
deXent
0.3 70.06% 54.10% 61.04% 2.61%
0.5 67.12% 58.71% 62.62% 2.70%
0.7 64.35% 63.41% 63.85% 2.72%
0.9 57.74% 71.12% 63.73% 3.14%
Table 4: Comparison of MDD performance between
the original cross-entropy loss (Xent) and proposed
decoupled cross-entropy loss (deXent).
## 4.2 MDD Performance
Overall Results: We evaluate the MDD
performance of HMamba by comparing it with
another celebrated multi-task learning approach,
Joint-CAPT-L1. As shown in Table 3, despite of
lower recall rate, HMamba achieves a significant
improvement in terms of F1-score over JointCAPT-L1, with an increase of 22.35%.
Additionally,thereisamarkedreductioninPERby
7.21%. These substantial enhancements
demonstrate that HMamba not only delivers
accurate pronunciation assessments but also
produces more robust and reliable
mispronunciation detection and diagnosis results.
EffectsofDecoupledCross-entropyLoss:Onthe
grounds of the distinct improvements in the MDD
performance, we further analyze the underlying
effects of our proposed decoupled cross-entropy
loss (deXent). As illustrated in Table 4, training a
text prompt-aware MDD model using the original
cross-entropy loss (Xent) often yields high
precision but low recall. This is because the model
primarily relies on input canonical phones, leading
it to predict prior phones and overlook the actual
mispronunciations of a learner. Such a model may
not be suitable for educational settings where
accurately detecting potential mispronunciations is
critical. To remedy this, the proposed deXent
method sufficiently provides a feasible solution.
By adjusting the weighting factor 𝛼, we can better
strike the balance between precision and recall,
thus optimizing the F1-score. This flexibility is
particularly vital in CAPT, where both overdetection and under-detection of
mispronunciations can severely disrupt the
learning process—a challenge often neglected by
most existing end-to-end methods. While adopting
deXent may result in a minor increase in PER, this
slight performance tradeoff is justifiable for the
significant gains in overall MDD effectiveness.
Limitations of Decoupled Cross-entropy Loss:
In Table 3, the MDD performance of HMamba is
reported based on maximizing the F1 score using
the deXent, as we believe both the precision and
recall metrics are critical for MDD. However, a
potential limitation of using deXent as the loss
functionisthat,whileitmayhelpbalanceprecision
and recall for MDD, it may not simultaneously
improve both metrics. This limitation likely stems
from the close relationship between the
mispronunciation distribution and the lossbalancing mechanism of deXent.
## 5 Conclusion
In this paper, we have presented a novel
hierarchical selective state space model (dubbed
HMamba) for multifaceted CAPT application.
Extensive experimental results substantiate the
viability and efficacy of the proposed method
compared to several top-of-the-line approaches in
terms of both the APA and MDD performance. In
future work, we envisage mitigating the issue of
data imbalance from an optimization perspective.
In addition, another key area for future research
involves tackling the assessment of open-response
scenarios in CAPT.
Limitations
Lack ofAccent Diversity. The dataset used in this
study comprises only Mandarin L2 learners, which
would probably limit the generalizability of the
proposed model. As a result, it might be
inapplicable when assessing L2 learners with
diverse accents. This lack of accent diversity could
lead to biases and inaccuracies in pronunciation
assessment for learners from different linguistic
backgrounds.
Limited Interpretability. The proposed model is
designed to replicate expert annotations without
relying on manual assessment rubrics or external
knowledge databases, which would make it
challenging to provide clear and reasonable
explanations for the assessment results. This
insufficiency of interpretability might hinder its
acceptance and trustworthiness among educators
andlearnerswhorequiretransparentandjustifiable
assessments.
Limited Generalizability This research is
centered on the “reading-aloud” pronunciation
training scenario, where it is assumed that the L2
learneraccuratelypronouncesapredeterminedtext
prompt. This would narrow the applicability of our
models to other learning contexts, such as
spontaneous speech or open-ended conversations.
Ethics Statement
We acknowledge that all co-authors of this work
comply with theACLCode of Ethics and adhere to
the code of conduct. Our experimental corpus,
speechocean762, is widely used and publicly
available, and we believe there are no potential
risks associated with this work.
Acknowledgments
ThisworkwassupportedbytheLanguageTraining
and Testing Center (LTTC), Taiwan. Any findings
and implications in the paper do not necessarily
reflect those of the sponsor.
## References
Juan Pablo Arias, Nestor Becerra Yoma, and Hiram
Vivanco.2010.Automaticintonationassessmentfor
computer aided language learning. Speech
Communication, volume 52, pages 254–267.
StefanoBannò,BhanuBalusu,MarkGales,KateKnill,
and Konstantinos Kyriakopoulos. 2022a. Viewspecific assessment of L2 spoken English. In
Proceedings of the Annual Conference of the
International Speech Communication Association
(INTERSPEECH), pages 4471–4475.
Stefano Bannò and Marco Matassoni. 2022b.
Proficiency assessment of L2 spoken English using
wav2vec 2.0. In Proceedings of IEEE Spoken
LanguageTechnologyWorkshop(SLT),pages1088-
1095.
Alexei Baevski,Yuhao Zhou,Abdelrahman Mohamed,
and MichaelAuli. 2020. wav2vec 2.0:Aframework
for self-supervised learning of speech
representations. In Proceedings of the Conference
on Neural Information Processing Systems
(NeurIPS), pages 12449–12460.
FuAn Chao, Tien Hong Lo, Tzu I. Wu,YaoTing Sung,
Berlin Chen. 2022. 3M: An effective multi-view,
multigranularity, and multi-aspect modeling
approach to English pronunciation assessment. In
Proceedings of the Asia-Pacific Signal and
Information Processing Association Annual Summit
and Conference (APSIPA ASC), pages 575–582.
Fu-An Chao,Tien-Hong Lo,Tzu-IWu,Yao-Ting Sung,
Berlin Chen. 2023. A hierarchical context-aware
modeling approach for multi-aspect and multigranular pronunciation assessment. In Proceedings
of the Annual Conference of the International
Speech Communication Association
(INTERSPEECH), pages 974–978.
Nancy F. Chen, and Haizhou Li. 2016. Computerassisted pronunciation training: From pronunciation
scoring towards spoken language learning. In
Proceedings of the Asia-Pacific Signal and
Information Processing Association Annual Summit
and Conference (APSIPA ASC), pages 1–7.
Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu
Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki
Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu,
LongZhou, ShuoRen,Yanmin Qian,YaoQian, Jian
Wu, Michael Zeng, Xiangzhan Yu, and Furu Wei.
## 2022 Wavlm: Large-scale self-supervised pretraining for full stack speech processing. IEEE
Journal of Selected Topics in Signal Processing,
volume 16, number 6, pages 1505-1518.
Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge
Belongie. 2019. Class-balanced loss based on
effective number of samples. In Proceedings of the
IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 9268-9277.
Heejin Do,Yunsu Kim, and Gary Geunbae Lee. 2023a.
Hierarchical pronunciation assessment with multiaspect attention. In Proceedings of the IEEE
International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pages 1–5.
Heejin Do,Yunsu Kim, and Gary Geunbae Lee. 2023b.
Score-balanced loss for multi-aspect pronunciation
assessment. In Proceedings of the Annual
Conference of the International Speech
Communication Association (INTERSPEECH),
pages 4998–5002.
Hongwei Ding, Xinping Xu. 2016. L2 English rhythm
in read speech by Chinese students. In Proceedings
of the Annual Conference of the International
Speech Communication Association
(INTERSPEECH), pages 2696-2700.
Bin Dong, Qingwei Zhao, Jianping Zhang, and
Yonghong Yan. 2004. Automatic assessment of
pronunciation quality. In Proceedings of IEEE
International Symposium on Chinese Spoken
Language Processing (ISCSLP), pages 137-140.
Maxine Eskenazi. 2009. An overview of spoken
language technology for education. Speech
Communication, volume 51, pages 832–844.
Keelan Evanini, Michael Heilman, Xinhao Wang, and
Daniel Blanchard. 2015. Automated scoring for the
TOEFL Junior® comprehensive writing and
speaking test. ETS Research Report Series
2015(1):1–11.
Kaiqi Fu, Jones Lin, Dengfeng Ke, Yanlu Xie, Jinsong
Zhang, and Binghuai Lin. 2021. A full textdependent end to end mispronunciation detection
and diagnosis with easy data augmentation
techniques. arXiv preprint arXiv:2104.08428.
Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang,
and James Glass. 2022. Transformer-based multiaspect multigranularity non-native English speaker
pronunciation assessment. In Proceedings of the
IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pages
7262–7266.
Albert Gu and Tri Dao. 2023. Mamba: Linear-time
sequence modeling with selective state spaces.
arXiv preprint arXiv:2312.00752.
Wei-NingHsu, BenjaminBolte,Yao-HungHubertTsai,
Kushal Lakhotia, Ruslan Salakhutdinov, and
Abdelrahman Mohamed. 2021. Hubert: Selfsupervised speech representation learning by
masked prediction of hidden units. IEEE/ACM
Transactions on Audio, Speech, and Language
Processing, volume 29, pages 3451–3460.
Alissa M. Harrison, Wai-Kit Lo, Xiao-Jun Qian, and
Helen Meng. 2009. Implementation of an extended
recognition network for mispronunciation detection
and diagnosis in computer-assisted pronunciation
training. In Proceedings of the Workshop on Speech
and Language Technology in Education (SLaTE),
pages 45-48.
Wenping Hu, Yao Qian, Frank K. Soong, and Yong
Wang. 2015. Improved mispronunciation detection
with deep neural network trained acoustic models
and transfer learning based logistic regression
classifiers. Speech Communication, volume 67,
pages 154–166.
Xilin Jiang, Cong Han, and Nima Mesgarani. Dualpath mamba: Short and long-term bidirectional
selective structured state space models for speech
separation. 2024. arXiv preprint arXiv:2403.18257.
Yassine Kheir, Ahmed Ali, and Shammur Chowdhury.
## 2023 Automatic pronunciation assessment - a
review. In Findings of the Association for
Computational Linguistics: EMNLP, pages 8304–
8324.
EesungKim,Jae-JinJeon,HyejiSeo,HoonKim.2022.
Automatic pronunciation assessment using selfsupervised speech representation learning. In
Proceedings of the Annual Conference of the
International Speech Communication Association
(INTERSPEECH), pages 1411–1415.
Ann Lee. 2016. Language-independent methods for
computer-assisted pronunciation training, Ph.D.
thesis, Massachusetts Institute of Technology.
Aobo Liang, Xingguo Jiang, Yan Sun, Xiaohou Shi,
and Ke Li. 2024. Bi-Mamba4TS: Bidirectional
mamba for time series forecasting. arXiv preprint
arXiv:2404.15772.
Binghuai Lin, Liyuan Wang, Hongwei Ding, Xiaoli
Feng. 2021a. Improving L2 English rhythm
evaluation with automatic sentence stress detection.
In Proceedings of IEEE Spoken Language
Technology Workshop (SLT), pages 713-719.
Binghuai Lin and Liyuan Wang. 2021b. Deep feature
transfer learning for automatic pronunciation
assessment. In Proceedings of the Annual
Conference of the International Speech
Communication Association (INTERSPEECH),
pages 4438–4442.
Tsung-YiLin,PriyaGoyal,RossGirshick,KaimingHe,
Piotr Dollár. 2017. Focal Loss for Dense Object
Detection. In Proceedings of the IEEE International
Conference on Computer Vision (ICCV), pages
2980-2988.
Kun Li, Xiaojun Qian, Shiying Kang, Pengfei Liu, and
Helen Meng. 2015. Integrating acoustic and statetransition models for free phone recognition in L2
English speech using multi-distribution deep neural
networks. In Proceedings of the Workshop on
Speech and Language Technology in Education
(SLaTE), pages. 119-124.
Arya D. McCarthy, Kevin P. Yancey, Geoffrey T.
LaFlair, Jesse Egbert, Manqian Liao, and Burr
Settles. 2021. Jump-starting item parameters for
adaptive language tests. In Proceedings of the
Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 883–899.
Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji
Watanabe. 2022. Branchformer: Parallel mlpattention architectures to capture local and global
context for speech recognition and understanding.
In Proceedings of the International Conference on
Machine Learning (PMLR), pages 17627–17643.
Hyungshin Ryu, Sunhee Kim, and Minhwa Chung.
## 2023 A joint model for pronunciation assessment
and mispronunciation detection and diagnosis with
multi-task learning. In Proceedings of the Annual
Conference of the International Speech
Communication Association (INTERSPEECH),
pages 959-963.
Jiatong Shi, Nan Huo, and Qin Jin. 2020. Contextaware goodness of pronunciation for computerassisted pronunciation training. In Proceedings of
the Annual Conference of the International Speech
Communication Association (INTERSPEECH),
pages 3057-3061.
Helmer Strik, Khiet Truong, Febe De Wet, and Catia
Cucchiarini. 2009. Comparing different approaches
for automatic pronunciation error detection. Speech
Communication, volume 51, number 10, pages 845-
852.
Khiet Truong, Ambra Neri, Catia Cucchiarini, and
Helmer Strik. 2004. Automatic pronunciation error
detection: an acoustic-phonetic approach.
InSTIL/ICALL Symposium.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. In Proceedings of the Conference on
Neural Information Processing Systems (NeurIPS),
pages 5998–6008.
Anjana S. Vakil and Jürgen Trouvain. 2015.
“Automatic classification of lexical stress errors for
German CAPT,” in Proceedings of the Workshop on
Speech and Language Technology in Education
(SLaTE), pages 47– 52.
Silke M. Witt and Steve J. Young. 2000. Phone-level
pronunciation scoring and assessment for
interactive language learning. Speech
Communication, volume 30, pages 95–108.
Klaus Zechner, Derrick Higgins, Xiaoming Xi, and
David M. Williamson. 2009. Automatic scoring of
non-native spontaneous speech in tests of spoken
English. Speech Communication, volume 51,
number 10, pages 883-895.
Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong
Wang, Wenyu Liu, and Xinggang Wang. 2024.
Vision mamba: Efficient visual representation
learning with bidirectional state space model. arXiv
preprint arXiv:2401.09417.
Xiangyu Zhang, Qiquan Zhang, Hexin Liu, Tianyi
Xiao, Xinyuan Qian, Beena Ahmed, Eliathamby
Ambikairajah, Haizhou Li, and Julien Epps. 2024.
Mamba in Speech: Towards an alternative to selfattention. arXiv preprint arXiv:2405.12609.
Junbo Zhang, Zhiwen Zhang, Yongqing Wang,
Zhiyong Yan, Qiong Song, Yukai Huang, Ke Li,
Daniel Povey, and Yujun Wang. 2021.
Speechocean762: An open-source non-native
English speech corpus for pronunciation assessment.
In Proceedings of the Annual Conference of the
International Speech Communication Association
(INTERSPEECH), pages 3710 –3714.
Appendix
A Implementation Details
Feature Extraction: We adopt an open-source
acoustic model5
to extract GOP features, which
also serves as an aligner for force alignment.
Subsequently, the phone-level duration, energy
statistics, and SSL features (average over time
frames) are computed according to the alignment.
The resulting acoustic features 𝑋 and all
embeddings are 128 dimensions. For all Mamba
blocks, we set the number of hidden units to 128
and use a kernel size of 4 for the 1-D convolution.
The SSM modules follow the original
configuration used in Mamba. 𝐿𝑝, 𝐿𝑤, 𝐿𝑢 are set
to 3, 1, 1, respectively. In addition, the word-level
1-D convolution has 256 kernels, each with a size
of 3.
Hyperparameters setting: 𝜏 in attention pooling
layer is set to 1.0. The combining weights 𝜔𝑔 for
APA loss are uniformly set to 1.0 for each
granularity level 𝑔. Parameters 𝛼 and 𝛽 are tuned
to be 0.7 and 0.003, respectively.
5
https://kaldi-asr.org/models/m13
B Feature Ablations
In Table 5, we conduct an ablation study on the
feature extraction to inspect the factors that
influence on APA. Specifically, we removed one
factor at a time to investigate the performance
variations.
Acoustic Features: According to the ablation
experiment, each acoustic feature used in this
work contributes to specific aspect assessments.
While the model may perform better in assessing
utterance completeness without phone duration
features, the other aspect assessments and MDD
performance decreases synchronously.
Furthermore, among all acoustic features, GOP is
the most crucial factor in relation to both of the
APAand MDD performance.
Phonological Features: As for the phonological
features, the canonical phoneme embeddings are
the most critical features overall, particularly in
MDD. Without canonical phoneme embeddings,
the performance dramatically degrades in F1
score and PER. Since the number of
mispronunciations is typically far less than the
APA MDD
Phone Word Utterance
F1↑ PER↓
MSE↓ Acc.↑ Acc.↑ Stress↑ Total↑ Acc.↑ Comp.↑ Fluency↑ Prosody↑ Total↑
HMamba 0.062 0.739 0.708 0.366 0.718 0.807 0.278 0.848 0.843 0.829 63.85% 2.72%
Acoustic Features
-wav2vec2 0.062 0.736 0.708 0.326 0.718 0.801 0.185 0.840 0.833 0.823 63.63% 2.79%
-HuBERT 0.063 0.735 0.706 0.344 0.715 0.804 0.216 0.843 0.838 0.825 63.49% 2.90%
-wavLM 0.063 0.731 0.705 0.355 0.715 0.806 0.247 0.844 0.838 0.827 63.39% 2.97%
-duration 0.063 0.734 0.705 0.341 0.715 0.804 0.299 0.844 0.838 0.826 63.70% 2.80%
-energy 0.063 0.735 0.706 0.358 0.716 0.802 0.257 0.840 0.834 0.823 63.28% 2.78%
-GOP 0.066 0.719 0.699 0.293 0.706 0.795 0.228 0.837 0.829 0.817 61.72% 2.79%
Phonological Features
-absolute-pos 0.063 0.735 0.706 0.332 0.715 0.802 0.261 0.843 0.838 0.825 63.48% 2.79%
-relative-pos 0.063 0.733 0.704 0.352 0.714 0.804 0.220 0.847 0.841 0.825 62.74% 2.81%
-canonical-phn 0.083 0.624 0.604 0.310 0.617 0.775 0.147 0.842 0.836 0.801 28.06% 14.55%
Table5:FeatureablationsofHMamba(MDDperformanceisreportedwithF1andPERasrepresentativemetrics).
correct pronunciations, canonical phoneme
embeddings can provide ample text-prompt
information to complement the acoustic cues. The
impactofcanonicalphonemeembeddingsonAPA
is consistent with Gong et al., 2022, and we take
it a step further in this work by demonstrating that
they are also pivotal for MDD.
C Mamba v.s. Transformer
To validate the effectiveness of Mamba over
Transformer in different facets, we replace each
Mamba block in HMamba with a vanilla
Transformer block (encoder only). In the
following, we perform a set of qualitative
analyses for comparisons of these two variant
structures.
Block Type
APA MDD
Phone Word Utterance
F1↑ PER↓
MSE↓ Acc.↑ Acc.↑ Stress↑ Total↑ Acc.↑ Comp.↑ Fluency↑ Prosody↑ Total↑
Transformer 0.071 0.692 0.689 0.294 0.700 0.797 0.165 0.844 0.839 0.819 60.14% 3.50%
Mamba 0.062 0.739 0.708 0.366 0.718 0.807 0.278 0.848 0.843 0.829 63.85% 2.72%
Table 6: Performance comparison between Mamba block and Transformer block.
Block Type Params(M)↓ MACs(G)↓
Transformer 1.469 3.806
Mamba 1.141 2.954
Table 7: Computational efficiency between Mamba block and Transformer block.
Figure 4: Comparison of the training curves for models equipped with Mamba and Transformer.
Performance comparison: In the first
experiment, we compare the APA and MDD
performance by utilizing Mamba or Transformer
as a basic block, respectively. According to the
results shown in Table 6, Mamba consistently
outperforms Transformer across all assessment
tasks and MDD, especially in phone- and wordlevel assessments. The key difference between
Mamba and Transformer is that Mamba leverages
1-D convolution to model the local context
dependency which has been shown crucial in
eitherAPA(Do et al., 2023a) or MDD (Lee, 2016)
task. These findings also align with other research
in the speech community, such as speech
separation (Jiang et al., 2024) and speech
enhancement (Zhang et al., 2024).
Computational efficiency: We further
investigate the computational efficiency of two
variants of architectures with the number of their
parameters and multiply-accumulate operations
(MACs). In Table 7, we observe that the model
equipped with Mamba has fewer parameters and
requires fewer MACs compared to the model with
Transformer. This reduced complexity suggests
that Mamba is not only more resource-efficient
but also potentially more scalable for practical
applications where both performance and
resource constraints are critical.
Training efficiency: To track the full training
process of the two architectures, we plot various
training curves in Figure 4, including the training
loss (log scale) and the changes in key metrics,
such as PER and the total scores of the phones,
words, and utterances. Mamba (red) consistently
exhibits lower training loss than Transformer
(blue) throughout the steps, suggesting that
Mamba enables faster and better convergence,
ultimately achieving a smaller loss overall. As
training progresses, Mamba rapidly outperforms
Transformer in terms of PER and PCC for both
phone and word total scores. Although this
advantageis less pronouncedinthePCCcurvefor
the utterance total score, Mamba still surpasses
Transformer to a moderate extent.
