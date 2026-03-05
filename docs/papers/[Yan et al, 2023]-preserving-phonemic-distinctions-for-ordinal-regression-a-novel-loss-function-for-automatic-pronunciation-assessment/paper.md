PRESERVING PHONEMIC DISTINCTIONS FOR ORDINAL REGRESSION: A NOVEL
LOSS FUNCTION FOR AUTOMATIC PRONUNCIATION ASSESSMENT
Bi-Cheng Yan1, Hsin-Wei Wang1, Yi-Cheng Wang1, Jiun-Ting Li1, Chi-Han Lin2, Berlin Chen1
National Taiwan Normal University, Taipei, Taiwan
E.SUN Financial Holding Co., Ltd., Taipei, Taiwan
{ bicheng, hsinweiwang, yichengwang, 60947036s, berlin}@ntnu.edu.tw; finalspaceman-19590@esunbank.com

ABSTRACT
Automatic pronunciation assessment (APA) manages to
quantify the pronunciation proficiency of a second language
(L2) learner in a language. Prevailing approaches to APA
normally leverage neural models trained with a regression
loss function, such as the mean-squared error (MSE) loss, for
proficiency level prediction. Despite most regression models
can effectively capture the ordinality of proficiency levels in
the feature space, they are confronted with a primary obstacle
that different phoneme categories with the same proficiency
level are inevitably forced to be close to each other, retaining
less phoneme-discriminative information. On account of this,
we devise a phonemic contrast ordinal (PCO) loss for training
regression-based APA models, which aims to preserve better
phonemic distinctions between phoneme categories
meanwhile considering ordinal relationships of the regression
target output. Specifically, we introduce a phoneme-distinct
regularizer into the MSE loss, which encourages feature
representations of different phoneme categories to be far
apart while simultaneously pulling closer the representations
belonging to the same phoneme category by means of
weighted distances. An extensive set of experiments carried
out on the speechocean762 benchmark dataset demonstrate
the feasibility and effectiveness of our model in relation to
some existing state-of-the-art models.
Index Terms— Automatic pronunciation assessment,
computer-assisted pronunciation training, deep regression
models, ordinal regression models
1. INTRODUCTION
Computer-assisted pronunciation training (CAPT) systems
have become increasingly popular and been used for a
multitude of use cases on language learning, with the purpose
to enable learners to practice their speaking skills, alleviate
the workloads of teachers [1], and others [2][3][4]. CAPT
research can hark back to the middle of the last century [5]
and has aroused increasing attention in recent years, showing
impressive performance by leveraging many advanced

Fig. 1. Visualization of phoneme representations for a
regression-based APA (GOPT) model trained with the meansquared error loss reveals that the regression model tends to
cluster features according to their proficiency scores.

machine learning technologies [6][7][8]. In common CAPT
systems, second language (L2) learners are initially presented
with a text prompt and instructed to read it aloud. By working
in conjunction with the input speech and the presented text
prompt, CAPT systems can access the learner’s speaking
proficiency and immediately provide instructive diagnostic
feedback [9][10][11]. Through persistent repetition and
practice, it is anticipated that L2 learners can gradually
improve their speaking skills.
According to the types of diagnostic feedback, the
research endeavors of CAPT fall into two broad categories:
one is phoneme-level mispronunciation detection and
diagnosis (MDD), and the other is automatic pronunciation
assessment (APA). The former aims to pinpoint phonemelevel erroneous pronunciations and provide L2 learners the
corresponding diagnostic feedback [12][13][14]. The latter,
in contrast, concentrates more on assessing and providing
cross-level pronunciation scores to reflect the learner’s
pronunciation quality on some specific aspects or traits of
their spoken language usage [15][16][17]. To this end, it
evaluates pronunciation proficiency at various linguistic

Word [yummy] Accuracy/ Stress/ Total Score

Phone
[AH1]
Score

Phone
[Y]
Score
Accuracy
Score

Fluency
Score

Completeness
Score

Prosodic
Score

Total
Score

Utterance

Utterance

Utterance

Utterance

Utterance

Accuracy
Head

Fluency
Head

Complete
Head

Prosodic
Head

Total
Head

Phn
Head

Word
Head

Phone
[M]
Score

Phn
Head

Word
Head

Feat !"

Feat !!

A Series of
Pronunciation Scores
'
$%&

Phone
[Y0]
Score

Phn
Head

Word
Head

Feat !#

Phn
Head

Word
Head

Feat !$

!

Phn-level Feats
H%
Transformer Encoder
"

Transformer Encoder
Pos[0]

Pos[1]

Pos[2]

Pos[3]

Pos[4]

Pos[5]

Pos[6]

Pos[7]

+

+

+

+

+

+

+

+

+

CLS
[utt-acc]

CLS
[utt-flu]

CLS

CLS
[utt-pros]

CLS
[utt-total]

Phn[Y]

Phn[AH1]

Phn[M]

Phn[Y0]

+

+

+

+

[utt-comp]

Regression Heads
&( "

Prepended Tokens

Pos[8]

GOP Projection Layer
!

!

!

X ! = {$" , $ # , … , $ $ }

GOP[Y]

GOP[AH1]

GOP[M]

GoP[IY0]

Acoustic Model

Input
Utterance U

Phone
Projection
Layer

(Audio Signals X)
Yummy
(Text Prompt T)

/Y AH1 M IY0/
Canonical Phones

Fig. 2. A schematic illustration of the GOPT model [15].

granularities (i.e., phoneme, word and utterance), with
diverse aspects (e.g., accuracy, fluency and completeness).
Contrary to MDD systems, APA systems possess the
capability to provide a broad spectrum of pronunciation
scores, making them conducive for L2 learners to improve
their speaking skills more comprehensively. Consequently,
APA systems have received much attention form academic
and commercial sectors, yet remains a challenging task
[18][19]. Prevailing approaches to the APA task usually
follow a supervised training paradigm, where various neural
models are employed to predict continuous scores that
mimics human experts’ evaluations on learners’ speaking
proficiency. Attributed to the continuous nature of the target
output, these models are typically optimized with a regression
loss function, namely mean-squared error (MSE). The MSE
loss effectively preserves the inherent ordinality of the target
output in the feature space by minimizing the average squared
difference between the model predictions and the human
evaluations on speaking proficiency. However, as illustrated
in Figure 1, the MSE loss inevitably suffers from a limitation
that different categories of phonemes that belong to the same
proficiency level are inadvertently forced to be close to one
another.
To address this issue, we propose a novel loss function,
dubbed phonemic contrast ordinal (PCO) loss, for enhancing
regression-based APA models by introducing a phonemedistinct regularizer into the MSE loss. This regularizer with
two decoupled mathematical terms, namely the phonemic
distinction and the ordinal tightness, manipulates the
distances between inter- and intra-phoneme categories in

both the feature and target spaces, respectively. The
phonemic distinction concentrates on increasing the distances
between feature centers of inter-phoneme categories, thereby
ensuring that different phoneme categories can stay far apart
from one another. The ordinal tightness preserves the ordinal
relationships of the target output by considering the
proficiency levels when pulling closer the intra-phoneme
distances for each phoneme category. The synergy of these
two terms provides the PCO loss with the capability to
overcome the shortage of the MSE loss commonly adopted
by existing APA models. To evaluate the effectiveness of the
proposed PCO loss, we pair it with an iconic regression-based
APA model, referred to as Goodness of Pronunciation
feature-based Transformer (GOPT). GOPT adopts a
transformer architecture as the backbone and simultaneously
predicts proficiency scores at multiple linguistic granularities
with various aspects [15].
2. PROPOSED METHOD
In this section, we begin by elaborating on the problem
formulation of APA. Next, we give a brief sketch of the
baseline APA model, GOPT. Finally, we introduce our
proposed PCO loss for use in APA.
2.1. Problem Definition
We tackle the automatic pronunciation assessment (APA)
task with the following processing flow. Given an input
utterance U, which consists of a sequence of audio signals X

uttered by an L2 learner, and a text prompt T that the learner
is expected to pronounce it correctly. Our APA model
manages to estimate a rich set of proficiency scores Y =
{𝐲!" |𝑔 ∈ G, 𝑎 ∈ A} that covers different linguistic
granularities G and aspects A. Here 𝐲!" stands for a vector of
pronunciation scores, and its length is associated with the
combination of linguistic granularity 𝑔 and aspect 𝑎 .
Specifically, for the input utterance U, the APA model trained
to estimate five aspect scores at the utterance-level (i.e.,
accuracy, fluency, completeness, prosody and total score),
qualify three aspect scores at the word-level (i.e., accuracy,
stress and total score), and access an accuracy score at the
phoneme-level. The aspect scores at utterance- and wordlevels range from 0-10, while those of the phoneme-level
aspects range from 0-2 in our experiments.
2.2. Revisiting the GOPT Model
GOPT, as shown in Figure 2, is a single, unified network
composed of a transformer encoder network 𝜑 and several
aspect-specific regression heads 𝑓# , where 𝑓# collectively is
a regression function with a parameter set Θ = {𝜃!" |𝑔 ∈
G, 𝑎 ∈ A}. For an input utterance U, the GOPT model first
aligns the audio signals X with the text prompt T to extract a
sequence of phoneme-level goodness of pronunciation (GOP)
features X $ = (𝐱%$ , 𝐱&$ , … , 𝐱'$ ), where each GOP feature 𝐱($ is
derived from a combination of log phone posterior (LPP) [17]
and log posterior ratio (LPR) [20]. Next, the encoder 𝜑 takes
as input the GOP feature sequence X $ to produce high-level
representations H = 𝜑(X $ ) , where H includes additional
utterance-level representations corresponding to five
trainable aspect embeddings prepended to the embeddings of
X $ . After that, regression heads 𝑓# are built upon H to predict
corresponding cross-level, aspect-specific proficiency scores
9 = {𝐲:!" | 𝑔 ∈ G, 𝑎 ∈
in parallel, resulting in predicted scores Y
"
A}, where 𝐲:! = 𝑓)" (H). The encoder 𝜑 and the regressor Θ
!
are learned by minimizing the MSE loss:
%

"

" &

ℒ*+, = |.|×|0| ∑"∈. ∑!∈0>𝐲! − 𝐲:! >& ,

(1)

where ‖. ‖& denotes the L2 norm.
2.3. Phonemic Contrast Ordinal Loss
As shown in Figure 1, the regression-based APA model like
GOPT, trained using the MSE loss, can capture the ordinal
relationship of the proficiency levels but tend to neglect the
distinction between phoneme categories, leading to the
aggregation of different categories of phonemes with the
same proficiency level. As a remedy, we propose a phonemic
contrast ordinal (PCO) loss by introducing a phonemedistinct regularizer into mean-squared error loss. This
regularizer consists of two mathematical terms, namely the
phonemic distinction and the ordinal tightness, which
seamlessly work together to regulate the distances within and

between the phoneme categories in both feature and target
space, respectively.
Applying the proposed loss function to different
regression heads at various linguistic levels has the potential
to improve the performance of the associated aspects.
However, as an initial attempt, this paper focus on design a
novel loss function that merely considers the phoneme-level
representation features (i.e., intermediate embeddings) H$ =
(𝐡% , 𝐡& , … , 𝐡' ) and their associated scores 𝐲 $ =
(𝑦%$ , 𝑦&$ , … , 𝑦'$ ), where H$ simply exclude the [CLS] token
from H.
Phonemic Distinction. The proposed PCO loss adopts a
phonemic distinction term ℒ$2 to address the distances
between inter- phoneme categories in the feature space.
Specifically, the phonemic distinction term ℒ$2 aims to
minimize negative distances between feature centers 𝐡3# ,
which is equivalent to maximizing the distances between
phoneme categories during the optimization process. The
feature centers 𝐡3# are calculated by taking a mean over all
some of the representation features H$ that belong to the
same phoneme category 𝑝4 . We define the phonemic
distinction term ℒ$2 by
%

ℒ$2 = − 5(57%) ∑5
4;% ∑49: E𝐡3# − 𝐡3$ E × 𝑚𝑐 ,
&

(2)

where 𝑀 is the number of feature centers in a batch of
samples or a sampled subset from a batch, and 𝑚" is a
positive hyper-parameter that stands for a margin to the
distance between two feature centers. Notably, owing to the
unbounded nature of the feature spaces, it is necessary to
normalize the features H$ before further processing.
Furthermore, the hyper-parameter 𝑚" empirically set to be 1
in our experiments.
Ordinal Tightness. Simply spreading the features may break
the inherent ordinality of the regression target output. As a
remedy, the ordinal tightness preserves the ordinal
relationships by considering the proficiency level when
reducing the scatterness of each phoneme category.
Formally, we introduce an additional ordinal tightness
term ℒ<= that minimizes the distance between each
representation feature 𝐡4 with its feature centers 𝐡3# while
being aware of pronunciation score in the label (output) space:
%

$
ℒ<= = ∑>
4;%>𝐡4 − 𝐡3# > × 𝑦4 ,
>

&

(3)

where 𝑁 is the batch size, and y4 is the corresponding
phoneme-level accuracy score. Adding this ordinal tightness
term encourages features with higher pronunciation scores to
be closer to their centers while those with lower scores are
farther away. The final formula of the PCO loss is defined by
ℒ$3< = ℒ*+, + 𝜆2 ℒ$2 + 𝜆< ℒ<= ,

(4)

where ℒ*+, is the MSE loss; and 𝜆2 and 𝜆< are tunable
parameters (set by cross-validation) that control the trade-off

Fig. 3. Visualization of phoneme representations learned from APA (GOPT) models trained with proposed PCO loss, we show (a) the
effect of phonemic distinction ℒ#$ with hyperparameters (𝜆$ , 𝜆% ) = (5,0), and (b) the synergy of using both phonemic distinction and
ordinal tightness ℒ%& with hyperparameters (𝜆$ , 𝜆% ) = (5,1).

balance between the phonemic distinction term ℒ$2 and the
ordinal tightness term ℒ<=
3. EXPERIMENTS
3.1. Dataset
We conducted APA experiments on the speechocean762
dataset, which is a publicly available open-source dataset
specifically designed for pronunciation assessment [21]. This
dataset contains 5,000 English-speaking recordings spoken
by 250 Mandarin L2 learners. The training and test sets are of
equal size, each of which has 2,500 utterances.
Speechocean762 contains comprehensive annotation
information, where pronunciation proficiency scores were
evaluated at multiple linguistic granularities with various
aspects. Each score was independently assigned by five
experts using the same rubrics, and the final score was
determined by selecting the median value from the five scores.
As with [15], we normalized utterance-level and word-level
scores to the same scale as the phone score (0-2) for training
APA models.
3.2. Implementation Details
Following settings presented in [15], we adopted the same
DNN-HMM acoustic models to extract 84-dimensional GOP
features. This acoustic model was based on a factorized timedelay neural network (TDNN-F) and trained using the
Librispeech 960-hour data with the widely-used Kaldi recipe
[22]. In order to evaluate the effectiveness of our proposed
loss function, we kept all training hyper-parameters of GOPT
compliant with the settings described in [15]. The number of
transformer block was set to be 3, with 24 hidden units in
each block. Each regression head is designed to use only one
layer for projecting the hidden representation features to their
corresponding pronunciation scores.

To ensure the reliability of our experimental results, we
repeated 5 independent trials, each consisting of 100 epochs,
using different random seeds. The experimental results are
reported by averaging the top 100 best-performing
experiments based on their Pearson Correlation Coefficient
(PCC) performance on the training set. The primary
evaluation metric is PCC, which measures the linear
correlation between predicted scores and ground-truth scores.
In addition, MSE value is used to assess phoneme-level
accuracy.
4. EXPERIMENTAL RESULTS
4.1. Visualization of Phoneme Representations
Before launching into a series of experiments on the APA
task, we visualize the phoneme-level intermediate
embeddings of the vanilla GOPT models optimized with the
proposed loss function under different settings, as shown in
Figure 3. In so doing, we can graphically examine the effects
of phonemic distinction and ordinal tightness on the phonedistinct regularizer. As demonstrated in Figure 3(a), it is
evident that the PCO loss with the phonemic distinction term
can effectively encourage feature representations to scatter
apart according to their respective phoneme categories. We
further plot the phoneme representations when adding the
ordinal tightness term in the Figure 3(b). From this figure, we
can observe that when the ordinal tightness term is added, the
phoneme representations with a proficiency score of 0 in
different phoneme categories are distributed further apart,
especially in the phoneme category of /EY/, /IH/ and /Z/. Due
to this property, GOPT trained with the proposed PCO loss
can significantly enhance the discriminability of phone
representations, meanwhile making them more sensitive to
the corresponding proficiency scores.

Table 1. Comparisons of performance among various APA models on speechocean762.
Phone-level

Model

Word-level (PCC)

Utterance-level (PCC)

MSE

PCC

ACC

Stress

Total

ACC

Completeness

Fluency

Prosody

Total

GOPT [15]

0.085

0.612

0.533

0.291

0.549

0.714

0.155

0.753

0.760

0.742

HiPAMA [17]

0.084

0.616

0.575

0.320

0.591

0.730

0.276

0.749

0.751

0.754

SBnum Loss [18]

0.086

0.605

0.531

0.386

0.547

0.722

0.427

0.750

0.752

0.747

PCO Loss

0.083

0.622

0.558

0.250

0.573

0.727

0.359

0.763

0.763

0.752

Phone-ACC

Word-ACC

Utt-ACC

0.8
0.714

0.716

0.723

0.731

0.728

0.612

0.613

0.620

0.619

0.619

0.546

0.552

0.553

0.548

1

3

5

7

PCC

0.7

0.6
0.533
0.5
0

(a) PCC results w.r.t. parameter !! .
Phone-ACC

Word-ACC

Utt-ACC

0.8
0.73

0.735

0.728

0.727

0.720

0.626

0.626

0.624

0.622

0.612

0.555

0.558

0.552

0.558

0.554

0.01

0.1

0.5

1

2

PCC

0.7

0.6

0.5

(b) PCC results w.r.t. parameter !" .

Fig. 4. Comparisons of the PCC results with respect to different
settings of hyper-parameters 𝜆$ and 𝜆% .

4.2. Overall Performance on the APA Task
In the second set of experiments, we discuss the overall
performance of our model in comparison with some other
completive models. To better verify the effectiveness of our
model on the APA task, we curate three top competitive
models stemming from the GOPT model, which were also
trained on speechocean762 without resort to any additional
speech datasets. The corresponding results are illustrated in
Table 1, from which we can make several observations. First,
the proposed method (denoted by PCO Loss) consistently
outperforms GOPT across three linguistic levels and various
aspects. In addition, our method stands out in terms of the
phoneme-level accuracy metric. Second, at the word-level

granularity, our method consistently outperforms SBnum
except for the stress evaluation. A possible reason for this is
that the stress aspect in the word-level granularity suffers
from the problem of severe data imbalance. As a side note
SBnum proposed a loss reweighting scheme for the MSE loss
so as to rebalance the loss contribution of frequent and
infrequent prediction cases of different aspects independently,
pertaining to their respectively training statistics [18].
Furthermore, our method slightly lags behind HiPAMA in the
evaluations of the word-level granularity, which may be
attributed to the better hierarchical model architecture of
HiPAMA that works in conjunction with the aspect attention
mechanisms. Third, in the utterance-level evaluation, our
method not only appears to perform on par with HiPAMA in
terms of the accuracy aspect and the total aspect, but also
outperforms the other competitive models in terms of the
fluency and prosody metrics which access the high-level
pronunciation skills taking into account factors such as
speaking style (e.g., repetition, stammering or hesitations),
rhythm and intonation. Notably, our method can effectively
distinguish phoneme representations belonging to lowerscoring groups from others by the ordinal tightness term,
simultaneously separating the representations according to
their phoneme categories with phonemic distinction term.
These collectively bring benefits to the evaluations on the
fluency and prosody aspects.
4.3. Ablation Studies
In the last set of experiments, we conducted ablation studies
to analyze the contribution of each component involved in the
PCO loss at three linguistic levels in terms of the PCC metric
on the accuracy aspect.
Effect of Phonemic Diversity. We first analyze the effect of
the phonemic diversity term by varying the parameter 𝜆2 .
Here we remove the ordinal tightness term (i.e., setting 𝜆< =
0), and report corresponding performance in Figure 4(a).
From this figure, we can observe that the performance of the
three linguistic levels is boosted as the value of 𝜆2 increases,
but tends to reach a plateau when 𝜆2 exceeds 5. Furthermore,
the best performance for the word- and utterance-level

speaking proficiency evaluations is achieved when 𝜆2 is set
to 5.
Effect of Ordinal Tightness. Next, we investigate the
effectiveness of the ordinal tightness term by altering the
value of parameter 𝜆< while keeping 𝜆2 fixed at 5. As shown
in Figure 4(b), we can observe a decreasing trend in the
accuracy evaluations of all granularity levels as the value of
𝜆< becomes larger. When 𝜆< is set equal to 0.1, the associated
evaluations of all granularity level yield the best performance.
5. CONCULSION AND FUTURE WORK
This paper has put forward a simple yet effective loss
function, dubbed the phonemic contrast ordinal (PCO) loss
for the APA task. The PCO loss introduces a phonemedistinct regularizer into the MSE loss to regulate the distances
between inter- and intra-phoneme categories in both the
feature and target spaces. A series of empirical experiments
conducted on the speechocean762 benchmark dataset has
revealed the feasibility of our proposed model in comparison
to some top-of-the-line models. In future work, we plan to
pair the PCO loss with more sophisticated model structures
that can integrate lexical and phonological cues, as well as
context-aware hierarchical information [24][25][26].
6. ACKNOWLEDGEMENTS
We are grateful to all the anonymous reviewers for their
helpful advice on various aspects of this work. This work was
supported in part by E.SUN bank under Grant No. 202308NTU-03. Any findings and implications in the paper do not
necessarily reflect those of the sponsors.
7. REFERENCES
[1]

S. Bannò, K. M. Knill, M. Matassoni, and V Raina, “L2
proficiency assessment using self-supervised speech
representations,” arXiv preprint arXiv:2211.08849, 2022.

[2]

A. M. Kamrood, M. Davoudi, S. Ghaniabadi, and S. M. R.
Amirian, “Diagnosing L2 learners’ development through
online computerized dynamic assessment,” Computer
Assisted Language Learning, vol. 34, pp. 868–897, 2021.

[3]

[4]

[5]

R. Ai, “Automatic pronunciation error detection and feedback
generation for call applications,” in Proceedings of the Second
International Conference on Learning and Collaboration
Technologies (LCT), pp. 175–186, 2015.
W. Li, S. M. Siniscalchi, N. F. Chen, and C. Lee, “Improving
non-native mispronunciation detection and enriching
diagnostic feedback with DNN-based speech attribute
modeling,” in Proceedings of the IEEE International
Conference on Acoustics, Speech and Signal Processing
(ICASSP), pp. 6135–6139, 2016.
E. B. Page, “Statistical and linguistic strategies in the computer
grading of essays,” in Proceedings of the conference on
Computational linguistics (COLING), pp. 1–13, 1967.

[6]

J. Shi, N. Huo and Q. Jin, “Context-aware goodness of
pronunciation for computer-assisted pronunciation training,”
in Proceedings of Interspeech (INTERSPEECH), pp. 3057–
3061, 2020.

[7]

D. Korzekwa, J. Lorenzo-Trueba, T. Drugman, and B. Kostek,
“Computer-assisted pronunciation training—Speech synthesis
is almost all you need,” Speech Communication, vol. 142, pp
22–33, 2022.

[8]

K. Li, X. Wu, and H. Meng, “Intonation classification for L2
English speech using multi-distribution deep neural networks,”
Computer Speech & Language, vol. 43, pp. 18–33, 2017.

[9]

Z. Zhang, Y. Wang, and J. Yang, “Text-conditioned
transformer for automatic pronunciation error detection,”
Speech Communication, vol. 130, pp. 55–63, 2021.

[10] B.-C. Yan, H.-W. Wang, and B. Chen, “Peppanet: Effective
mispronunciation detection and diagnosis leveraging phonetic,
phonological, and acoustic cues,” in Proceedings of the IEEE
Spoken Language Technology Workshop (SLT), pp. 1045–
1051, 2023.
[11] D. Korzekwa, R.Barra-Chicote, S. Zaporowski2, G. Beringer,
J. Lorenzo-Trueba, A. Serafinowicz, J. Droppo, T. Drugman,
and B Kostek , “Detection of lexical stress errors in non-native
(L2) English with data augmentation and attention,” in
Proceedings of the Annual Conference of the International
Speech Communication Association (INTERSPEECH), pp.
3915–3919, 2021.
[12] B.-C. Yan, H. W.-Wang, Y.-C. Wang, and B. Chen, “Effective
graph-based modeling of articulation traits for
mispronunciation detection and diagnosis,” in Proceedings of
the IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pp. 1–5, 2023.
[13] C. Richter and J. Guðnason, “Relative dynamic time warping
comparison for pronunciation errors,” in Proceedings of the
IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pp. 1–5, 2023.
[14] D. Y. Zhang, S. Saha, and S. Campbell, “Phonetic RNNtransducer for mispronunciation diagnosis,” in Proceedings of
the IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pp. 1–5, 2023.
[15] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and James Glass,
“Transformer-based multi-aspect multigranularity non-native
English speaker pronunciation assessment,” in Proceedings of
the IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pp. 7262–7266, 2022.
[16] F.-A. Chao, T.-H. Lo, T.-I. Wu, Y.-T. Sung, and B. Chen, “3M:
An effective multi-view, multi-granularity, and multi-aspect
modeling approach to English pronunciation assessment,” in
Proceedings of the Asia-Pacific Signal and Information
Processing Association Annual Summit and Conference
(APSIPA ASC), pp. 575–582, 2022.
[17] H. Do, Y. Kim, and G. G. Lee, “Hierarchical pronunciation
assessment with multi-aspect attention,” in Proceedings of the
IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pp. 1–5, 2023.

[18] H. Do, Y. Kim, and G. G. Lee, “Score-balanced loss for multiaspect pronunciation assessment,” in Proceedings of the
Annual Conference of the International Speech
Communication Association (INTERSPEECH), pp. 4998–
5002, 2023.
[19] Y. Liang, K. Song, S. Mao, H. Jiang, L. Qiu, Y. Yang, D. Li,
L. Xu, and L. Qiu, “End-to-end word-level pronunciation
assessment with MASK pre-training.,” arXiv preprint
arXiv:2306.02682, 2023.
[20] S.M. Witt and S. J. Young, “Phone-level pronunciation scoring
and assessment for interactive language learning,” Speech
Communication, vol. 30, pp. 95–108, 2000.
[21] W. Hu, Y. Qian, F. K. Soong, and Y. Wang, “Improved
mispronunciation detection with deep neural network trained
acoustic models and transfer learning based logistic regression
classifiers,” Speech Communication, vol. 67, pp. 154–166,
2015.
[22] J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K.
Li, D. Povey, and Y. Wang, “speechocean762: An open-source
non-native English speech corpus for pronunciation
assessment,” in Proceedings of the Annual Conference of the
International
Speech
Communication
Association
(INTERSPEECH), pp. 3710–3714, 2021.
[23] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur,
“Librispeech: An ASR corpus based on public domain audio
books,” in Proceedings of the IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP), pp.
5206–5210, 2015.
[24] S. Mao, F. Soong, Y. Xia, and J. Tien, “A universal ordinal
regression for assessing phoneme-level pronunciation,” in
Proceedings of the IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), pp. 6807–
6811, 2022.
[25] J. Zeng, Y. Xie, X. Yu, J. Lee, and D.-X. Zhou, “Enhancing
automatic readability assessment with pre-training and soft
labels for ordinal regression,” in Proceeding of Findings of the
2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP), pp. 4557–4568, 2022.
[26] W. Liu, K. Fu, X. Tian, S. Shi, W. Li, Z. Ma, and Tan Lee, “An
ASR-free fluency scoring approach with self-supervised
learning,” in Proceedings of the IEEE International
Conference on Acoustics, Speech and Signal Processing
(ICASSP), pp. 1–5, 2023.

