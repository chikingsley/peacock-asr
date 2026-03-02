# Segmentation-Free Goodness of Pronunciation

JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 1
Segmentation-Free Goodness of Pronunciation
Xinwei Cao, Zijian Fan, Torbjørn Svendsen, Senior Member, IEEE, Giampiero Salvi, Senior Member, IEEE
Abstract—Mispronunciation detection and diagnosis (MDD)
is a significant part in modern computer-aided language learning (CALL) systems. Most systems implementing phoneme-level
MDD through goodness of pronunciation (GOP), however, rely
on pre-segmentation of speech into phonetic units. This limits
the accuracy of these methods and the possibility to use modern
CTC-based acoustic models for their evaluation. In this study,
we first propose self-alignment GOP (GOP-SA) that enables the
use of CTC-trained ASR models for MDD. Next, we define a
more general segmentation-free method that takes all possible
segmentations of the canonical transcription into account (GOPSF). We give a theoretical account of our definition of GOPSF, an implementation that solves potential numerical issues as
well as a proper normalization which allows the use of acoustic
models with different peakiness over time. We provide extensive
experimental results on the CMU Kids and speechocean762
datasets comparing the different definitions of our methods,
estimating the dependency of GOP-SF on the peakiness of the
acoustic models and on the amount of context around the target
phoneme. Finally, we compare our methods with recent studies
over the speechocean762 data showing that the feature vectors
derived from the proposed method achieve state-of-the-art results
on phoneme-level pronunciation assessment.
Index Terms—Computer-aided pronunciation training, mispronunciation detection and diagnosis, speech assessment, goodness of pronunciation, CTC, child speech, L2.
## I INTRODUCTION
COMPUTER-AIDED language learning (CALL) and
computer-assisted pronunciation training (CAPT) are becoming more important and helpful among language learners
and teachers both because they are ubiquitously available and
because they maintain a high degree of self-controlled manner
of study. One of the desirable features for these systems is the
ability to provide instant feedback and intervention when the
learner makes any mispronunciation. However, the automatic
mispronunciation detection and/or diagnosis (MDD) modules
currently available are not sufficiently reliable.
MDD can be performed at different linguistic levels:
phoneme, word, and utterance. One of the main challenges
with MDD is the scarcity of data that is specifically annotated
for the task, including information about mispronunciations.
This problem is especially severe for phoneme-level assessment which is the focus of this paper.
Witt et al. [1] proposed a widely used method for MDD at
the phoneme level that is based on a measure called goodness
of pronunciation (GOP). The advantage of this method is that
it relies on acoustic models exclusively trained for automatic
speech recognition (ASR). The ASR models are used to
All authors are with the Department of Electronic Systems, Norwegian University of Science and Technology (NTNU), Norway. {xinwei.cao, zijian.fan,
torbjorn.svendsen, giampiero.salvi}@ntnu.no
This research was partly funded by the NordForsk Teflon project (#103893)
and by the Research Council of Norway SCRIBE project (#322964).
score new utterances, and only a small amount of MDD
annotations are required to optimize a detection threshold that
separates acceptable pronunciations from mispronunciations.
ASR models can also be used to transcribe speech at the word,
character, or phoneme level and detect mispronunciations by
comparing the resulting output sequence with the canonical
pronunciation or orthographic transcription, respectively [2],
[3], [4], [5], [6].
Another approach is to train end-to-end models for MDD.
However, training such models from scratch would require
large amounts of MDD annotated data. A solution is to use
foundation models that are either trained on large amounts of
unlabeled speech data, or fine-tuned for ASR. These models
can be further fine-tuned with small amounts of MDD annotated data for the MDD task. For example, Xu et al. [7] follow
this strategy starting from a Wav2Vec2.0 model [8] whereas,
the authors of [9], [10] start from a HuBERT model [11]. Liu
et al. [12] attempt to perform MDD at utterance-level based
on hidden representations directly derived from the foundation
models without further fine-tuning.
In this paper, our goal is to combine the advantages of the
GOP method, the foundation models, and end-to-end ASR
training based on CTC loss [13]. This raises a number of
challenges due to the fact that GOP requires segmentation
of speech into phonetic units. This is typically obtained by
forced alignment of the canonical pronunciation with the
spoken utterance. If the pronunciation is correct, the obtained
phonetic boundaries may vary due to coarticulation effects. In
case of mispronunciations, the segmentation may be even less
reliable. Finally, CTC trained models tend to give activations
that are not necessarily aligned with acoustic speech segmentation. In [14] we introduced a framework for combining
CTC trained models and GOP. In particular, we introduced
1) a self-aligned GOP method that uses the same activations
from CTC trained models for alignment and GOP evaluation,
and 2) a segmentation-free GOP method that can assess a
specific speech segment without committing to a particular
segmentation.
In this paper, we enhance these methods by making the
following novel contributions:
• We define proper segment length normalization for our
segmentation-free GOP definition without committing to
a particular segmentation.
• We provide a theoretical derivation of the segmentationfree GOP method that exposes the assumptions required
for its definition.
• We introduce a novel implementation of our method that
eliminates numerical problems.
• We provide extensive experimental results that compare
the different methods we propose to the state-of-the-art
on binary and ternary MDD tasks.
arXiv:2507.16838v3 [eess.AS] 5 Feb 2026
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 2
0
1
2
3
4
5
6
7
8
frequency (kHz)
Target
Phone
0
1
amplitude
2.5 2.6 2.7 2.8 2.9 3.0
time(seconds)
transc.
l p ix n dcl t ix q ih n ih w ah n hv ux tcl t r ay dcl t ix s tcl t er tcl t
happened to anyone who tried to start
Fig. 1. Standard GOP methods use forced alignment to segment speech for
analysis.
• We provide an analysis of the effect of peakiness of the
model on the segmentation-free GOP method.
• We provide an analysis of the effect of context length
(utterance length) on the segmentation-free GOP method.
• We make all code available.1
The paper is organized as follows: in Section II we provide
the background and review the related work that is necessary
to understand our methods. In Section III we give our GOP
definitions and theoretical derivations showing the assumptions
that are implicit in them. Sections IV and V report on our
experimental settings and empirical results. Finally, Section VI
concludes the paper.
## II BACKGROUND AND RELATED WORK
In this section we give background information and review
the related work that are necessary to understand the proposed method. We will focus on phoneme-level pronunciation
assessment that is the goal of this paper. We first need to
distinguish between continuous measures for pronunciation
assessment and the specific task of MDD. In the first case,
the goal is to introduce a metric that can indicate how far
a specific pronunciation of a phoneme is from the acceptable
variability in the language. An example of this kind of measure
is GOP. The MDD task, on the other hand, is to provide
a binary or, sometimes ternary, decision on whether the
pronunciation is correct or not. This task can be performed
1) on a single GOP-like score by defining thresholds for the
different output classes, 2) by more advanced classifiers based
on multidimensional engineered features, or 3) by end-to-end
systems that take raw speech as input.
In this paper, we focus on MDD methods that are based
on GOP scores and GOP features. We will first present the
different definitions of GOP in Section II-A, the problem
arising from the need for accurate speech segmentation in
Section II-B, and, finally, the problem of using end-to-end
peaky models with GOP in Section II-C.
A. The definition of GOP
GOP was initially proposed as a measure of how closely
the pronunciation of a specific phoneme matches its expected
canonical pronunciation [1]. This is a segmental measure that
1https://github.com/frank613/CTC-based-GOP
requires time-aligned phonetic annotations (see, e.g., Figure 1).
Witt’s original definition of GOP [1] corresponds to the log
posterior of the canonical phoneme li given the sequence of
observations Ot2
t1
= {ot1
,...,ot2
} in the segment under test,
normalized by the sequence length:
GOP(li) =
log(p(li|Ot2
t1
))
t2 − t1
. (1)
Mispronunciations are detected on the basis of the GOP value
and an empirically determined threshold.
The estimation of the posteriors was originally implemented
using Hidden Markov Models and Gaussian Mixture Models (HMM-GMM) and by using Bayes rule:
p(li|Ot2
t1
) =
p(Ot2
t1
|li)P(li)
P
q∈Q p(Ot2
t1
|q)P(q)
, (2)
where P(q) represents the prior probability of each phoneme
in the phoneme inventory Q and the likelihood p(Ot2
t1
|li) can
be directly evaluated using the acoustic model, for example
with the forward algorithm. For efficiency reasons, Witt approximates the summation in the denominator by obtaining
the best path over the full utterance through a phone-loop
model. From Eq. (2), GOP is then the ratio of the likelihood
of the segment estimated from the canonical phone model
(numerator) and the likelihood of the best path (allowing any
phone sequence) through the segment (denominator).
The phoneme boundaries t1 and t2 could be obtained by
human annotations. However, this is not practical: During
model training, annotating large amounts of data would be
too costly. More importantly, this would make the methods
not suitable for providing immediate and automated feedback
to the students. The segmentation step is, therefore, commonly
automated by forced alignment using the canonical pronunciation LC = {l1,...,l|LC|} and a phonetic model trained for
ASR. This model does not need to be the same as that used for
the GOP evaluation, and it is typically a context-independent
HMM-GMM model.
Although Witt’s GOP has been successful, many later works
argue that the phone-loop approximation is not reliable in
estimating the denominator of Eq. 2. For example, Latticebased GOP [15] includes contributions from N-best hypotheses
to avoid underestimating the value of the denominator. In [16],
the authors show performance improvements that can be
obtained when the phone-loop is evaluated multiple times over
the target segment for each phoneme rather than once along
the whole utterance. This implementation follows more closely
the original definition of the denominator in Eq.2.
Phonological rules from the learner’s first language (L1)
can also be included in the GOP methods, where the acoustic
models are trained with target language (L2) only, to achieve
better accuracy [17], [18].
The models used for estimating the GOP values are trained
for ASR and therefore do not need any data that are annotated
specifically for the MDD task. However, in order to detect
mispronunciations, a small amount of human-annotated data
is required to estimate the optimal threshold that separates
good pronunciations from bad ones. These thresholds are
usually phoneme-dependent but even finer-grained definitions
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 3
(for example speaker-dependent) may improve accuracy (e.g.
[19]).
Using hybrid HMM-Deep Neural Networks (DNN) acoustic
models has been shown to improve MDD performance [20].
In this case, the posterior defined in Eq. 1 can be evaluated
directly from the posterior probabilities p(li|ot) from the
softmax output of the last DNN layer for the acoustic frames
in the target phoneme. The corresponding GOP definition is
then the average of the log posterior probabilities:
GOP-Avg(li) =
1
t2 − t1
t2 X
t=t1
log(p(li|ot)). (3)
The DNN is often trained frame-wise with cross entropy (CE) loss. Similarly to the HMM-GMM based GOP, the
segment boundaries t1 and t2 are obtained from an external
aligner that is also used to provide the frame-wise labels for
training the DNN. We call this method external alignment
(EA) in the following. In [21], the authors show that the
performance of DNN-based GOP can be further improved
by merging the probabilities of fine-grained sub-phoneme
symbols (senones) to phonemes.2
In Eq. 3 each frame is evaluated independently using the
output of the DNN, neglecting the transition probabilities in
the HMM. In [22], the author argues that these probabilities
can contribute with contextual information and including them
in the GOP evaluation can improve performance. In [23],
transition and duration factors are introduced to increase
context awareness in GOP.
One of the issues with using GOP for MDD is that the
definitions in Eqs. 2 and 3 only account for substitution errors.
However, deletion and insertion of phonemes could also be
potential sources of mispronunciations. This is one of the
limitations that we address in this work.
B. The alignment issues for GOP
There are several issues with traditional definitions of GOP
related to speech segmentation. The first is the ability to perform a perfect alignment of phonetic segments to the recorded
speech. Phonetic segmentation is an ill-posed problem because
it is based on the assumption that speech is produced as a
sequence of discrete events. However, coarticulation effects in
speech production question the existence of explicit phonetic
boundaries. This explains the disagreement on segmentation
even between trained phoneticians. An example is shown in
Figure 1, where it is difficult to tell the exact beginning and
end of the sonorant [w] as it happens in the transition of two
vowel sounds [ı] and [2].
Even assuming the existence of well-defined phonetic
boundaries, alignment errors may occur in case of mispronunciations because the models used for segmentation are usually
trained on correctly pronounced speech. Furthermore, as studied in [24], [25], [26], forced alignment may be unreliable due
to speaker variability: age, accent, dialect, health condition, to
name a few. The uncertainty over phonetic boundaries has a
2To compare with CTC-based methods (GOP-CTC-SF and GOP-CTCSA) where GOP is also operated at phoneme level, our later discussion and
experiments are all based on models trained directly at phoneme level.
strong impact on traditional GOP definitions which consider
the boundaries as deterministic.
Another important limitation of traditional GOP emerges
from the characteristics of the models and loss functions used
in modern ASR training. Even assuming perfect segmentation of speech, there is no guarantee that the activations of
the models used for GOP evaluation are aligned with this
segmentation. A typical case is with end-to-end transformerbased models where the alignment between input speech and
output symbols is somewhat arbitrary. This aspect is rarely
emphasized in the literature, where the method used for
alignment is often not specified.
In [27] the authors propose to use the general posterior
probability (GPP) to mitigate alignment issues by allowing
evaluation over any segments that overlap with the target
segment. Zhang et al. [28] propose a pure end-to-end method
that uses a sequence-to-sequence model without having to
segment the speech. However, this method requires a large
number of human annotations for MDD and an additional
step that compares the canonical phoneme sequence with the
human-annotated sequence before training the model.
In this work, we propose two methods to relax the dependency of GOP on the accuracy of segmentation. The first
has the goal of reducing the effect of misalignment between
the acoustic model activations and the speech segments, the
second is completely segmentation-free. The latter also allows
for insertion and deletion errors that were mentioned in
Section II-A.
## C End-to-end ASR models, CTC and peaky behavior
In this section, we introduce some details of the Connectionist Temporal Classification (CTC) loss used in modern endto-end ASR models. This premise is important to understand
our methods in Section III.
End-to-end ASR models were initially introduced to map
speech to output symbols, typically characters3
, directly. In
general, the sequence of output symbols L = {l1,...,l|L|}
has a different rate that the input speech feature vectors
OT
1 = {o1,...,oT }. To overcome this problem in training,
the CTC loss [13] was introduced. This makes use of an
additional blank (ϕ) output symbol. Given a speech sequence
OT
1 of length T, we define V as the set of all output symbols
including ϕ. Then any vector u = {u1,...,uT } ∈ U = V T
is
an alignment path between the input sequence and the output
symbols. The probability of a path u under the assumption
of conditional independence is p(u|OT
1 ) =
QT
t=1 p(ut|ot). As
a consequence
P
u∈U p(u|OT
1 ) = 1 representing all possible
paths given the model and the speech. During training, for a
given target output sequence L and the speech sequence OT
1 ,
the model learns to minimize the following loss:
LCTC(L) = −log
X
π∈{u:B(u)=L}
p(π|OT
1 ) (4)
where π spans over the paths that can be mapped to L using the
many-to-one function: B : U → L by removing all repeated
3The methods for MDD described in this paper use phonemes as output
symbols. However, all the arguments in this section apply regardless of the
nature of the output symbols.
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 4
0.0
0.5
1.0
sil SH AE M sil
sil SH AE M sil
sp
fa
ac
Correctpronunciation,CELoss
0.0
0.5
1.0
sil SH AE M sil
sil SH AE M sil
sp
fa
ac
Correctpronunciation,CTCLoss
0.0
0.5
1.0
sil SH AE L sil
sil SH AE M sil
sp
fa
ac
Substitutionerror,CELoss
0.0
0.5
1.0
sil SH AE L sil
sil SH AE M sil
sp
fa
ac
Substitutionerror,CTCLoss
0.0
0.5
1.0
sil SH AE sil
sil SH AE M sil
sp
fa
ac
Deletionerror,CELoss
0.0
0.5
1.0
sil SH AE sil
sil SH AE M sil
sp
fa
ac
Deletionerror,CTCLoss
0 20 40 60 80 100
time(frames)
0.0
0.5
1.0
sil SH S AE M sil
sil SH AE M sil
sp
fa
ac
Insertionerror,CELoss
0 20 40 60 80 100
time(frames)
0.0
0.5
1.0
sil SH S AE M sil
sil SH AE M sil
sp
fa
ac
Insertionerror,CTCLoss
Fig. 2. Illustration of the issues with standard GOP methods for models trained with CE loss (left) and CTC loss (right). Each plot corresponds to a different
kind of mispronunciation and is divided in three parts: “sp” shows that segments that were actually spoken; “fa” shows the forced alignment based on the
canonical pronunciation; “ac” shows the activations from the model and for the canonical segments. Finally, for the CTC models, the red dashed lines show
the alternative forced alignment from the GOP-CTC-align method.
symbols and followed by removal of the blank symbols, e.g.,
B(aϕbϕbϕccϕ) = abbc.
Since training the CTC does not require frame-level alignment, the timing information tends to be ignored by the
model. A well-known phenomenon of CTC-trained model
is the “peaky” behavior of model output, as illustrated in
the right column of Figure 2. There are two dimensions of
peakiness: “peaky over time” (POT) and “peaky over state”
(POS). POT behavior corresponds to the fact that the blank
symbol is activated for most of the time steps, whilst the nonblank symbols that correspond to the target label sequence
only become activated for a few time steps. Forced alignment
(Viterbi search) performed with CTC-trained models can result
in distorted alignment between input speech and output symbols. On the other hand, POS behavior is observed because
the model activations at each frame are always close to 1.0
for one output and 0.0 for all the others.
For these reasons, CTC is not feasible for applications where
the time alignment is essential, such as voice activity detection,
speech segmentation, or conventional GOP evaluation for
MDD. An indication of this is the lack of literature in these
areas that use CTC-based models. However, several works
tried to mitigate the peaky behavior of CTC-based models for
ASR, e.g. [29], [30], [31].
The two methods proposed in this paper, have as goal to
make it possible to use CTC-trained models with phoneme
output targets as a basis for computing GOP for MDD tasks.
## III METHODS
The goals of our methods are (i) to make it possible to use
CTC-based models for a reliable evaluation of GOP, (ii) to
reduce the sensitivity of GOP to precise speech segmentation
and to consider the uncertainty of phonetic alignment, (iii) to
introduce context awareness into the definition of GOP and
(iv) to extend the method of GOP to allow for detecting
insertion and deletion errors as well as substitutions.
To achieve these goals we propose two methods that will
be detailed in Section III-A and III-B.
A. Self-alignment GOP (GOP-SA)
Firstly, we consider the problem of mismatch between the
segmentation of speech into phonetic units and the activations
of the models used for GOP estimation as mentioned in
Section II-B. This phenomenon is exemplified in Figure 2
and is especially critical for CTC trained models (right plots)
which exhibit peaky activations for each phoneme that may not
correspond in time with the corresponding speech segment. In
order to address the problem, we propose to use the same
GOP definition as for DNN models (Eq. 3) but to perform
the alignment of the target segment (t1,t2) based on the same
model used for GOP evaluation instead of an external forced
aligner. We refer to this method as self-alignment GOP (GOPSA). Figure 2 shows this alignment with red dashed vertical
lines for the CTC-Loss trained models. We want to stress
that the goal of alignment in this method is not to find the
segment corresponding to the target phone, but, rather, to find
the activations of the model corresponding to the target phone.
We hypothesize that using CTC-based models for GOP
would reduce the impact of the alignment errors due to mispronunciations which we mentioned in Sec.II-B. We will show
that this method leads to improvements in MDD accuracy not
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 5
φ φ φ M AE SH
OU
AH
N
φ
φ φ φ M AE SH φ
Fig. 3. The figure exemplifies the computation of GOP-SF (Eq. 12) using
a CTC-trained model. In the example, LCTC(LC,OT
1 ) can be computed
using the graph at the top, whereas LCTC(LSDI,OT
1 ) using the graph at the
bottom. For the sake of simplicity we omit the skip connections and self-loop
connections.
only for the peaky models it was designed for, but for all the
other models we have tested.
B. Segmentation-free GOP (GOP-SF)
The second method is segmentation-free (SF) and evaluates
the GOP for a particular speech segment without the need for
explicitly segmenting the utterance under test. The motivation
for this method is to overcome the following limitations of
standard GOP: 1) the evaluation of pronunciation of each
phoneme is exclusively based on the corresponding speech
segment (see Figure 1); 2) the evaluation of GOP is sensitive
to the specific alignment (t1,t2) and does not take into account
uncertainty in alignment; 3) the common implementation of
GOP uses Viterbi decoding and therefore only considers one
path through the model possibly leaving out part of the
probability mass; finally, 4) standard GOP does not allow for
insertion and deletion errors.
We assume that we have recorded an utterance
OT
1 = {o1,...,oT } with a canonical transcription
LC = {l1,...,li,...,l|LC|}. We are interested in evaluating
the pronunciation of a phone li. We can therefore split the
canonical transcription into three parts, the left context (LL)
the target phone li, and the right context (LR):
LC = {l1,...,li−1
| {z }
LL
, li,
|{z}
target phone
li+1,...,l|LC|
| {z }
LR
}.
As in previous sections, we define t1 and t2 as the start and
end frame index for the target phone li. Instead of committing
to a specific segmentation as in standard GOP, in the proposed
segmentation-free GOP (GOP-SF) we compute the log posterior for the target phone li given the full observation sequence
OT
1 and all the phonemes in the left and right context:
GOP-SF(li) = log p(li|OT
1 ,LL,LR)

. (5)
This is the same definition that we introduced in [14], although
not explicitly written as in Eq. 5, there.
In this work, we also introduce a version of this definition
that is normalized by the estimated length of the model
activations for the target speech segment:
GOP-SF-Norm(li) =
log p(li|OT
1 ,LL,LR)

E[t2 − t1|OT
1 ,LL,LR]
. (6)
The reason for this new definition is to reduce the variance of
the GOP estimates with the different lengths of the activations.
This is similar to the standard GOP definition of Eq. 1, where
the posterior is normalized by |t2 − t1| with the following
differences: 1) our normalization factor is not related to the
length of the target segment, but, rather, to the length of the
activations of the model corresponding to the target segment.
This accommodates both peaky and non-peaky models; 2) we
do not commit to a specific speech segmentation and incorporate the uncertainty in the alignment into our estimation.
In the following subsections, we give theoretical and practical accounts of how to estimate the numerator and denominator in Eqs. 5 and 6.
## C Segmentation-free target phoneme posterior estimation
In this section we show how to compute the argument of the
log function in Eqs. 5 and 6, that is the alignment-free estimation of the posterior for the target phoneme p(li|OT
1 ,LL,LR).
We first rewrite the expression using the chain rule of probabilities:
p(li|OT
1 ,LL,LR) =
p(LL,li,LR|OT
1 )
p(LL,LR|OT
1 )
. (7)
We now consider a specific alignment (t1,t2) for the target
phone li and we define the set of all possible alignments as
A(li) = {(t1,t2) : i − 1 < t1 ≤ t2 < T − (|LC| − i)}. (8)
Because the left context LL and the right context LR contain
respectively i − 1 and |LC| − i symbols, the lower bound for
t1 and the upper bound for t2 ensure that there is at least one
frame for each phone in the left and right context.
We can now expand numerator and denominator in Eq. 7
by considering the specific alignment (t1,t2) and by summing
over all possible alignments A(li):
p(LL,li,LR|OT
1 )
p(LL,LR|OT
1 )
=
=
P
(t1,t2)∈A(li)
p(LL,li,LR,t1,t2|OT
1 )
P
(t1,t2)∈A(li)
p(LL,LR,t1,t2|OT
1 )
=
=
P
(t1,t2)∈A(li)
p(LL,li,LR|OT
1 ,t1,t2)p(t1,t2|OT
1 )
P
(t1,t2)∈A(li)
p(LL,LR|OT
1 ,t1,t2)p(t1,t2|OT
1 )
. (9)
Special attention must be paid to the term p(t1,t2|OT
1 )
within both sums at the numerator and denominator of Eq. 9.
This is the probability of a certain alignment for the ith
segment, given the observation sequence, but independent of
the actual transcription. We can interpret this as a prior with
respect to the transcription over all possible segmentations.
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 6
This distribution could be estimated from the training data.
However, in our definition of GOP-SF, we make the simplifying assumption that this distribution is uniform. Under this
assumption p(t1,t2|OT
1 ) is a constant and can be taken out of
the sums and finally cancels out between the numerator and
the denominator of Eq. 9 which becomes:
p(LL,li,LR|OT
1 )
p(LL,LR|OT
1 )
=
P
(t1,t2)∈A(li)
p(LL,li,LR|OT
1 ,t1,t2)
P
(t1,t2)∈A(li)
p(LL,LR|OT
1 ,t1,t2)
.
(10)
Finally, the terms within the sums in Eq. 10 can be computed with the CTC’s forward variables α and β as defined in
[13]:
P
(t1,t2)∈A(li)
αt1−1(i − 1)
hQt2
t=t1
yt(li)
i
βt2+1(i + 1)
P
(t1,t2)∈A(li)
αt1−1(i − 1)βt2+1(i + 1)
. (11)
In fact, the numerator in Eq. 11 is equivalent to the original
definition of the CTC loss for the canonical pronunciation
LC and the acoustic features OT
1 , except for a −log(.) term.
Similarly, the denominator can be computed with CTC loss if
we define a modified canonical pronunciation LSDI as:
LSDI = {l1,...,li−1
| {z }
LL
, .∗,
|{z}
any phoneme sequence
li+1,...,l|LC|
| {z }
LR
,}.
This is the set of all possible transcriptions in which the left
and right contexts are equal to the canonical transcription, but
we allow any sequence of phonemes in place of the target
phoneme li
4
. This corresponds to allowing any number of
“substitution”, “deletion” and “insertion” errors in pronunciation, justifying the SDI subscript. This CTC loss can be
implemented by defining a graph exemplified in Figure 3
(bottom) and running the forward algorithm using this graph.
In summary, combining Eqs 5 and 11, the GOP-SF can be
efficiently computed as the difference between the CTC loss
for the pair (LSDI,OT
1 ) and the CTC loss for (LC, OT
1 ):
GOP-SF(li) = LCTC(LSDI,OT
1 ) − LCTC(LC,OT
1 ). (12)
Note that, because we are considering contributions from
all possible segmentations, our method is able to deal with
uncertainty in alignment. Also, the definition of LSDI with
a graph as in Figure 3 gives us flexibility on the kind of
pronunciation errors we consider. If the full graph is used,
any substitution (S), deletion (D) and insertion (I) errors are
considered. We refer to this version of the method as GOPSF-SDI. If we remove the blue path from the graph, we
only consider substitutions and deletions (GOP-SF-SD). This
corresponds to a modified canonical pronunciation
LSD = {l1,...,li−1
| {z }
LL
, .?,
|{z}
any phoneme or empty
li+1,...,l|LC|
| {z }
LR
,}.
4In this and later expressions, we have borrowed notation from regular
expressions where “.” corresponds to any phoneme, “*” is zero or more
occurrences, and “?” is zero or one occurrence.
If we also remove the red path, we only consider substitutions as in the traditional GOP (GOP-SF-S) and the modified
canonical pronunciation is
LS = {l1,...,li−1
| {z }
LL
, .,
|{z}
any phoneme
li+1,...,l|LC|
| {z }
LR
,}.
In [14] we used the unnormalized forward variables αs
to compute LCTC(LSDI,OT
1 ) due of implementation issues.
However, this may lead to numerical underflow errors when
multiplying probabilities over longer speech segments. In this
paper, we implement a version of the algorithm that uses the
normalized forward variables α̂t(s) defined in [13] for both
terms in Eq. 12. When making comparisons, we refer to this
version as GOP-SF-numerical.
In the experiments, we also consider alternative methods to
compute the loss, besides standard CTC.
D. Segmentation-free activation length estimation
We now turn to the denominator of Eq. 6, that is on
the estimation of the model activation length for the target
phoneme E[t2 − t1|OT
1 ,LL,LR].
We call NC the set of central nodes (yellow and purple) in
the graph in Figure 3 that correspond to the “.*” term in LSDI.
Then, the expected value of the duration of the activations
corresponding to li, is the sum over the whole observation
sequence of the normalized forward variables α̂ corresponding
to the nodes in NC:
E[t2 − t1|OT
1 ,LL,LR] =
T X
t=1
X
s∈NC
α̂t(s) := Occ(i). (13)
This expression can be efficiently computed with the forward
algorithm that is also used to estimate the posterior of the
target phoneme li detailed in the previous section. Note that
the definitions of LSDI and LSD allow deletion of the target
segment as a possible mispronunciation error. In this case,
E[t2 −t1|OT
1 ,LL,LR] will tend to zero and GOP-SF will tend
to diverge. In the actual implementation, therefore, we define
a floor value of 1.
We expect this normalization factor to be most relevant for
models that are not peaky in time. Peaky models tend to have
activations that span over a single frame, thus making GOP-SF
roughly equivalent to GOP-SF-Norm.
E. Computational complexity
We can compute the complexity of calculating GOP-SF
(Eq. 12) with the help of dynamic programming. First we
note that we only need to evaluate the CTC loss twice, once
for the first term L(LC,OT
1 ) and once for the second term
L(LSDI,OT
1 ) in Eq. 12. The graph used to compute L(LC,OT
1 )
(Figure 3, top) contains |G| = 2|LC|+1 nodes (the transcription labels interleaved by ϕs). The graph used to compute
L(LSDI,OT
1 ) (Figure 3, bottom) contains |G| = 2|LC| + |V |
nodes, where V is the set of output symbols from the neural
network. The complexity of running dynamic programming on
these graphs is therefore O(T ×|G|) = O(T ×(|LC|+|V |)),
that is, it is linear both in the length of the utterances, in the
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 7
number of symbols in the canonical transcription augmented
by the number of output symbols, which is usually a constant.
For GOP-SF-Norm, the only overhead is the summation in
Eq 13 because the normalized forward variables α̂ are already
computed.
F. Segmentation-free GOP features
Similarly to the approaches in [21], [32], [33], the GOP-SF
of the ith phoneme in a canonical sequence can be expanded
into a feature vector using LPP (log posterior probability) and
LPR (log posterior ratio vector):
FGOP-SF(li) = {LPP,LPR(li)} (14)
where LPP = logp(LC|OT
1 ) = −LCTC(LC) follows the
definition of CTC’s log-posterior in Eq.(4) and LPR is a vector
of length |C| + 1:
LPR(i) =

log

p(LC|OT
1 )
p(L|OT
1 )

with L ∈ LSD(li)

(15)
where LSD(li) has already been defined in Section III-C. We
do not include insertion errors here because that would result
in infinite length for the feature vectors. Similar to GOP-SFNorm, we append the expected activation count in Eq. 13 as
an extra dimension of the feature vector which forms:
FGOP-SF-Norm(li) = {LPP,LPR(li),Occ(i)}. (16)
The feature vectors can be fed into multi-dimensional MDD
classification models that take the feature vectors as input.
G. Measures of Peakiness
In Section II-B, we have introduced the peaky behavior
of CTC-trained models both with respect to time (POT) and
symbols (POS). Our two methods were introduced in part
to cope with this behavior. Due to the definitions given in
Section III, we expect the performance of GOP-SA (selfalignment) to increase with both POT and POS, in contrast
to standard methods. We also expect GOP-SF (segmentationfree) to be less affected by POT and POS. In order to test the
effect of POS and POT on the different GOP methods in a
reproducible way, we define objective measures of POS and
POT.
POT, can be measured by the frequency of activation of the
blank symbols (ϕ). To estimate this, we use the average of
post-softmax activation for ϕ over the speech dataset D for a
given model M:
BC(D,M) =
1
N
N X
n=1
1
Tn
X
t∈Tn
p(ϕ|On
t ;M), (17)
where the dataset D contains N utterances, each represented
by a sequence of speech features On
of length Tn and the corresponding transcriptions Ln
. We call this the blank coverage
(BC). High BC corresponds to very short activations of the
non-blank output symbols and, therefore, to high peakiness in
time (POT).
For POS, we borrow the idea from [30] and define the
average conditional entropy (ConEn) as:
ConEn(D,M) =
1
N
N X
n=1
H(p(π|Ln
,On
)), (18)
where
H(p(π|Ln
,On
)) =
= −
X
π∈{u∈V |O|:B(u)=L}
p(π|Ln
,On
)logp(π|Ln
,On
). (19)
Low ConEn corresponds to high peakiness in symbols (POS).
## IV EXPERIMENTAL SETTINGS
A. Data
We perform our experiments on two datasets including child
speech: CMU Kids and speechocean762.
CMU Kids [34] comprises 9.1 hours of speech from children in the range 6–11 years old, and a total of 5180 sentences
spoken by 65 females and 24 males. In CMU Kids, each
utterance is equipped with a phonetic transcription and utterances including mispronunciations are marked. We determine
the distribution of pronunciation errors by comparing5
the
phonetic transcription with the canonical pronunciation from
the CMU pronunciation dictionary provided by Kaldi [35].
There are 150899 labeled phonemes in total, among which,
90.2% are correct; 3.2% are substitution errors; 2.3% are
deletion errors; and 4.3% are insertion errors. We refer to
MDD experiments that use the labels obtained this way as
“real errors”.
Following [36], we also create an alternative MDD task
based on the CMU Kids data, where we systematically change
each phoneme in the canonical pronunciation to any other
phoneme. We call this task “simulated errors” because we
pretend that the recorded speech was incorrectly pronounced.
Finally, the speechocean762 dataset [37] includes 5000
utterances read by 250 gender-balanced L2 English learners,
half of which are children in the range from 5 years- to
15 years-old. The dataset is collected specifically for pronunciation assessment tasks with rich annotations at different
linguistic levels. We focus on the phoneme-level where each
phoneme is labeled as one of the three categories: 0 (mispronounced), 1 (strongly accented) and 2 (correctly spoken).
speechocean762 provides canonical phone-level transcriptions
using the same phone inventory as the CMU pronunciation
dictionary that we used for CMU Kids. We were also interested
in comparing the results for children and adults. In order
to do this we split the speechocean762 test set according
to the reported age of the speaker. The children (age 6–11)
account for 38.4% of the utterances whereas adults (age 12–
43) account for 61.6%. Note however, that the results on
speechocean762 children are not comparable to those in CMU
Kids because the two data sets define different tasks (ternary
vs binary assessment).
5The comparing procedure is performed only for those utterances that are
marked as problematic. For the correct utterances, on the other hand, we
simply label all the canonical phonemes as correct.
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 8
B. Baseline segmentation model
The external segmentation used for all the experiments in
this study is based on the same baseline segmentation model.
The model is a context-independent GMM-HMM model that is
trained using the standard Kaldi recipe “train-mono.sh” on the
100 hours training set “train-100-clean” from Librispeech [38].
Since the output symbols of the baseline segmentation model
match the symbols of the canonical sequences, we used the
Kaldi’s command “gmm-align” for obtaining the segmentation
for both CMU Kids and speechocean762 for further experiments.
## C Pre-training and fine-tuning of the acoustic models
The acoustic models “DNN” (CMU Kids baseline) and
“TDNN” (speechocean762 baseline) are trained from scratch
using the Kaldi recipes6
. All the other models are based
on the wav2vec2.0-large-xlsr-53 [39] which is available on
Huggingface7
and have a transformer-based structure [8]. This
model, with the feature_extractor frozen, was finetuned on the Librispeech “train-100-clean” set according to
the corresponding loss functions:
• CE: cross entropy
• CTC: connectionist temporal classification
• EnCTC: CTC with entropy regularization [30]
• EsCTC: CTC with equal space constraint [30]
Fine-tuning was performed for at most 10 epochs or by early
stopping based on the validation loss, with learning rate 1e-4
and batch size 32. The same model was used for the CMU
Kids and speechocean762 experiments.
D. Evaluation
The evaluation on CMU Kids is based on the canonical
transcription for each utterance. Each phone in the canonical
transcription is marked as mispronounced if it deviates from
the phonetic transcription from the data set. Then we compute the receiver operative characteristic curve (AUC-ROC)
according to various GOP scores, where correctly detecting
a mispronunciation is a “true positive”. We opt for this
threshold-free criterion because we want a robust method
to evaluate the performance of our GOP definitions that is
not dependent on a specific threshold chosen to perform the
MDD task. This also allows us to use all the data for testing
because we do not need to set aside a training set for threshold
optimization. Moreover, AUC-ROC is considered to be robust
to label bias which is the case for our data. We compute AUCROC for each phoneme category and report the arithmetic
mean.
For the MDD-oriented dataset speechocean762 the task
is to predict three different classes. In this case, we train
and evaluate several typical MDD models using the standard
training and test splits. We follow the recommendations of
using Pearson Correlation Coefficient (PCC) from the dataset’s
paper [37] as MDD performance metric.
6./egs/librispeech/s5/local/{nnet2/run 5a clean 100, nnet3/run tdnn}.sh
7https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
TABLE I
MISPRONUNCIATION DETECTION ON CMU KIDS
Method AUC (95% confidence intervals)
Simulated errors Real errors
GOP-DNN-Avg [36] 0.824 (± 1.6E-3) 0.796 (± 3.8E-3)
GOP-CE-Avg 0.967 (± 7.2E-4) 0.851 (± 3.0E-3)
GOP-CTC-SA 0.988 (± 4.3E-4) 0.905 (± 2.1E-3)
GOP-CTC-SF-S 0.989 (± 4.1E-4) 0.891 (± 2.4E-3)
GOP-CTC-SF-SD 0.986 (± 4.7E-4) 0.914 (± 2.0E-3)
GOP-CTC-SF-SDI 0.938 (± 9.9E-4) 0.859 (± 2.9E-3)
To give a reference on the quality of the different acoustic
models in this study, we also report on phoneme recognition
results measured with phoneme error rate (PER). Beam-search
decoding without language models is used for recognition.
E. Experiments
The set of experiments we include in this paper aims at
answering the following questions:
1) Which GOP definition results in the best MDD performance? In this case we test GOP-X-Avg, GOP-X-SA, and
the different variants of GOP-X-SF (GOP-X-SF-SDI, GOPX-SF-SD, GOP-X-SF-S) where “X” corresponds to acoustic
models trained with different loss functions and architectures
as explained in Sec.IV-C.
2) How are the results dependent on the peakiness of the
ASR models used for the assessment? In this case we test
the different GOP definitions in combination with models that
are trained with loss functions that are specifically defined
to control peakiness: CTC, EnCTC, EsCTC and CE. We use
our best method GOP-SF-SD as segmentation-free version in
this experiment and due to numerical problem with CE-trained
models, we use the “numerical” implementation as discussed
in Sec.III-B.
3) How do the results for GOP-SF depend on the length of
the context around the target phoneme? In this case we create
utterances by cropping the original recordings and gradually
increasing the context around the target phoneme. Each time
we add one phoneme left and one right and we extract the
corresponding utterance with the help of forced alignment.
We go from no context to context length 7 (both right and
left) and we compare to the results obtained with the original
utterances (full context).
4) How do our results compare to the state-of-the-art
in MDD? In order to compare our results to the state-ofthe-art, we tested our best method (GOP-X-SF-SD) on the
speechocean762 dataset, and we also include FGOP, that is,
segmentation-free GOP features. Moreover, we compare our
new implementations that solve the numerical issues (denoted
as -numerical) and the normalized versions of both the scalar
GOP and the GOP features. Because the focus is on comparing
between different GOP and GOP-features definitions, we keep
the models for MDD simple in our state-of-the-art comparison.
In particular, we limit our tests to polynomial regression,
support vector regression (SVR) and GOPT as in [40]. The
PCC is computed between the model’s output value and the
true label. For the sake of fair comparison, we preserve
all the details for evaluations as the baseline papers, e.g,
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 9
TABLE II
PHONE RECOGNITION AND PRONUNCIATION ASSESSMENT RESULTS VS PEAKINESS.
Peakiness and Phone Recognition GOP Methods (AUC, 95% confidence intervals)
Models BC (%) ConEn PER (%) GOP-X-Avg GOP-X-SA GOP-X-SF
CE 42.22 22.9886 13.25 0.860 (±2.91E-3) [14] 0.885 (±2.50E-3) 0.870 (±2.75E-3)
EnCTC-0.20 79.10 37.8672 11.63 0.842 (±3.19E-3) 0.860 (±2.91E-3) 0.913 (±2.01E-3)
EnCTC-0.15 79.55 35.9145 11.64 0.841 (±3.20E-3) 0.861 (±2.81E-3) 0.915 (±1.97E-3)
EnCTC-0.10 80.11 34.9024 11.47 0.841 (±3.20E-3) 0.864 (±2.84E-3) 0.914 (±1.99E-3)
EsCTC 88.52 14.2370 21.06 0.580 (±6.00E-3) 0.884 (±2.52E-3) 0.896 (±2.31E-3)
CTC 88.62 2.8287 11.46 0.824 (±3.46E-3) 0.909 (±2.08E-3) 0.914 (±1.99E-3)
whether to round the output before calculating the PCC; using
polynomial order two for polynomial regression; applying the
Radial Basis Function (RBF) kernel for the SVR models etc.
Due to relatively high variance of the GOPT model, same as
in [40], we run the training of GOPT five times with random
initialization, selecting the best model for each run and then
averaging the results. Following the same idea in [36] to reduce
the variance of the GOPT model during training, we limit the
gradients to flow from phoneme-only loss instead of from all
multi-aspects losses.
## V RESULTS
A. Method comparison
Table I shows the results comparing the different methods
on the CMU Kids data both for simulated and real errors. AUC
values are reported together with 95% confidence intervals
computed according to [41]. The methods based on models
trained with CTC outperform both methods based on models
trained with CE (GOP-DNN-Avg and GOP-CE-Avg). GOPCTC-SF-S, that is, the segmentation-free method restricted to
substitution errors is significantly better on the simulated errors
whereas GOP-CTC-SF-SD, that also allows deletion errors is
significantly better on the real errors. This can be explained
by the fact that simulated errors are produced exclusively by
substitution, but real errors may include deletions as well.
The standard evaluation criterion used here only considers
phones in the canonical transcription. This means that insertion
errors only have an effect on the evaluation if they modify the
pronunciation of the neighbouring canonical phones. This does
not allow to show the full potential of GOP-CTC-SF-SDI, that
in our tests performs worse than GOP-CTC-SF-SD. However,
defining an ad-hoc evaluation criterion is outside the scope of
this study.
B. Performance versus peakiness
Table II shows the results of testing the effect of peakiness
of the acoustic models on the real errors of the CMU Kids. All
the models were trained up to checkpoint 8000. The left part
of the table reports blank coverage (BC) as a measure of POT
and conditional entropy (ConEn) as a measure of POS, as well
as phone error rates. The right part of the table reports MDD
results. The models were ordered in the table in increasing
order of peakiness over time (POT).
Standard CTC results in the highest BC and lowest ConEn,
which means that CTC is the peakiest model both with respect
to POT and POS. It is also the best phone recognizer with the
lowest PER among the models.
For the EnCTC models, POS decreases as the weight β of
the entropy term in the loss function increases. In our tests we
varied β in the range [0.10,0.15,0.20]. This is not surprising
because the entropy term is similar to ConEn. More interesting
is the fact that POT also decreases with β as shown by BC. The
EnCTC performance in phone recognition seems to degrade
with the peakiness of the models.
The model trained with EsCTC has the highest PER, a
similar POT compared to CTC but a much lower POS (higher
ConEn). This is in line with the experiments in [30], because
the model makes the strong assumption that the labels should
be uniformly distributed over time which is not always true
for speech.
As we expected, the model trained with CE has the lowest
BC because it is trained frame-wise according to baseline segmentation. Surprisingly, the ConEn of the CE-trained model
is still lower than EnCTC implying that the distribution of
possible alignment paths are more concentrated for CE-trained
model.
The experimental results for MDD are shown in the right
side of Table II in terms of AUC and 95% confidence intervals.
When using the standard GOP definition (GOP-X-Avg), the
CE model performs the best. This is expected because GOPX-Avg uses the same baseline segmentation that was used
to train the CE model. The CTC and EnCTC models are
comparable. We believe that this is because, although the
models have different degrees of peakiness, their activation is
still within the segments defined by the baseline segmentation.
On the contrary, the AUC of GOP-EsCTC-Avg is particularly low, most probably because the model activations do
not necessarily follow standard speech segmentation. It is
therefore possible that GOP is evaluated, in this case, based on
activations other than the target phoneme. This interpretation
is also supported by the fact that the use of the self-aligned
definition (GOP-EsCTC-SA) gives results that are comparable
with the model trained with CE. However, GOP-X-Avg is
clearly outperformed by the proposed GOP-X-SA and GOPX-SF methods that will be analyzed next.
In general, the proposed GOP-X-SA is always superior to
GOP-X-Avg, suggesting that this method is able to focus on
the regions corresponding to the relevant model activations. It
is interesting to note that this phenomenon can be observed
even for the models trained with CE, for which the mismatch
between model activations and baseline segmentation should
be minimal. Among the different models, CTC training perJOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 10
0 2 4 6 8 10 12 14 full
contextlength
0.990
0.992
0.994
0.996
0.998
1.000
AUC
AUCvscontextlength(simulatederrors)
0 2 4 6 8 10 12 14 full
contextlength
0.8225
0.8250
0.8275
0.8300
0.8325
0.8350
0.8375
0.8400
AUC
AUCvscontextlength(realerrors)
Fig. 4. AUC versus context length for simulated and real errors on the CMU Kids data. The shaded area shows 95% confidence intervals computed with
[41].
forms the best when using GOP-X-SA. Our interpretation is
that the self-aligned definition of GOP (GOP-X-SA) prefers
to work with peaky models (both POT and POS) such as the
model trained with CTC. Comparing to the standard definition
(GOP-X-Avg), this method makes it possible to take advantage
of the highly-discriminative model which is often good for
recognition. This interpretation is also supported by the slight
degradation in AUC results compared to standard CTC for
the EnCTC models which are less peaky and less performant
in terms of PER. Exceptionally, the behavior of the model
trained with EsCTC has by far the worst phone recognition
performance. However, the model’s peakiness (with the second
highest BC and second lowest ConEn) could still make it
comparable with CE in MDD performance when combined
with GOP-X-SA.
Finally, the last column of the table reports the results with
the proposed segmentation-free method (GOP-X-SF). This
method obtains the best overall performance for MDD by a
significant margin. Furthermore, according to the confidence
intervals it is not possible to determine which acoustic model
performs the best with this method because the results obtained
with CTC and EnCTC are not statistically different. This is
in accordance with our hypothesis that GOP-X-SF would be
less affected by POS of the acoustic models, compared to
methods that rely on segmentation. We can also observe that,
by considering all possible segmentations, GOP-X-SF is able
to extract much more useful information for MDD not only
compared to the external segmentation (GOP-X-Avg) but also
compared to an alignment that is matched to the activations
of the acoustic model (GOP-X-SA).
## C GOP-SF and context length
By definition, the segmentation-free GOP method that we
have proposed (GOP-SF) is computed considering contributions from the entire utterance, even if those contributions are
weighted by how likely it is that each part of the utterance
belongs to the target phone. A reasonable question to ask is
how the length of left context (LL) and right context (LR)
affect the pronunciation assessment results.
Figure 4 displays the AUC results on CMU Kids where we
have varied the number of phones in LL and LR from 0 to 7
resulting in a total context length from 0 to 14 in increments
of 2. We also report results with the original utterances (full
context). In this case, the length of the context depends on the
specific utterance, but it is always greater than 14. The left
plots simulated errors, whereas the right plots real errors.
For simulated errors, the AUC values for different context
lengths remain at a high level, and the impact of context length
can be neglected. The results show that GOP-CTC-SF is robust
to different context length in the ideal case.
Also for real errors, the dependency on context length is
usually under the variability described by the 95% confidence
intervals. The only exception is when we reduce the context
to zero, which corresponds to the self-aligned GOP definition
(GOP-SA). This confirms the superiority of the segmentationfree method that is able to take advantage of the context to
assess the pronunciation of the target phoneme. Using context
lengths between 8 and 14 is visually indistinguishable from
using the full context length. This suggests that the information
needed to assess the pronunciation of the target phoneme is
relatively local in the utterance.
D. Comparison to state-of-the-art
In order to compare the performance of our best
method (GOP-X-SF-SD) with respect to the state-of-the-art,
we report results on the speechocean762 dataset in Table III.
The table compares different scalar definitions of GOP with
a simple polynomial regression MDD model. It also includes
results with GOP feature vectors in combination with SVR or
GOPT MDD models.
Similarly to the previous experiments on CMU Kids, we
find that the methods relying on models trained with the CE
loss are less performant than those trained with the CTC
loss. Results for the method “GOP-CE-SF-SD” are missing
due to numerical problems. These were solved in the new
implementation (“GOP-CE-SF-SD-numerical”). The results
are further improved using the definition that normalizes GOP
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 11
TABLE III
SPEECHOCEAN762, REPORTED WITH PEARSON CORRELATION
COEFFICIENTS
Features MDD model PCC
GOP-TDNN [14] poly. reg. 0.361±0.008
GOP-CE-SF-SD poly. reg. N/A
GOP-CE-SF-SD-numerical poly. reg. 0.314±0.008
GOP-CE-SF-SD-Norm poly. reg. 0.332±0.008
GOP-CTC-SF-SD [14] poly. reg. 0.433±0.007
GOP-CTC-SF-SD-numerical poly. reg. 0.450±0.006
GOP-CTC-SF-SD-Norm poly. reg. 0.449±0.006
FGOP-TDNN [37] SVR 0.441±0.007
FGOP-CTC-SF [14] SVR 0.568±0.006
FGOP-CTC-SF-numerical SVR 0.580±0.006
FGOP-CTC-SF-Norm SVR 0.581±0.006
FGOP-TDNN [40] GOPT 0.605±0.002
FGOP-CTC-SF [14] GOPT 0.618±0.002
FGOP-CTC-SF-numerical GOPT 0.646±0.002
FGOP-CTC-SF-Norm GOPT 0.648±0.002
by the estimated length of the target segment (“GOP-CE-SFSD-Norm”).
In all cases, when combined with a CTC-trained model,
our segmentation-free methods (GOP-CTC-SF, and FGOPCTC-SF) outperform the baseline methods that use traditional GOP with the Time-Delayed-Neural-Network (TDNN)
acoustic models by a significant margin. We also see improvements when using the implementation which solves the
numerical issues compared to our previous results. As we
expected, the extra normalization steps for the methods are
less significant for CTC-trained acoustic models than CEtrained ones. The best combination of our method is “FGOPCTC-SF-Norm” which has 5.01% relatively higher PCC than
previously reported. We also tested this method on the split
of the speechocean762 test set that separates children (age
≤ 11) from adults (age > 11). The corresponding PCCs are
0.635±0.005 (children) and 0.651±0.004 (adults). The slight
gap between child and adult result may be reduced by adapting
the acoustic models to child speech.
At the time of writing and to the best of our knowledge, the
highest phoneme-level PCC obtained for the speechocean762
dataset is 0.693 [42]. The relatively limited PCC improvements
of this model with respect to ours are, however, obtained at
the cost of higher computational complexity, heavier feature
engineering and more sophisticated loss functions that take
all the linguistic aspects into account. For this reason, we
believe that our method is still preferable for practical realworld applications.
Finally, we could not find results on pure end-to-end models
for MDD on the speechocean762 data. This is probably due to
the limited amount of training data which prevents the proper
training of those models.
## VI CONCLUSIONS
In this work, we propose improvements to a framework
for pronunciation assessment that we recently introduced in
[14]. We propose two methods with the goal of allowing the
use of goodness of pronunciation (GOP) features extracted
from modern high-performance automatic speech recognition
models. The main idea of our methods is to release the
assumption that the GOP evaluation should be performed on a
predetermined segment of speech corresponding to the target
phone. We do this either by adjusting the segment under test to
the activations of the acoustic models (self-aligned GOP, GOPSA) or by proposing a definition of GOP that is independent
of the segmentation of test utterance (segmentation-free GOP,
GOP-SF).
Our theoretical derivations make the underlying assumptions for GOP-SF explicit. We also provide experimental
results exploring a number of properties of our methods.
GOP-SA is always better than traditional GOP regardless
of the characteristics of the acoustic model used for the
evaluation. GOP-SF obtains the overall best results for both
the CMU Kids and for the speechocean762 material. On the
speechocean762 data we obtain state-of-the-art results keeping
the MDD model constant. We also show how the peakiness of
the acoustic models affects the MDD results for the standard
GOP definition and for the two proposed methods. Finally we
show that the performance of GOP-SF, that considers the full
utterance to assess the pronunciation of the target phone, is
not affected by the length of the context.
We believe that the proposed methods are potentially very
appealing for phoneme-level pronunciation assessment, both
because of their performance but also for their simple implementation and very low computational cost.
## REFERENCES
[1] S. M. Witt and S. J. Young, “Phone-level pronunciation
scoring and assessment for interactive language learning,” Spe. Comm., vol. 30, no. 2-3, pp. 95–108, 2000.
[2] W.-K. Leung, X. Liu, and H. Meng, “CNN-RNNCTC based end-to-end mispronunciation detection and
diagnosis,” in IEEE ICASSP, 2019, pp. 8132–8136.
[3] K. Fu, J. Lin, D. Ke, Y. Xie, J. Zhang, and B. Lin,
“A full text-dependent end to end mispronunciation
detection and diagnosis with easy data augmentation
techniques,” ArXiv, vol. abs/2104.08428, 2021.
[4] Y. Feng, G. Fu, Q. Chen, and K. Chen, “SED-MDD:
Towards sentence dependent end-to-end mispronunciation detection and diagnosis,” in IEEE ICASSP, 2020,
pp. 3492–3496.
[5] L. Zhang et al., “End-to-end automatic pronunciation
error detection based on improved hybrid ctc/attention
architecture,” Sensors, vol. 20, no. 7, 2020.
[6] Y. Getman et al., “wav2vec2-based speech rating system
for children with speech sound disorder,” in Interspeech,
2022, pp. 3618–3622.
[7] X. Xu, Y. Kang, S. Cao, B. Lin, and L. Ma, “Explore
wav2vec 2.0 for mispronunciation detection.,” in Interspeech, 2021, pp. 4428–4432.
[8] A. Baevski, H. Zhou, A. Mohamed, and M. Auli,
“wav2vec 2.0: A framework for self-supervised learning
of speech representations,” in NeurIPS, 2020.
[9] E. Kim, J.-J. Jeon, H. Seo, and H. Kim, “Automatic Pronunciation Assessment using Self-Supervised
Speech Representation Learning,” in Interspeech, 2022,
pp. 1411–1415.
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 12
[10] Y. Shen, Q. Liu, Z. Fan, J. Liu, and A. Wumaier, “Selfsupervised pre-trained speech representation based endto-end mispronunciation detection and diagnosis of
mandarin,” IEEE Access, vol. 10, pp. 106451–106462,
2022.
[11] W.-N. Hsu, B. Bolte, Y.-H. Tsai, K. Lakhotia, R.
Salakhutdinov, and A. Mohamed, “HuBERT: Selfsupervised speech representation learning by masked
prediction of hidden units,” IEEE/ACM Trans. Audio,
Speech, Language Process., pp. 3451–3460, 2021.
[12] H. Liu, M. Shi, and Y. Wang, “Zero-Shot Automatic Pronunciation Assessment,” in Interspeech, 2023,
pp. 1009–1013.
[13] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist temporal classification: Labelling
unsegmented sequence data with recurrent neural networks,” in ICML, 2006, pp. 369–376.
[14] X. Cao, Z. Fan, T. Svendsen, and G. Salvi, “A framework for phoneme-level pronunciation assessment using
CTC,” in Interspeech 2024, 2024, pp. 302–306.
[15] Y. Song, W. Liang, and R. Liu, “Lattice-based GOP in
automatic pronunciation evaluation,” in ICCAE, vol. 3,
2010, pp. 598–602.
[16] D. Luo, Y. Qiao, N. Minematsu, Y. Yamauchi, and
K. Hirose, “Analysis and utilization of MLLR speaker
adaptation technique for learners’ pronunciation evaluation,” in Interspeech, 2009.
[17] A. M. Harrison, W.-K. Lo, X.-j. Qian, and H.
Meng, “Implementation of an extended recognition
network for mispronunciation detection and diagnosis
in computer-assisted pronunciation training,” in SLaTE,
2009, pp. 45–48.
[18] S. Dudy, S. Bedrick, M. Asgari, and A. Kain, “Automatic analysis of pronunciations for children with
speech sound disorders,” Computer Speech & Language, vol. 50, pp. 62–84, 2018.
[19] S. Kanters, C. Cucchiarini, and H. Strik, “The goodness of pronunciation algorithm: a detailed performance
study,” in SLaTE, 2009, pp. 49–52.
[20] W. Hu, Y. Qian, and F. K. Soong, “A new DNN-based
high quality pronunciation evaluation for computeraided language learning (CALL),” in Interspeech, 2013,
pp. 1886–1890.
[21] W. Hu, Y. Qian, F. Soong, and Y. Wang, “Improved
mispronunciation detection with deep neural network
trained acoustic models and transfer learning based
logistic regression classifiers,” Spe. Comm., vol. 67,
2015.
[22] S. Sudhakara, M. K. Ramanathi, C. Yarra, and
P. K. Ghosh, “An Improved Goodness of Pronunciation (GoP) Measure for Pronunciation Evaluation
with DNN-HMM System Considering HMM Transition
Probabilities,” in Interspeech, 2019, pp. 954–958.
[23] J. Shi, N. Huo, and Q. Jin, “Context-Aware Goodness
of Pronunciation for Computer-Assisted Pronunciation
Training,” in Interspeech, 2020, pp. 3057–3061.
[24] V. C M et al., “The impact of forced-alignment errors
on automatic pronunciation evaluation,” in Interspeech,
2021, pp. 1922–1926.
[25] T. Mahr, V. Berisha, K. Kawabata, J. Liss, and K.
Hustad, “Performance of forced-alignment algorithms
on children’s speech,” JSHLR, vol. 64, pp. 1–10, 2021.
[26] W. Hu, Y. Qian, and F. K. Soong, “An improved
DNN-based approach to mispronunciation detection and
diagnosis of L2 learners’ speech,” in SLaTE, 2015,
pp. 71–76.
[27] L. Wai Kit and F. Soong, “Generalized posterior probability for minimum error verification of recognized
sentences,” in IEEE ICASSP, 2005, pp. 85–88.
[28] Z. Zhang, Y. Wang, and J. Yang, “Text-conditioned
transformer for automatic pronunciation error detection,” Spe. Comm., vol. 130, no. C, pp. 55–63, 2021.
[29] A. Zeyer, R. Schlüter, and H. Ney, “Why does
CTC result in peaky behavior?” arXiv preprint
arXiv:2105.14849, 2021.
[30] H. Liu, S. Jin, and C. Zhang, “Connectionist temporal
classification with maximum entropy regularization,” in
NeurIPS, 2018.
[31] R. Huang et al., “Less peaky and more accurate CTC
forced alignment by label priors,” in IEEE ICASSP,
2024.
[32] J. Doremalen, C. Cucchiarini, and H. Strik, “Using nonnative error patterns to improve pronunciation verification,” in Interspeech, 2010, pp. 590–593.
[33] S. Wei, G. Hu, Y. Hu, and R.-H. Wang, “A new method
for mispronunciation detection using support vector
machine based on pronunciation space models,” Spe.
Comm., vol. 51, no. 10, pp. 896–905, 2009.
[34] M. Eskenazi, J. Mostow, and D. Graff, The CMU Kids
Corpus LDC97S63. Web Download. Philadelphia: Linguistic Data Consortium, accessed 03/07/2023, 1997.
[35] D. Povey et al., “The Kaldi speech recognition toolkit,”
in ASRU, 2011.
[36] X. Cao, Z. Fan, T. Svendsen, and G. Salvi, “An Analysis
of Goodness of Pronunciation for Child Speech,” in
Interspeech, 2023, pp. 4613–4617.
[37] J. Zhang et al., “speechocean762: An open-source nonnative english speech corpus for pronunciation assessment,” in Interspeech, 2021, pp. 3710–3714.
[38] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur,
“Librispeech: An ASR corpus based on public domain
audio books,” in IEEE ICASSP, 2015, pp. 5206–5210.
[39] A. Conneau, A. Baevski, R. Collobert, A. Mohamed,
and M. Auli, “Unsupervised Cross-Lingual Representation Learning for Speech Recognition,” in Interspeech,
2021, pp. 2426–2430.
[40] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass,
“Transformer-based multi-aspect multi-granularity nonnative english speaker pronunciation assessment,” in
IEEE ICASSP, 2022, pp. 7262–7266.
[41] J. Hanley and B. McNeil, “The meaning and use of
the area under a receiver operating characteristic (ROC)
curve.,” Radiology, vol. 148, pp. 29–36, 1982.
JOURNAL OF L A TEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 13
[42] F.-A. Chao, T.-H. Lo, T.-I. Wu, Y.-T. Sung, and B. Chen,
“A hierarchical context-aware modeling approach for
multi-aspect and multi-granular pronunciation assessment,” in Interspeech, 2023, pp. 974–978.
Xinwei Cao is a PhD candidate in Signal Processing and Machine Learning at NTNU, Norway,
researching speech assessment using state-of-theart ASR and TTS systems. He previously worked
at Cerence Inc., Germany (2019-2022), developing
language models for automotive ASR, and at IFlytek
Inc., China (2018-2019), building chat-bot systems.
He received his M.S. in Computer Science from
Karlsruhe Institute of Technology, Germany (2018),
and B.S. from Zhejiang University, China (2012).
His research interests include speech processing,
acoustic and language modeling, neural machine translation and generative
AI.
Zijian Fan is a PhD candidate in Signal Processing
and Machine Learning at NTNU, Norway, researching Child speech ASR sytsems. He received his
M.S. in Information and Network Engineering from
KTH Royal Institute of Technology, Sweden (2020)
and B.S. in Electrical Engineering from Dalian University of Technology, China (2018). His research
interests include speech augmentation, end-to-end
models, and generative models.
Torbjørn Svendsen is a Professor at the Department
of Electronic Systems. Professor Svendsen holds a
MScEE, and a PhD both from the NTNU. He is
an ISCA Fellow and IEEE Life Senior Member. He
received his dr.ing. (PhD) degree in 1985 from the
then Norwegian Institute of Technology (NTH). He
has published over 110 scientific papers and has supervised 18 PhD graduates and more than 100 Masters students. His research activities have spanned
numerous application areas of speech technology,
including speech recognition; speech compression;
speaker identification; spoken language identification and speech synthesis.
He is an ISCA Fellow, and a former Vice President of ISCA. He is an elected
member to the Norwegian Academy of Technical Sciences.
Giampiero Salvi (Senior Member, IEEE) is a Full
Professor at the Department of Electronic Systems
at the Norwegian University of Science and Technology (NTNU), Trondheim, Norway. Prof. Salvi
received the MSc degree in Electronic Engineering
from Università la Sapienza, Rome, Italy and the
PhD degree in Computer Science from KTH. He
was a post-doctoral fellow at the Institute of Systems
and Robotics, Lisbon, Portugal and an Associate
Professor at KTH Royal Institute of Technology,
Stockholm, Sweden. He was a co-founder of the
company SynFace AB, active between 2006 and 2016. His main interests are
machine learning, speech technology, and cognitive systems and has authored
and co-authored more than 110 papers in these areas. Prof. Salvi is a senior
member of IEEE Signal Processing Society (SPS) and a member of the
International Speech Communication Association (ISCA).
