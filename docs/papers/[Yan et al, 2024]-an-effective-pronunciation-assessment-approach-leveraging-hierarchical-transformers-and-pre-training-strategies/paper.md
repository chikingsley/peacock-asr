           An Effective Pronunciation Assessment Approach Leveraging
              Hierarchical Transformers and Pre-training Strategies
        Bi-Cheng Yan1*, Jiun-Ting Li1, Yi-Cheng Wang1, Hsin-Wei Wang1, Tien-Hong Lo1,
                Yung-Chang Hsu2, Wei-Cheng Chao3, Berlin Chen1*
                                    1
                                 National Taiwan Normal University, 2EZAI
                      3
                        Advanced Technology Laboratory, Chunghwa Telecom Co., Ltd.
                           {bicheng, berlin}@ntnu.edu.tw, weicheng@cht.com.tw


                          Abstract
                                                                                                    Reading-aloud Scenario

         Automatic pronunciation assessment (APA)                                                       We call it bear.
         manages to quantify a second language (L2)
                                                                                                        We call it bear.
         learner's pronunciation proficiency in a
         target language by providing fine-grained                                     Automatic Pronunciation Assessment Results
                                                                              Utterance level                  Word level              Phone level
         feedback with multiple pronunciation                                Aspects      Scores       Words    Aspects     Scores   Phones   Scores
         aspect scores at various linguistic levels.                                                            Accuracy      2        W       2.0
                                                                            Accuracy        1.6
         Most existing efforts on APA typically                                                         We       Stress       2
                                                                                                                                      IY0      2.0
                                                                                                                  Total       2
         parallelize the modeling process, namely                            Fluency        1.8                 Accuracy      2        K       2.0
         predicting multiple aspect scores across                                                       Call     Stress       2       AO0      1.8

         various linguistic levels simultaneously.                         Completeness         2
                                                                                                                  Total
                                                                                                                Accuracy
                                                                                                                              2
                                                                                                                              2
                                                                                                                                       L       1.8
                                                                                                                                      IH0      2.0
         This inevitably makes both the hierarchy of                                                     It      Stress       2        T       2.0
         linguistic units and the relatedness among                          Prosody        1.8                   Total       2
                                                                                                                                       B       2.0
                                                                                                                Accuracy     1.2
         the pronunciation aspects sidelined.                                                          Bear      Stress       2       EH0      1.0
                                                                              Total         1.6
         Recognizing such a limitation, we in this                                                                Total      1.2       R       1.0

         paper first introduce HierTFR1, a hierarchal
         APA method that jointly models the                            Figure 1: A running example curated from the
         intrinsic structures of an utterance while                    speechocean762 dataset (Zhang et al., 2021)
         considering the relatedness among the                         illustrates the evaluation flow of an APA system
         pronunciation aspects. We also propose a                      in the reading-aloud scenario, which offers an L2
         correlation-aware regularizer to strengthen                   learner in-depth pronunciation feedback.
         the connection between the estimated
         scores and the human annotations.                        learners to practice pronunciation skills in a stress-
         Furthermore, novel pre-training strategies               free and self-directed learning manner (Eskenazi
         tailored for different linguistic levels are             2009; Evanini and Wang, 2013; Evanini et al., 2017;
         put forward so as to facilitate better model             Rogerson-Revell, 2021). As a crucial ingredient of
         initialization. An extensive set of empirical
                                                                  CAPT, automatic pronunciation assessment (APA)
         experiments        conducted      on      the
         speechocean762        benchmark       dataset
                                                                  aims to evaluate the extent of L2 learners’ oral
         suggest the feasibility and effectiveness of             proficiency and then provide fine-grained feedback
         our approach in relation to several                      on specific pronunciation aspects in response to a
         competitive baselines.                                   target language (Bannò et al., 2022; Chen and Li,
                                                                  2016; Kheir et al., 2023). A de-facto standard for
   1     Introduction                                             APA systems is typically instantiated with a
                                                                  “reading-aloud” scenario, where an L2 learner is
   With the rising trend of globalization, more and               presented with a text prompt and instructed to
   more people are willing or being demanded to learn             pronounce it correctly. To offer in-depth feedback
   foreign languages. This surging need calls for                 on learners’ pronunciation quality, recent efforts
   developing computer-assisted pronunciation                     have drawn attention to the notion of multi-aspect
   training (CAPT) systems, as they can offer tailored            and multi-granular pronunciation assessments,
   and informative feedback for L2 (second-language)              which normally devises a unified scoring model to

                                                                  1
   * Corresponding author.                                            https://github.com/bicheng1225/HierTFR

                                                             1737
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1737–1747
                              August 11-16, 2024 ©2024 Association for Computational Linguistics
jointly evaluate pronunciation proficiency at
various linguistic levels (i.e., phone-, word-, and
utterance-levels) with diverse aspects (e.g.,
accuracy, fluency, and completeness), as the
running example depicted in Figure 1. Methods
along this line of research usually follow a parallel
modeling paradigm, wherein the Transformer
network and its variants serve as the backbone
architecture to take as input a sequence of phone-
level pronunciation features and in turn predict
multiple aspect scores across various linguistic
levels simultaneously via a multi-task learning                   Figure 2: A correlation matrix derived from the
regime (Chao et al., 2022; Do et al., 2023a; Gong                 expert annotations of the training set. Each
et al., 2022).                                                    element in the matrix corresponds to the PCC
    Albeit models stemming from the parallel                      score of a pair of measured aspects.
modeling paradigm have demonstrated promising
                                                              so as to boost model initialization and hence reduce
results on a few APA tasks, they still suffer from at
                                                              the reliance on large amounts of supervised training
least two weaknesses. First, the language hierarchy
                                                              data. A comprehensive set of experimental results
of an utterance is nearly sidelined, which, for
                                                              reveal that the proposed model achieves significant
example, assumes that all phones within a word are
                                                              and consistent improvements over several strong
of equal importance and might insufficiently
                                                              baselines on the speechocean762 benchmark
capture the word-level structural traits. Second,
                                                              dataset (Zhang et al., 2021).
most of these methods largely overlook the
                                                                  In summary, the main contributions of our work
relatedness among the pronunciation aspects. As an
                                                              are at least three-fold: (1) we introduce HierTFR, a
illustration, we visualize the correlation matrix in
                                                              hierarchical neural model for APA, which is
Figure 2, which shows the Pearson Correlation
                                                              designed to hierarchically represent an L2 learner’s
Coefficients (PCCs) between any pair of expert
                                                              input utterance and effectively capture relatedness
annotated aspect scores on the training set. We can
                                                              within and across different linguistic levels; (2) we
observe that except for the aspects of utterance-
                                                              propose a correlation-aware regularizer for model
completeness and word-stress, the rest
                                                              training, which encourages prediction scores to
pronunciation aspects exhibit strong correlations
                                                              consider the relatedness among disparate aspects;
not only within the same linguistic level but also
                                                              and (3) extensive sets of experiments carried out on
across different linguistic levels2. Building on these
                                                              a public APA dataset confirm the utility of our
observations, we in this paper present a novel
                                                              proposed        pre-training     strategies,   which
language hierarchy-aware APA model, dubbed
                                                              considerably boosts the effectiveness of
HierTFR, which leverages a hierarchical
                                                              assessments across various linguistic levels.
Transformer-based architecture to jointly model
the intrinsic multi-level linguistic structures of an
                                                              2     Methodology
utterance while considering relatedness among
aspects within and across different linguistic levels.        2.1     Problem Formulation
To explicitly capture the relatedness within and
across different linguistic levels, an aspect attention       Given an input utterance U, consisting of a time
mechanism and a selective fusion module are                   sequence of audio signals X uttered by an L2
introduced. The proposed model is further                     learner, and a reference text prompt T with 𝑀
optimized with an effective correlation-aware                 words and 𝑁 phones, an APA model is trained to
regularizer, which encourages the correlations of             estimate the proficiency scores pertaining to
predicted aspect scores to match those of their               multiple pronunciation aspects at various linguistic
counterparts provided by human annotations.                   granularities. Let G = {𝑝, 𝑤, 𝑢} be a set of
Furthermore, distinct pre-training strategies                 linguistic granularities, where 𝑝, 𝑤, 𝑢 stands for the
tailored for three linguistic levels are put forward,         phone-, word-, and utterance-level linguistic units,

2
 Both the aspects of utterance completeness and word stress   of the assessments receiving the highest score (Do et al.,
suffer from label imbalance problems, with more than 90%      2023a).

                                                          1738
                                               Utterance Pooling Module
                                                                                                                                                                                                     Aspect Atten

                                                                                                                                                                                                                                                                                                                                           Utt-level                                           Utterance




                                                                                                                                                                                                    GATE GATE GATE GATE GATE
                                                                                                                                                                                                                                    MHCA MHCA MHCA MHCA MHCA
                                                                                                                                                                                                                                                                                                                                     Utt-level




                                                                                                                                                            Aggregation Layer
                                                                                                                                                                                                                                                                        Selective Fusion                                       Utt-level Regressor                                   Accuracy/Fluency/Completeness


                                                      We Call It Bear
                                                                                                                                                                                                                                                                                                                        Utt-level Regressor

                          Word 1 (We)                                                                               Utt-level Atten
                                                                                                                                                                                                                                                                          Mechanism                                Utt-level Regressor                                                      /Prosodic/Total
                                          W                                  …

                                                                                                                                                                                                                                                                                                                      Regressor
                                                                                                                                       %̅)! , %̅)" , %̅)#
                                                                                …
                                          IY
                                                                                …

                                                                                                                                                                                                                                                                                                                  Regressor                                                                      Score




                          Word 2 (Call)
                                          K                                X! , H! , H,
                                          AO
                                               Word-level Modeling
                                          L




      We Call It Bear
                                                                                             Pos & Emb
                                                               CLS                                                                                                                                                                                                                                             Aspect Atten


                                                                                                                                                                                                                                                                                                                                                                 Word-level
                                                                   …                                                                   …

                                                                                                                                                                                                                                                                                                                                MHCA
                         Word 3 (It)
                                                                                                                                                                                                                                                                                                                GATE

                                                                                                                                                                                                                                                                                                                                                                         Regressor
                                                              CLS




                                                                                                                                                                                Transformer Block
                                                                                                                                                                                                                                                                                                                                                                                                                            Word
                                          IH




                                                                                                                                                                                                                                       Transformer Block
                                                                                                                                                            …
                                                                                                                                       E,


                                                                                                                                                                                                                                                                                                                                                         Word-level
                                          T                     W
                                               We

                                                                                                                                                                                                                                                                                                                                                                 Regressor
                                                                                                                                                                                                                                                                                                               GATE             MHCA                                                                                 Accuracy/Stress/Total
                                                                IY                                                                                                                                                             …
                                                                                                                                                                                                                                                                           …




                                                                                                 Word-level Atten
                                                                  K


                                                                                                                                                                                                                                                                                                                                                                                Word-level
                                               Call

                                                                                                                                                                                                                                                                                                                                                         Regressor
                                                                                                                                                                                                                                                                                                                                                                                                                            Score
                         Word 4(Bear)
                                           B                  AO                                                                                            …
                                                                                                                                                                                                                                                                                                                GATE            MHCA
                                                                                                                                      …
                                                                   L                                                                                                                                                                                                    H-
                                          EH                                     …
                                                                                                                                        …
                                                …                …                 …
                                          R                                                                                                                  +
                                                                                                                                                            H,
                                                                  B
                                               Bear           EH                                                                      ( , , X,
                                                                                                                                      H
                                                                 R
                                                                               H! , X !




       Audio Signals X
                                               Phone-level Modeling




                                                                                           Pos & Emb
                                                      CLS




                                                                                                                                                                                                               Transformer Block               Transformer Block                           Transformer Block
                                                          …                                                                            …

                                                                                                                                                                                                                                                                                                                                                                                            Phone
                                                      CLS
                                                                                                                                                              …

                                                                                                                                      E!                                                                                                                                                                                                               Phone-level
                                                                                                                                                                                                                                                                       …
                                                      W
                                                                                                                                                                                                                                                                                                                                                       Regressor                        Accuracy Score
                                                                                                                                                                                                                                                                                                                                …


                                                                                            Cat & Proj
                                                      IY                     …                                                                                …
                                                       …                       ……                                                      …                                                                                                                                                                                        H!
                                                       R
                                                                                                                                                            H!+
                                                                                                                                      X!
                                                                        E "#$ , E %&' , E ()*


    Figure 3: An architecture overview of the proposed model, which consists of a phone-level modeling
    component, a word-level modeling component, and an utterance pooling module.
respectively. For each linguistic unit 𝑔 ∈ 𝐺 , the                                                                                                                              (GOP)-based features E%&' , as well as prosodic
APA model learns to predict a set of aspect scores                                                                                                                              features composed of duration E()* and energy
         ! !         !
A! = {𝑎" , 𝑎# , … , 𝑎$! }, where 𝑁! is the number of                                                                                                                            E+,! statistics (Witt and Young, 2000; Hu et al.,
pronunciation aspects of the linguistic unit 𝑔.                                                                                                                                 2015; Zhu et al., 2022; Shen et al., 2021) 3. All these
                                                                                                                                                                                features are then concatenated and subsequently
2.2                               Hierarchical Interactive Transformer                                                                                                          projected to from a sequence of acoustic features
                                  Architecture                                                                                                                                  X - . In the meantime, the phone-level text prompt
The overall architecture of our proposed APA                                                                                                                                    is mapped into an embedding sequence E- via a
model is schematically depicted in Figure 3, which                                                                                                                              phone and position embedding layer and then
consists of three ingredients: phone-level modeling,                                                                                                                            point-wisely added to X - for enriching the
word-level modeling, and utterance pooling                                                                                                                                      phonetic information of X - . The resulting
modules. After obtaining the representations of                                                                                                                                 representations H-. are prepend with five trainable
various pronunciation aspects, fully-connected                                                                                                                                  “[CLS]” embeddings and in turn fed into a phone-
neural layers is functioned as the regressors to                                                                                                                                level transformer to obtain the contextualized
collectively generate the corresponding aspect                                                                                                                                  representations H- (Vaswani et al., 2017):
score sequence for an input utterance.
                                                                                                                                                                                                                                   X ! = W ∙ [E"#$ ; E%&' ; E()* ] + 𝐛,                                                                                                                                                                      (1)
Phone-level Modeling. For an input utterance U,
various pronunciation features are extracted to                                                                                                                                                                                                                                H!+ = X ! + E! ,                                                                                                                                              (2)
portray the L2 learner’s pronunciation quality,                                                                                                                                                                                                                H   !
                                                                                                                                                                                                                                                                       = Transformer,-. 6H!+ 7,                                                                                                                                              (3)
which includes the goodness of pronunciation

3
  Further details on pronunciation feature extractions can be
found in Appendix A.

                                                                                                                                                                     1739
where W and 𝐛 are learnable parameters. To assess                      = /! = W4 ∙ H/ + 𝐛4 ,
                                                                       H                                     (8)
a sequence of phone-level aspect scores, H-                   = /'!
                                                              H                    /
                                                                      = σ @W*! ∙ C + 𝐛*! B ⨂ H ,  = /!       (9)
(excluding the first 5 embeddings) is forward
                                                                                 = /'! , C'5 7,
                                                                      H/! = MHCA6H                          (10)
propagated to the corresponding regressors. The
                       -
excluded embeddings H":0   are expected to convey         where H 9 1" are aspect-specific representations, and
the holistic pronunciation information and are            C1 = [H  9 1# , … , H
                                                                              9 1$% ] includes all aspect-specific
further fed into the subsequent selective fusion          representations. In MHCA, H         9 1*" is linearly
mechanism for use in utterance-level assessments.         projected to act as the query matrix, while C*2 is
Word-level Modeling. For the word-level                   linearly projected to form the key and value
assessments, a word-level attention pooling is used       matrixes. Additionally, the masking strategy
to produce a word representation vector from its          ensures that the output representation at a specific
corresponding phones, which can be implemented            position is only influenced by the other aspects of
as a multi-head attention layer followed by an            the word unit. Lastly, the aspect representations
average operation. The word-level input                   H1" are taken as input to the corresponding
representations H1.
                     can be obtained by applying          regressor to predict a score sequence for the 𝑗-th
the word-level attention to the phone-level               word-level pronunciation aspect.
representations X - and H- individually, followed         Utterance Pooling Module. For the utterance-
by a linear combination with the word-level textual       level assessments, utterance-level attention pooling
embeddings E1 . Next, H1  .
                             is prepend with five         is introduced to generate an utterance-level holistic
trainable “[CLS]” embeddings and fed into a               representation from the corresponding input
transformer to calculate the contextualized               representations, which can be effectively
representations H1 at word-level:                         implemented by attention pooling (Peng et al.,
                                                          2022). In more detail, the utterance-level
                X / = Atten0123 (X ! ),          (4)      representation 𝐡) can be obtained by feeding the
                = / = Atten0123 (H! ),
                H                                (5)      vector sequences X - , H- , and H1 into an
                 +        = / + E/ ,
                                                          utterance-level        attention   pooling      module
                H/ = X/ + H                      (6)
                                                          individually, followed by an aggregation operation:
            /                       + ).
        H       = Transformer0123 (H/            (7)
                          1                                            𝐡̅&" = Atten677 (X ! ),              (11)
Note here that H (excluding the first 5
embeddings) is utilized in the word-level                              𝐡̅&# = Atten677 (H! ),               (12)
                                           1
assessments while the excluded embeddings H":0                     𝐡̅&$ = Atten677 (H/ ),                   (13)
are fed into in subsequent selective fusion                   𝐡& = W& 6𝐡̅&" + 𝐡̅&# + 𝐡̅&$ 7 + 𝐛& ,          (14)
mechanism for use in the utterance-level
assessments.                                              where W) , 𝐛) are trainable parameters.

    After that, an aspect attention mechanism is              Next, a selective fusion mechanism is proposed
introduced to capture the relatedness among               to integrate contextualized representations across
disparate aspects (Do et al., 2023b; Ridley et al.,       multiple linguistic levels for the utterance-level
2021). This mechanism consists of two sub-layers:         pronunciation assessments (Xu et al., 2021).
a self-gating layer and a multi-head cross-attention      Specifically, for the estimation of 𝑗-th utterance-
layer. Specifically, for the 𝑗-th word-level aspect,      level aspect score, an aspect attention operation is
the relation-aware representations H    9 1*" are first   first performed on 𝐡) to a produce intermediate
                1
derived from H via a self-gating layer which aims         representation @𝐡)" . Note also that the gate values
                                                                           )            )                    )
to abstract away from redundant information while         for the phone (𝑔- " ), word (𝑔1" ) and utterance (𝑔) " )
considering the information gathered from other           granularities are used to control the extent to which
aspects. In addition, a multi-head cross-attention        these contextualized representations can flow into
(MHCA) process alongside a masking strategy is            the fused representation 𝐡)" :
employed to calculate aspect representations H1"               &
                                                                                           K &! L + 𝑏! B,
                                                              𝑔! ! = 𝜎 @𝐰!! ∙ J𝐡4! ; 𝐡4/ ; 𝐡                (15)
                                                                                                      !
from a collection of all relation-aware aspect                 &
                           9 1*# , … , H
                                       9 1*$% < . The         𝑔/! = 𝜎 @𝐰/! ∙ J𝐡4! ; 𝐡4/ ; K𝐡&! L + 𝑏/! B,   (16)
representations C*2 = ;H
                                                               &
following equations illustrate the operations of                                           K &! L + 𝑏& B,
                                                              𝑔& ! = 𝜎 @𝐰&! ∙ J𝐡4! ; 𝐡4/ ; 𝐡                (17)
                                                                                                      !
aspect attention:

                                                       1740
                &              &            &
   𝐡&! = 𝑔! ! ∙ 𝐡4! + 𝑔/! ∙ 𝐡4/ + 𝑔& ! ∙ K𝐡&! ,      (18)      Lakhotia et al., 2021). At lower linguistic levels, we
                                                               leverage       the        mask-predict       objective
where 𝐡3- and 𝐡31 are 𝑗-th representation vectors of           (Ghazvininejad et al., 2019) in the pre-training
H- and H1 ; and 𝐰-" , 𝐰1" , 𝐰)" , 𝑏-" , 𝑏1" , and 𝑏)"          stage. To this end, we first mask a portion of input
are trainable parameters. The fused representation             text prompt at phone- and word-levels. The
𝐡)" is then passed to the corresponding regressor              corresponding Transformers are then tasked on
to assess the proficiency score for a given                    recovering the masked tokens conditioning on the
utterance-level aspect.                                        unmasked prompt sequence and the associated
                                                               pronunciation representations (i.e., H-. , and H1  .
                                                                                                                    ).
2.3      Optimization                                          For the utterance level, we base the proposed pre-
Automatic Pronunciation Assessment Loss.                       training strategy on predicting the relatively high or
The loss for multi-aspect and multi-granular                   low accuracy scores for a pair of utterances.
pronunciation assessment, ℒ4'4 , is calculated as a            Namely, given any two utterances, the objective is
weighted sum of the mean square error (MSE)                    to predict whether the former has a higher, lower,
losses corresponding to different linguistic levels.           or the same accuracy score as the latter. Note here
               𝜆#          𝜆%          𝜆&                      that, utterance pairs are randomly selected from a
      ℒ!"! =      9 ℒ#!" +    9 ℒ%!# +    9 ℒ&!$ ,   (19)
               𝑁#
                    $"
                           𝑁%
                                   $#
                                       𝑁&
                                                $$
                                                               training batch, and this mechanism is employed to
                                                               pretrain their utterance-level representations,
where ℒ-"& , ℒ1 "% , and ℒ)"' are phone-level,                 denoted as 𝐡)=)># , and 𝐡)=)>( . Next, we feed the
word-level, and utterance-level losses for disparate           concatenation of these vector representations
aspects, respectively. The parameters 𝜆- , 𝜆1 , and            𝐡)=)> = [𝐡)=)># ; 𝐡)=)>( ] into a three-way classifier,
𝜆) are adjustable parameters which control the                 using the cross-entropy loss as the training
influence of different granularities, and 𝑁- , 𝑁1 ,            objective.
and 𝑁) mark the numbers of aspects at the phone-,
word-, and utterance-levels, respectively.                     3     Experimental Settings
Correlation-aware Regularization Loss. The
correlation-aware regularization loss is defined as            3.1    Evaluation Dataset and Metrics
the difference between the correlation matrix of the           We conducted APA experiments on the
predicted aspect scores ∑  9 and the correlation               speechocean762 dataset, which is a publicly
matrix of the corresponding target labels ∑:                   available open-source dataset specifically designed
                                                               for research on APA (Zhang et al., 2021). This
                                  = , ∑),
                         ℒ89' = ℓ(∑                  (20)      dataset contains 5,000 English-speaking recordings
where ℓ is the regularization loss function, and               spoken by 250 Mandarin L2 learners. The training
each element in ∑9 <3 (or ∑<3 ) is defined as a Pearson        and test sets are of equal size, and each of them has
correlation coefficient between 𝑖 -th aspect score             2,500 utterances, where pronunciation proficiency
                                                               scores were evaluated at multiple linguistic
and 𝑗-th aspect score4. We adopt the MSE criterion
                                                               granularities with various pronunciation aspects.
for computing ℓ ; the overall loss thus can be
                                                               Each score was independently assigned by five
expressed by:
                                                               experienced experts using the same rubrics, and the
                     ℒ = ℒ:$: + 𝜆ℒ89' ,              (21)      final score was determined by selecting the median
                                                               value from the five scores. The evaluation metrics
where 𝜆 ∈ [0, 1] is a tunable parameter, which is
                                                               include Pearson Correlation Coefficient (PCC) and
experimentally set to 0.01 based on the development
                                                               Mean Square Error (MSE). PCC is the primary
set.
                                                               evaluation metric, quantifying the linear
2.4      Pre-training Strategies                               correlation between predicted and ground-truth
                                                               scores. A higher PCC score reflects a stronger
It is without doubt that a proper initialization is
                                                               correlation between the predictions and human
vital for the estimation of a neural model, due
                                                               annotations. In the following experiments, we
mainly to the highly nonconvex nature of the
                                                               report the MSE value in order to evaluate the
training loss function (Tamborrino et al., 2020;

4 To calculate PCC scores between aspects across different     granularities to match the aspect scores at the lower
granularities, we duplicate the aspect scores of higher        granularities.

                                                            1741
                Phone Score       Word Score (PCC)                     Utterance Score (PCC)
      Models
               MSE↓ PCC↑ Accuracy↑ Stress↑ Total↑ Accuracy↑ Completeness↑ Fluency↑ Prosody↑ Total↑
   Lin2021       -        -        -        -      -          -           -           -          -     0.720
  Kim2022        -        -        -        -      -          -           -        0.780       0.770     -
  Ruy2023        -        -        -        -     -       0.719           -        0.775       0.773   0.743
      LSTM     0.089    0.591    0.514    0.294 0.531     0.720         0.076      0.745       0.747   0.741
               ±0.000   ±0.003   ±0.003   ±0.012 ±0.004   ±0.002       ±0.086      ±0.002    ±0.005    ±0.002
      GOPT     0.085    0.612    0.533    0.291 0.549     0.714         0.155      0.753       0.760   0.742
               ±0.001   ±0.003   ±0.004   ±0.030 ±0.002   ±0.004       ±0.039      ±0.008    ±0.006    ±0.005
         0.084 0.616
  HiPAMA ±0.001                  0.575    0.320 0.591     0.730         0.276      0.749       0.751   0.754
                ±0.004           ±0.004   ±0.021 ±0.004   ±0.002       ±0.177      ±0.001    ±0.002    ±0.002
          0.081 0.644
  HierTFR ±0.000                 0.622    0.325 0.634     0.735         0.513      0.801       0.795   0.764
                 ±0.000          ±0.002   ±0.022 ±0.002   ±0.008       ±0.204      ±0.004    ±0.002    ±0.002

 Table 1: The performance evaluations of our model and all compared methods on speechocean762 test set.

phoneme-level APA accuracy in comparison with             and then separately models the corresponding
prior arts.                                               utterance-level aspects with recurrent neural
                                                          models. In addition, LSTM, GOPT (Gong et al.,
3.2     Implementation Details                            2022; Ruy et al. 2023), and HiPAMA (Do et al.,
For the input feature extraction of the phone-level       2023b) are multi-aspect and multi-granular
energy and the duration statistics, we follow the         pronunciation assessments. First, LSTM and
processing flow suggested by Zhu et al. (2022) and        GOPT follow a parallel modeling regime, both of
Shen et al. (2021), where a phone-level feature is        which treat the phone-level input features as a
constructed from time-aggregated frame-level              flattened sequence and assess higher level
features according to the forced alignment. Both          pronunciation scores through stacking LSTM
the phone- and word-level Transformers for                layers or Transformer blocks. Second, Ruy et al.
contextual representation modeling consist of 3           (2023) introduces a unified model architecture that
processing blocks utilizing multi-head attention          jointly optimizes phone recognition and APA tasks.
with 3 heads and 24 hidden units, respectively. In        Lastly, HiPAMA is a hierarchical APA model that
addition, for the word- and utterance-level               more resembles our model than the other methods
attention pooling, we use a single-layer multi-head       compared in this paper. Different from our method,
attention with 3 heads and 24 hidden units. The           HiPAMA extracts high-level pronunciation
combination weights used in Eq. (19) for the APA          features from low-level features based on a simple
loss ( 𝜆- , 𝜆1 , 𝜆) ) are assigned as (1, 1, 1) ,         average pooling mechanism. Furthermore, the
respectively. To ensure the reliability of our            aspect attention mechanism used in HiPAMA
experimental results, we repeated 5 independent           performs on the logistics, whereas our model
trials, each of which consisted of 100 epochs with        operates on the intermediate representations.
different random seeds. The test set results are
reported by averaging those achieved by the top           4       Experimental Results
100 best-performing models which are determined           4.1      Main Results
based on their PCC scores on the development set.
                                                          Table 1 reports the results on the speechocean762
3.3     Compared Methods                                  dataset, which is divided into three parts: the first
We compare our proposed model (viz. HierTFR)              part shows the results of single-aspect assessment
with several families of top-of-the-line methods.         models, the second part presents the results of
Lin et al. (2021) and Kim et al. (2022) are single-       multi-aspect and multi-granular pronunciation
aspect assessment models. The former develops a           methods, and the third part reports the results of our
bottom-up hierarchical scorer evaluating the              model. We further provide a comparison with
accuracy scores at the utterance level. The latter        another hierarchical APA model (viz. HiPAMA) in
leverages self-supervised features (Baevski et al.,       the third part.
2020) to describe the learner’s pronunciation traits

                                                       1742
                                 Accuracy                       Stress                         Total




                                              (a) Word-level Aspect Predictions

                Accuracy           Completeness             Fluency                  Prosody              Total




                                            (b) Utterance-level Aspect Predictions

                Accuracy           Completeness             Fluency                  Prosody              Total




                   (c) Gate Values in Selective Fusion Mechanism for Utterance-level Aspect Predictions

 Figure 4: Qualitative visualization of model parameters when predicting each aspect score. We show (a) the
 averaged attention values for word-level aspects, (b) the averaged attention weights for utterance-level aspects,
 and (c) the averaged gate values for three linguistic levels.

    First, a general observation is that our approach,          recognition task simultaneously. In comparison
HierTFR, excels in all assessment tasks, especially             with the parallel modeling approaches (i.e., GOPT
at the linguistic levels of utterance and word. This            and LSTM), we can observe that HierTFR
performance gain confirms that our model works                  substantially improves the performance across all
comparably better for capturing the relationships               tasks, where its performance gains reveal the
between linguistic units than the other competitive             importance of capturing the hierarchical linguistic
methods. In terms of the utterance-level total score,           structures of an input utterance. Notably, compared
the single-aspect assessment method (viz. Lin2021)              to the HiPAMA, our model consistently achieves
largely falls behind the other multi-aspect and                 superior performance on a variety of pronunciation
multi-granular pronunciation assessment models,                 assessment tasks. This superiority stems from our
which we attribute to the fact that the single-aspect           tactfully designed selective fusion mechanism and
assessment method is unable to harness the                      the correlation-aware loss. The former allows our
dependency relationships between aspects through                model to assess utterance-level aspect scores by
the multi-task learning paradigm. By leveraging                 leveraging information from diverse linguistic
self-supervised learning features, Kim2022                      levels, while the latter explicitly models the
achieves significant improvements over most APA                 relatedness among different aspects during the
methods in terms of the utterance-level                         optimization.
assessments. Next, we scrutinize the performance
of multi-aspect and multi-granular pronunciation                4.2      Qualitative Analysis
assessment methods. Ruy2023 demonstrates                        Qualitative Visualization of Relatedness Among
significant advancements in the utterance-level                 Aspects. In the second set of experiments, we
fluency and prosody assessments due probably to                 examine the relatedness among disparate aspects at
the joint training of the APA model on the phone                both word- and utterance-levels, where the

                                                           1743
                    Phone
                                     Word Score                            Utterance Score
      Models        Score
                   Accuracy   Accuracy   Stress   Total   Accuracy   Completeness   Fluency   Prosody   Total
   HierTFR          0.644      0.622     0.325    0.634    0.735        0.513        0.801     0.795    0.764
   w/o CorrLoss     0.639      0.605     0.348    0.620    0.728        0.520        0.796     0.789    0.758
   w/o Pretrain     0.621      0.545     0.318    0.559    0.716        0.215        0.770     0.772    0.739
   w/o SFusion      0.630      0.608     0.328    0.622    0.728        0.378        0.784     0.782    0.756
   w/o AspAtt       0.636      0.584     0.290    0.596    0.724        0.383        0.784     0.775    0.746

 Table 2: Ablation study on HierTFR, reporting PCC scores on three linguistic levels.

attention weights of the aspect attention                 and utterance-level representations exhibit
mechanisms were determined based on the                   minimal impact on the completeness and total
development set when assessing a specific aspect          aspects, respectively. One possible reason is that
score. For the word-level assessments, the                the completeness aspect somehow reflects
distributions of attention weights are in close           pronunciation intelligibility, and our model learns
accordance with the manual scoring rubrics of the         to distill the information from the phone- and
speechocean762 dataset. In Figure 4(a), the total         utterance-level representations. On the other hand,
aspect serves as a comprehensive assessment and           the total aspect evaluates an overall speaking skill.
the corresponding weights are contributed from            Our model thus tends to capture the subtle
various pronunciation aspects. In contrast, the           information by distilling the fine-grained traits
accuracy aspect measures the percentage of                inherent in the phone- and word-levels.
mispronounced phones within a word, leading to
the attention weights being more concentrated on a        4.3    Ablation Study
word-level unit itself. Furthermore, the stress score     To gain insight into the effectiveness of each model
also highly attends to the accuracy aspect,               component of HierTFR, we conduct an ablation
reflecting the strong relation between lexical stress     study to investigate their impacts. These variations
and word-level pronunciation accuracy (Korzekwa           include excluding the correlation-aware regularizer
et al., 2022). In regard to the relatedness within the    (w/o CorrLoss), removing the proposed pre-
utterance-level aspects, inspecting Figure 4(b) we        training strategies (w/o Pretrain), omitting the
find that the attention weights of the prosody and        selective fusion mechanism (w/o SFusion), and
total aspects scatter across various pronunciation        eliminating the aspect attention mechanism at both
aspects, whereas the attention weights of the             word and utterance levels (w/o AspAtt). From
accuracy and completeness center primarily on the         Table 2, we can observe that the proposed
completeness aspect. One possible reason is that          correlation-aware regularization loss is beneficial
the prosody and total scores both measure high-           for most pronunciation assessment tasks. Next, the
level oral skills, and when the human annotators          proposed pre-training strategies are crucial to
judge the proficiency scores, they also take              obtaining better performance as the model trained
multiple pronunciation aspects into account               without them tends to perform relatively worse for
simultaneously. Next, the completeness aspect             all pronunciation assessment tasks. This highlights
measures the percentage of words with good                the efficacy of the pre-training strategies for
pronunciation quality in an utterance. This               hierarchical APA models, thereby alleviating the
implicitly reflects the intelligibility of a learner's    requirement for large amounts of supervised
pronunciation and is vital to the accuracy                training data. Third, removing the selective fusion
assessment.                                               mechanism leads to degradations in the utterance-
Qualitative Visualization of Interactions Across          level aspect assessments, while removing the
Linguistic Levels. In Figure 4(c), we report on the       aspect attention mechanism deteriorates the
average gate values of utterances for three               performance on word-level aspect assessments.
linguistic granularities by estimating the utterance-
level pronunciation aspect scores based on the            5     Related Work
development set. We can observe that the phone-
level representations bear high impacts on the            Early studies on APA focused primarily on single-
utterance-level aspect assessments, in comparison         aspect assessments, typically through individually
to the other linguistic levels. Next, the word-level      constructing scoring modules to predict a holistic

                                                      1744
pronunciation proficiency score on a targeted             be untenable when assessing the L2 learners with
linguistic level or some specific aspect with             diverse accents.
different sets of hand-crafted features, such as the      The lack of Interpretability. The model of the
phone-level posterior probability (Witt and Young,        proposed method simply trains to mimic expert’s
2000), word-level lexical stress (Ferrer et al., 2015),   annotations without resorting to manual
or various utterance-level pronunciation aspects          assessment rubrics or other external knowledge,
(Coutinho et al., 2016). More recently, with the          making it not straightforward to provide reasonable
rapid progress of deep learning (Vaswani et al.,          explanations for the assessment results.
2017; Raffel et al., 2020; Hsu et al., 2021), several     Ethics Statement
neural scoring models have been successfully
developed for multi-aspect and multi-granular             We hereby acknowledge that all of the co-authors
pronunciation assessment. Gong et al. (2022)              of this work compile with the provided ACL Code
proposed a GOP feature-based Transformer                  of Ethics and honor the code of conduct. Our
(GOPT) architecture to model pronunciation                experimental corpus, speechocean762, is widely
aspects at multiple granularities with a multi-task       used and publicly available. We think there are no
learning scheme. Do et al. (2023b) employed a             potential risks for this work.
neural scorer with a hierarchical structure to mimic
the language hierarchy of an utterance to deliver         References
state-of-the-art performance for APA.                     Stefano Bannò, Bhanu Balusu, Mark Gales, Kate
                                                             Knill,and Konstantinos Kyriakopoulos. 2022. View-
6   Conclusion                                               specific assessment of L2 spoken English. In
                                                             Proceedings of Interspeech (INTERSPEECH),
In this paper, we have put forward a novel                   pages 4471–4475.
hierarchical modeling method (dubbed HierTFR)             Alexei Baevski, Henry Zhou, Abdelrahman Mohamed,
for multi-aspect and multigranular APA. To                  and Michael Auli. 2020. Wav2vec 2.0: A framework
explicitly capture the relatedness between                  for    self-supervised    learning     of    speech
pronunciation aspects, a correlation-aware                  representations. In Proceedings of the International
regularizer loss has been devised. We have further          Conference on Neural Information Processing
developed model pre-training strategies for our             Systems (NIPS), pages 12449–12460.
HierTFR model. Extensive experimental results             Fu An Chao, Tien Hong Lo, Tzu I. Wu, Yao Ting Sung,
confirm the feasibility and effectiveness of the             Berlin Chen. 2022. 3M: An effective multi-view,
proposed method in relation to several top-of-the-           multigranularity, and multi-aspect modeling
                                                             approach to English pronunciation assessment. In
line methods. In future work, we plan to examine
                                                             Proceedings of the Asia-Pacific Signal and
the proposed HierTFR model on open-response                  Information Processing Association Annual Summit
scenarios, where learners speak freely or respond            and Conference (APSIPA ASC), pages 575–582.
to a given task or question (Wang et. al., 2018; Park
                                                          Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu
and Choi, 2023). In addition, the issues of
                                                            Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki
explainable pronunciation feedback are also left as         Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu,
a future extension.                                         Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian
                                                            Wu, Michael Zeng, Xiangzhan Yu, Furu Wei. 2022.
Limitations
                                                            Wavlm: Large-scale self-supervised pre-training for
Limited Applicability. In this research, the                full stack speech processing. IEEE Journal of
proposed model focus on the “reading-aloud”                 Selected Topics in Signal Processing, volume 16,
pronunciation training scenario, where the                  pages1505–1518.
assumption is that the L2 learner pronounces a            Nancy F. Chen, and Haizhou Li. 2016. Computer-
predetermined text prompt correctly, which                  assisted pronunciation training: From pronunciation
restricts the applicability of our models to other          scoring towards spoken language learning. In
learning scenarios, such as freely speaking or open-        Proceedings of the Asia-Pacific Signal and
ended conversations.                                        Information Processing Association Annual Summit
Lack of Accent Diversity. The used dataset merely           and Conference (APSIPA ASC), pages 1–7.
contains Mandarin L2 learners, hindering the              Eduardo Coutinho, Florian Hönig, Yue Zhang, Simone
generalizability of the proposed model and could            Hantke, Anton Batliner, Elmar Nöth, and Björn
                                                            Schuller. 2016. Assessing the prosody of non-native


                                                      1745
  speakers of English: Measures and feature sets. In     Wenping Hu, Yao Qian, Frank K. Soong, and Yong
  Proceedings of the International Conference on           Wang. 2015. Improved mispronunciation detection
  Language Resources and Evaluation (LREC), pages          with deep neural network trained acoustic models
  1328–1332.                                               and transfer learning based logistic regression
                                                           classifiers. Speech Communication, volume 67,
Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023a.
                                                           pages 154–166.
  Score-balanced Loss for Multi-aspect Pronunciation
  Assessment. In Proceedings of Interspeech              Eesung Kim, Jae-Jin Jeon, Hyeji Seo, Hoon Kim. 2022.
  (INTERSPEECH), pages 4998–5002.                          Automatic pronunciation assessment using self-
                                                           supervised speech representation learning. In
Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023b.
                                                           Proceedings of Interspeech (INTERSPEECH),
  Hierarchical pronunciation assessment with multi-
                                                           pages 1411–1415.
  aspect attention. In Proceedings of the IEEE
  International Conference on Acoustics, Speech and      Yassine Kheir, Ahmed Ali, and Shammur Chowdhury.
  Signal Processing (ICASSP), pages 1–5.                   2023. Automatic Pronunciation Assessment - A
                                                           Review. In Findings of the Association for
Maxine Eskenazi. 2009. An overview of spoken
                                                           Computational Linguistics: EMNLP, pages 8304–
  language technology for education. Speech
                                                           8324.
  communication, volume 51, pages 832–844.
                                                         Daniel Korzekwa, Jaime Lorenzo-Trueba, Thomas
Keelan Evanini, and Xinhao Wang. 2013. Automated
                                                           Drugman, and Bozena Kostek. 2022. Computer-
  speech scoring for Nonnative middle school
                                                           assisted pronunciation training—Speech synthesis
  students with multiple task types. In Proceedings of
                                                           is almost all you need. Speech Communication,
  Interspeech (INTERSPEECH), pages 2435–2439.
                                                           volume 142, pages 22–33.
Keelan Evanini, Maurice Cogan Hauck, and Kenji
                                                         Kushal Lakhotia, Eugene Kharitonov, Wei-Ning Hsu,
  Hakuta. 2017. Approaches to automated scoring of
                                                           Yossi Adi, Adam Polyak, Benjamin Bolte, Tu-Anh
  speaking for K–12 English language proficiency
                                                           Nguyen, Jade Copet, Alexei Baevski, Abdelrahman
  assessments. ETS Research Report Series, pages 1–
                                                           Mohamed, and Emmanuel Dupoux. 2021. On
  11.
                                                           Generative Spoken Language Modeling from Raw
Luciana Ferrer, Harry Bratt, Colleen Richey, Horacio       Audio. Transactions of the Association for
  Franco, Victor Abrash, and Kristin Precoda. 2015.        Computational Linguistics, volume 9, pages 1336–
  Classification of lexical stress using spectral and      1354.
  prosodic features for computer-assisted language
                                                         Binghuai Lin and Liyuan Wang. 2021. Deep feature
  learning      systems. Speech      Communication,
                                                           transfer learning for automatic pronunciation
  volume 69, pages 31–45.
                                                           assessment. In Proceedings of Interspeech
Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and           (INTERSPEECH), pages 4438–4442.
  Luke Zettlemoyer. 2019. Mask-Predict: Parallel
                                                         Silke M. Witt and S. J. Young. 2000. Phone-level
  Decoding of Conditional Masked Language Models.
                                                            pronunciation scoring and assessment for
  In Proceedings of the Conference on Empirical
                                                            interactive   language     learning.    Speech
  Methods in Natural Language Processing and the
                                                            Communication, volume 30, pages 95–108.
  International Joint Conference on Natural
  Language Processing (EMNLP-IJCNLP), pages              Jungbae Park and Seungtaek Choi. 2023. Addressing
  6112–6121.                                                cold start problem for end-to-end automatic speech
                                                            scoring.     In   Proceedings     of   Interspeech
Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang,
                                                            (INTERSPEECH), pages 994–998.
  and James Glass. 2022. Transformer-based multi-
  aspect multigranularity non-native English speaker     Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji
  pronunciation assessment. In Proceedings of the           Watanabe. 2022. Branchformer: Parallel mlp-
  IEEE International Conference on Acoustics,               attention architectures to capture local and global
  Speech and Signal Processing (ICASSP), pages              context for speech recognition and understanding.
  7262–7266.                                                In International Conference on Machine Learning
                                                            (PMLR), pages 17627–17643
Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai,
  Kushal Lakhotia, Ruslan Salakhutdinov, and             Yu Wang, M.J.F. Gales, Kate M Knill, Konstantinos
  Abdelrahman Mohamed. 2021. HuBERT: Self-                 Kyriakopoulos, Andrey Malinin, Rogier C van
  Supervised Speech Representation Learning by             Dalen, Mohammad Rashid. 2018. Towards
  Masked Prediction of Hidden Units. IEEE/ACM              automatic assessment of spontaneous spoken
  Transactions on Audio, Speech and Language               English. Speech Communication, volume 104,
  Processing, volume 29, pages 3451–3460.                  pages 47–56.



                                                     1746
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine           In Proceedings of Interspeech (INTERSPEECH),
  Lee, Sharan Narang, Michael Matena, Yanqi Zhou,             pages 3710 –3714.
  Wei Li, and Peter J. Liu. 2020. Exploring the limits
                                                         Chuanbo Zhu, Takuya Kunihara, Daisuke Saito,
  of transfer learning with a unified text-to-text
                                                           Nobuaki Minematsu, Noriko Nakanishi. 2022.
  transformer. The Journal of Machine Learning
                                                           Automatic prediction of intelligibility of words and
  Research, volume 21, pages 5485–5551.
                                                           phonemes produced orally by japanese learners of
Robert Ridley, Liang He, Xin-yu Dai, Shujian Huang,        English. In IEEE Spoken Language Technology
  and Jiajun Chen. 2021. Automated cross-prompt            Workshop (SLT), pages. 1029–1036.
  scoring of essay traits. In Proceedings of the AAAI
  conference on artificial intelligence (AAAI), volume   A Pronunciation Feature Extractions
  35, pages 13745–13753.                                 GOP Feature. To extract the GOP feature, we first
Pamela M Rogerson-Revell. 2021. Computer-assisted        align audio signals X with the text prompt T by using
  pronunciation training (CAPT): Current issues and      an ASR model 5 to obtain the timestamps for each
  future directions. RELC Journal, volume 52, pages      phone in the canonical phone sequence. Next, frame-
  189–205.                                               level phonetic posterior probabilities are produced by
                                                         the ASR model and then averaged over the time
Hyungshin Ryu and Sunhee Kim and Minhwa Chung.           dimension based on the phone-level timestamps. The
  2023. A joint model for pronunciation assessment       resulting phone-level posterior probabilities are
  and mispronunciation detection and diagnosis with      converted into a GOP feature vector as a combination
  multi-task learning. In Proceedings of Interspeech     of log phone posterior (LPP) and log posterior ratio
  (INTERSPEECH), pages 959–963.                          (LPR). Owing to the used ASR model containing 42
Yang Shen, Ayano Yasukagawa, Daisuke Saito,              phones, the GOP feature of a canonical phone 𝑝 can be
  Nobuaki Minematsu, and Kazuya Saito. 2021.             represented as an 84-dimensional vector:
  Optimized prediction of fluency of L2 English
  based on interpretable network using quantity of           [LPP(𝑝; ), … , LPP(𝑝<= ),                      (22)
  phonation and quality of pronunciation. In                                   LPR(𝑝; |𝑝), … , LPR(𝑝<= |𝑝)]
  Proceedings of IEEE Spoken Language Technology                     LPP(𝑝> ) = log𝑝(𝑝> |𝐨; t ? , t @ )
  Workshop (SLT), pages 698–704.                                                       A%
                                                                            1                              (23)
Alexandre Tamborrino, Nicola Pellicanò, Baptiste                     =             ` log𝑝(𝑝> |oA ),
                                                                       𝑡@ − 𝑡? + 1
  Pannier, Pascal Voitot, and Louise Naudin. 2020.                                   ABA&
  Pretraining is (almost) all you need: An application
  to commonsense reasoning. In Proceedings of the              LPR(𝑝> |𝑝) = log𝑝(𝑝> |𝐨; t ? , t @ )        (24)
  Association for Computational Linguistics (ACL),                              − log𝑝(𝑝|𝐨; t ? , t @ ),
  pages 3878–3887.                                       where LPR is the log posterior ratio between phones 𝑝>
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob         and 𝑝; 𝑡? and 𝑡@ are the start and end timestamps of
  Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz          phone 𝑝, and 𝑜A is the input acoustic observation of the
  Kaiser, and Illia Polosukhin. 2017. Attention is all   time frame 𝑡.
  you need. In Proceedings of the Conference on          Energy Feature. The energy feature is a 7-
  Neural Information Processing Systems (NeurIPS),       dimensional vector comprised of (viz., [mean, std,
  pages 5998–6008.                                       median, mad, sum, max, min]) over phone segments,
                                                         where the root-mean-square energy (RMSE) is
Heng-Da Xu, Zhongli Li, Qingyu Zhou, Chao Li,            employed to compute energy value for each time frame,
  Zizhen Wang, Yunbo Cao, Heyan Huang, and Xian-         with 25-millisecond windows and a stride of 10
  Ling Mao. 2021. Read, listen, and see: Leveraging      milliseconds.
  multimodal information helps Chinese spell             Duration Feature. The duration feature is a 1-
  checking. In Findings of the Association for           dimensional vector indicating the length of each phone
  Computational Linguistics (ACL-IJCNLP Findings),       segment in seconds.
  pages 716–728.
Junbo Zhang, Zhiwen Zhang, Yongqing Wang,
   Zhiyong Yan, Qiong Song, Yukai Huang, Ke Li,
   Daniel Povey, and Yujun Wang. 2021.
   Speechocean762: An open-source non-native
   English speech corpus for pronunciation assessment.


5
 A public-assessable ASR model trained with English
speech corpus: https://kaldi-asr.org/models/m13.

                                                      1747
