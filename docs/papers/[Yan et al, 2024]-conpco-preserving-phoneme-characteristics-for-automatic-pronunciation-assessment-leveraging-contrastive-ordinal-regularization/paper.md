      ConPCO: Preserving Phoneme Characteristics for Automatic Pronunciation
            Assessment Leveraging Contrastive Ordinal Regularization
    Bi-Cheng Yan1*, Wei-Cheng Chao2, Jiun-Ting Li1, Yi-Cheng Wang1, Hsin-Wei Wang1, Meng-Shin
                                        Lin1, Berlin Chen1*
                                1
                               National Taiwan Normal University, Taipei, Taiwan
                   2
                       Telecommunication Laboratories, Chunghwa Telecom Co., Ltd., Taiwan
                                            {bicheng, berlin}@ntnu.edu.tw


                            Abstract                                                                      Reading-aloud Scenario
                                                                                                            What about the bus.
Automatic pronunciation assessment (APA) manages to
evaluate the pronunciation proficiency of a second language                                                 What about the bus.
(L2) learner in a target language. Existing efforts typically draw                              Automatic Pronunciation Assessment Results
on regression models for proficiency score prediction, where                        Utterance level               Word level               Phone level
                                                                                   Aspects      Scores    Words   Aspects      Scores   Phones       Scores
the models are trained to estimate target values without                                                          Accuracy       2        W              2.0
explicitly accounting for phoneme-awareness in the feature                        Accuracy        1.8
                                                                                                          What     Stress        2       AH0             1.8

space. In this paper, we propose a contrastive phonemic ordinal                                                     Total        2        T              2.0

regularizer (ConPCO) tailored for regression-based APA                             Fluency        1.8             Accuracy       2       AH0             2.0
                                                                                                          About    Stress        2        B              2.0
models to generate more phoneme-discriminative features                                                             Total        2      AW1              2.0
while considering the ordinal relationships among the                            Completeness         2
                                                                                                                  Accuracy       2        T              2.0

regression targets. The proposed ConPCO first aligns the                                                   The     Stress        2       DH              2.0
                                                                                   Prosody        1.6
phoneme representations of an APA model and textual                                                                 Total        2       AH0             2.0
                                                                                                                  Accuracy       2        B              2.0
embeddings of phonetic transcriptions via contrastive learning.                     Total         1.6      Bus     Stress        2       AH0             2.0
Afterward, the phoneme characteristics are retained by                                                              Total        2        S              2.0

regulating the distances between inter- and intra-phoneme                 Figure 1: An example curated from the speechocean762
categories in the feature space while allowing for the ordinal            dataset [7] illustrates the evaluation flow of an APA
relationships among the output targets. We further design and             system in the reading-aloud scenario, which offers an
develop a hierarchical APA model to evaluate the effectiveness            L2 learner in-depth pronunciation feedback.
of our method. Extensive experiments conducted on the
speechocean762 benchmark dataset suggest the feasibility and         linguistic levels (i.e., phone, word, and utterance levels) with
efficacy of our approach in relation to some cutting-edge            multiple aspects (e.g., accuracy, fluency, and completeness), as
baselines.                                                           the running example depicted in Figure 1. Due to the continuity
                                                                     in the values of output targets, prevailing approaches typically
Index Terms: computer-assisted pronunciation training,
                                                                     adopt a regression loss function, such as mean-squared error
automatic pronunciation assessment, contrastive learning
                                                                     (MSE), as the objective for training neural models to estimate
                                                                     multiple aspect score sequences across various linguistic units
                       1. Introduction                               [8][10][11]. Although some promising results have been
Automatic pronunciation assessment (APA) aims to determine           achieved, the regression-based APA models are merely trained
the levels of second language (L2) learners’ oral proficiency        to mimic expert’s annotations, where the distinct features of
and then provide detailed feedback on specific pronunciation         phoneme categories are largely neglected in the optimization
aspects pertaining to a target language [1][2]. As APA systems       process. In this work, we identify three limitations in existing
can deliver steady and instant assessment results, they are          regression-based APA models: (1) the phoneme representations
beneficial towards the field of language education, such as          of input speech, derived by an APA model, and the textual
providing pronunciation feedback for L2 learners without             embeddings of phoneme-level text prompts are located in
subjective factors [3], greatly alleviating the workload of          separate feature spaces, posing challenges for accessing the
teachers [4], and serving as a handy reference for professionals     phoneme scores while rentaining awareness of phoneme
(e.g., interviewers and examiners) in high-stake assessment          identities, (2) different phoneme representations belonging to
tasks [5][6].                                                        the same proficiency level are inadvertently forced to be close
                                                                     to one another, which would harm the performance of
    A de-facto standard for APA systems is instantiated in a
                                                                     assessment tasks related to pronunciation clarity [12], and (3)
reading-aloud scenario, where an L2 learner is presented with a
                                                                     the ordinal relationships among the regression targets are
text prompt and instructed to pronounce it accordingly. To
                                                                     almost overlooked in the training objective, where the
provide in-depth pronunciation feedback, recent efforts have
                                                                     ordinalities observed in the label space are not properly
drawn attention to multi-aspect and multi-granular
                                                                     reflected in the feature space.
pronunciation assessment, which evaluates oral skills at various



*
    Corresponding author.
    To address these limitations, we present a novel training                                           Accuracy           Fluency Completeness Prosody                   Total


regime, dubbed contrastive phonemic ordinal regularizer                                                      Utt
                                                                                                            Head
                                                                                                                             Utt
                                                                                                                            Head
                                                                                                                                              Utt
                                                                                                                                             Head
                                                                                                                                                            Utt
                                                                                                                                                           Head
                                                                                                                                                                           Utt
                                                                                                                                                                          Head




                                                                                                                                                                                    Utterance-level
(ConPCO), for enhancing regression-based APA models by
                                                                                                             Atten           Atten            Atten         Atten          Atten
                                                                                                            Pooling         Pooling          Pooling       Pooling        Pooling




capturing phoneme characteristics in the feature representations
                                                                                                                                      Utterance Encoder

                                                                                                                                                                                                                  Weighted Comb
                                                                                                                                         Cat & Proj

while maintaing the ordinal relationships among the regression                Total
                                                                          Stress
                                                                        Accuracy
                                                                                                Total
                                                                                            Stress
                                                                                          Accuracy
                                                                                                                          Merge (Weighted Average Operation)                                                                      Dropout

targets. The proposed ConPCO aligns the output
                                                                                                                                                                                                        Dropout




                                                                                                                                                                                    Word-level
                                                                         Word         …    Word                                         Word Encoder                                                                         Pointwise Conv
                                                                         Head              Head

representations from a phoneme encoder of an APA model with
                                                                                                                                                                                                      Feed-forward
                                                                       Word [Call] … Word [Bear]                 Cat & Proj                 Cat & Proj             Cat & Proj
                                                                                                                                                                                                                             ReLU Activation


the embeddings of phoneme-level text prompt via a contrastive            Accuracy         Accuracy           Atten Pooling              Atten Pooling          Atten Pooling                           LayerNorm
                                                                                                                                                                                                                               LayerNorm
                                                                          Phn               Phn




                                                                                                                                                                                    Phoneme-level
                                                                                      …                                               Phoneme Encoder

loss, which pulls the paired phoneme representations closer
                                                                          Head              Head                                                                                                                             Depthwise Conv
                                                                         Phn [k]      …    Phn [R]
                                                                                                        0             1      2          3          4       5         6      7                           Dropout

while pushing those of the non-matched pair apart. To model
                                                                                                                                                                                                                             GLU Activation
                                                                                                                                      Feature Extraction
                                                                                                                                                                                                      Self-Attention         Pointwise Conv

the nuances of phoneme categories, the feature representations            Input Audio Signals X:
                                                                                                                                                                                                       LayerNorm               LayerNorm

from the same phoneme category are pulled closer together by            Text Prompt T : Call It Bear
                                                                                                             K        AO
                                                                                                                 Word 1 (Call)
                                                                                                                                 L          IH
                                                                                                                                            Word 2 (It)
                                                                                                                                                       T       B     EH
                                                                                                                                                               Word 3 (Bear)
                                                                                                                                                                             R



considering the ordinal relationships, and meanwhile the                                                                    (a)                                                                                        (b)
representations of different categories are forced to be further       Figure 2: The architecture of the proposed hierarchical APA
spread apart. In addition, we also design a novel hierarchical         model, built upon the novel convolution-augmented
APA model, dubbed HierCB, built upon the newly introduced              Branchformer encoder block. As designed, (a) HierCB
Convolution-augmented Branchformer blocks to demonstrate               hierarchically represents an L2 learner’s input utterance,
the effectiveness of ConPCO.                                           and (b) the proposed convolution-augmented Branchformer
    In summary, the main contributions of this work are: (1) to        block.
the best of our knowledge, ConPCO is the first attempt to
explore contrastive learning for projecting the phoneme              where 𝜙(𝐳($ , 𝐳<) ) is dot product between ℓ& normalized vectors
representations of an APA model and the embeddings of                𝐳($ and 𝐳<) (cosine similarity). During training, the set ℳ is
phoneme-level text prompt into a shared latent feature space, (2)    constructed from each batch, where we empirically sample the
we further develop a simple yet effective hierarchical APA           data instances with the highest proficiency score to calculate
model to verify the proposed training regime, which enhances         centroid vectors.
the Branchformer model [13] with a newly proposed                    Phonemic Characteristic Term. The phonemic characteristic
convolution module, (3) Extensive sets of experiments carried        term ℒ$! preserve the phonemic proximity information by
out on a public APA dataset confirm the utility of our proposed                                                                  $
                                                                     minimize the negative distances between centroid vectors 𝐳( :
method which considerably improves the effectiveness of                                      %      *        $     $
multiple assessments across various linguistic levels.                        ℒ$! = −             ∑(;% ∑(><>𝐳( − 𝐳< > ,         (4)
                                                                                                                          *(*=%)                                                                         &
                                                                     where ℒ$! is equivalent to maximizing the distances between
                    2. Methodology                                   phoneme categories during the optimization process.
2.1. Contrastive Phonemic Ordinal Regularizer (ConPCO)               Ordinal Term. To reflect ordinal relationships of regresion
The proposed ConPCO regularizer consists of three                    targers in the feature space, the ordinal term ℒ" is defined to
mathematical terms: the contrastive term ℒ!"# , the phonemic         minimize the distance between the feature representations 𝐡?(
characteristic term ℒ$! , and the ordinal term ℒ" . ℒ!"# aims to     and their corresponding phoneme centroied vectors 𝐳($ while
simultaneously project the phoneme representations generated         being aware of relative differences of proficiency score:
                                                                                      %              $             $
from an APA model and the embeddings of phoneme-level text                     ℒ" = ' ∑'       ?
                                                                                         (;%>𝐡( − 𝐳( >& × ?𝐶 − 𝑦( ?,            (5)
prompt into a joint feature space. ℒ$! seeks to adjust the
                                                                     where 𝑦($ is the corresponding phone-level proficiency score,
distances between inter- and intra-phoneme categories in the
                                                                     and the 𝐶 is a tuneable constant, chosen as 2 due to the highest
learned feature representations, while ℒ" reflects the ordinal
                                                                     phoneme-level score.
relationships in the feature space.
Contrastive Term. Let H$ = (𝐡%$ , 𝐡$& , … , 𝐡$' ) stand for the      2.2. Hierarchical APA Model Based on Convolution-
phoneme representation sequence of an utterance generated by         augmented Branchformer (HierCB)
a phoneme encoder of an APA model, and E$ =                          Figure 2(a) illustrates the architecture of the proposed
  $ $           $                                                    hierarchical APA model, dubbed HierCB, consisting of three
(𝐞% , 𝐞& , … , 𝐞' ) denote the embedding sequence of a phoneme-
                                                                     main components: phoneme-level, word-level, and utterance-
level text prompt. To obtain a set of paired phoneme
                                                                     level modeling. Each encoder in the different modeling
representations ℳ = {-𝐳($ , 𝐳() /, 𝑖 = 1, … , 𝑀}, we calculate the   processes exploits a newly designed convolution-augmented
centroid vectors for each phoneme category in H$ and E$ ,            Branchformer block, as shown in Figure 2(b).
followed by separate linear projections. In set ℳ, the 𝑀 × 𝑀
similarities are generated, and the contrastive term ℒ!"# seeks      2.2.1. Convolution-augmented Branchformer
to maximize the similarity between paired phoneme                    Owing to the capability of various ranged context modeling,
representations while minimizing the similarity of unpaired          Branchformer [14] is more suitable for constructing a
ones at the same time [15][16]. The contrastive term ℒ!"#            hierarchical APA model compared to other advanced neural
includes two losses, with a temperature hyper-parameter 𝜏 that       models [17][18]. The Branchformer block consists of two
controls the strength of penalties on negative samples:              parallel branches, where one branch captures global context
                      ℒ!"# = ℒ$&) + ℒ)&$ ,                    (1)    through a multi-head self-attention (MHA) module, while the
                    %
                                           "
                                 +,- (01𝐳! ,𝐳!# 4/6)                 other learns local context via a multi-layer perceptron module
         ℒ$&) = − ∑*
                   (;% log %                             ,   (2)     with a convolutional gating mechanism (cgMLP) [19]. To
                   *           ∑$&' +,-909𝐳!" ,𝐳$# :/6:
                                               "
                                                                     effectively    represent    localized     information    within
                    %            +,- (01𝐳!# ,𝐳! 4/6)
         ℒ)&$ = − * ∑*
                     (;% log %                           ,   (3)     pronunciation feature sequences at lower levels of granularity
                               ∑$&' +,-909𝐳!# ,𝐳$":/6:               (viz. phoneme and word units), the proposed convolution-
augmented Branchformer block replaces the cgMLP layers in                      Table 1: Statistics of the speechocean762.
the original architecture with a convolution module. As shown                                                   Score            # of Counts
in Figure 3(b), our convolution module starts with a gating            Granularities             Aspects
                                                                                                               Interval        Train     Test
mechanism that includes a pointwise convolution and a gated             Phoneme                 Accuracy        [0, 2]        47,076 47,369
linear unit function (GLU) [20]. Stacked on top of the above                                    Accuracy
module, a 1-D depth-wise convolution layer, with a kernel size            Word                    Stress       [0, 10]        15,849   15,967
of 3, is applied to capture local information, which is then                                      Total
normalized with layer normalization and activated by a rectified                                Accuracy
linear unit (ReLU) function. Next, another pointwise                                           Completeness
convolution is applied, followed by a dropout layer to prevent          Utterance                Fluency       [0, 10]        2,500    2,500
overfitting. Meanwhile, the other branch retains the MHA                                         Prosody
module. The entire block is structured as a residual network,                                     Total
with these two branches are further merged by a weighted            obtain word-level output representations W   HK . Then, the 1-D
average operation [13].                                             depth-wise convolution layers are individually added on top of
2.2.2. Hierarchical APA modeling                                                   W K , which are further combined with a linear
                                                                    X $ , H$ , and H
                                                                    projection to form a sequence of utterance-level input
Phoneme-level Modeling. For an input utterance, we first            representations X D . Afterward, an utterance encoder is
extract various pronunciation features to portray the               exploited to generate utterance-level contextualized
pronunciation quality of the L2 learner at phoneme-level, which     representations HD :
are then concatenated and projected to obtain a sequence of                       W K = Merge(HK' , HK( , HK) ),
                                                                                  H                                           (13)
condensed acoustic features X $ . The feature extraction process          D                                         W K )]),
                                                                         HJ = LinearD ([DC% (X $ ); DC& (H$ ); DCM (H         (14)
is formulated as:
       X $ = Linear$ ([E@AB ; ECDE ; EF#G ; E HHI ]),       (6)                         HD = UttEnc(HJD ),                    (15)
where Linear$ (∙) is a linear layer, E@AB is goodness of            where Merge(∙) is a weighted average operation [13], UttEnc(∙
pronunciation (GOP)-based features [22], ECDE and EF#G are          ) is a single convolution-augmented Branchformer block, and
prosodic features of duration and energy statistics [23][24], and   DC% (∙), DC& (∙), DCM (∙) are distinct 1-D depthwise convolution
E HHI are self-supervised learning (SSL) based features. We then    layers, each with a kernel size of 3. Finally, five separate
add phoneme-level textual embeddings EB to X $ , and employ a       attention pooling modules are added on top of HD to generate
phoneme encoder to obtain aspect-specific representations H$ :      utterance-level aspect representation vectors which are then
                                                                    processed by regression heads to derive the corresponding
                      HJ$ = X $ + EB ,                       (7)
                                                                    utterance-level scores.
                  H$ = PhnEnc-HJ$ /.                         (8)    Training Objective. The training objective of the multi-aspect
Here, EB is generated by passing a phoneme-level text prompt        and multi-granular pronunciation assessment, ℒNBN , is
into a phoneme and position embedding layer, and the phoneme        calculated as a weighted sum of the mean square error (MSE)
encoder PhnEnc(∙) is a stack of 3 convolution-augmented             losses gathered from different granularity levels. Furthermore,
Branchformer blocks. Next, the regression head is built on top      we also integrate ConPCO regularizer into the optimization
of H$ to access phoneme accuracy scores.                            process:
Word-level Modeling. For the word-level assessments, we                                ℒ = ℒNBN + ℒO"#BOA ,                   (16)
                                                                                       𝜆               𝜆              𝜆
start with word-level attention pooling to derive a word                  ℒNBN = 𝑁𝑝 ∑𝑗𝑝 ℒ𝑝𝑗𝑝 + 𝑁𝑤 ∑𝑗𝑤 ℒ𝑤𝑗𝑤 + 𝑁𝑢 ∑𝑗𝑢 ℒ𝑢𝑗𝑢 ,                (17)
representation vector from its constituent phones, achieved by                             𝑝               𝑤              𝑢

a 1-D depthwise convolution layer followed by an MHA layer                        ℒO"#BOA = ℒ!"# + ℒ$! + ℒ" ,                             (18)
and an average operation. The word-level input representations      where ℒ$$" , ℒK $* , and ℒD$+ are phone-level, word-level, and
X K are computed by individually passing X $ and H$ into the
word-level attention pooling. The resulting representations are     utterance-level losses for disparate aspects, respectively. The
then packed together via a linear projection:                       parameters 𝜆$ , 𝜆K , and 𝜆D are adjustable parameters which
       PK , P
       X    HK = AttPoolL' (X $ ), AttPoolL( (H$ ),       (9)       control the influence of different granularities, and 𝑁$ , 𝑁K , and
                                                                    𝑁D mark the numbers of aspects at the phone-, word-, and
                              PK ; P
              X K = LinearK (SX    HK T).                   (10)    utterance-levels, respectively.
                                       K                K
The word-level textual embeddings E are added to X , and a
word encoder is employed to generate word-level                                                  3. Experiments
contextualized representations HK :
                   HJK = X K + EK ,                  (11)           3.1.1. Experimental Settings
                 H = WordEnc(HJK ),
                  K
                                                     (12)           Dataset. We conducted APA experiments on the
                                                                    speechocean762 dataset, which is a publicly available dataset
where EK are obtained by mapping a text prompt into its
                                                                    specifically designed for research on APA [7]. This dataset
corresponding embedding sequence via a word and position
                                                                    contains 5,000 English-speaking recordings spoken by 250
embedding layer, and WordEnc(∙) consists of 2 convolution-
augmented Branchformer blocks. Finally, three distinct 1-D          Mandarin L2 learners. The training and test sets are of equal
                                                                    size, and each of them has 2,500 utterances, where
depth-wise convolution layers are performed on HK to generate
                                                                    pronunciation proficiency scores were evaluated at multiple
word-level aspect representations (i.e., HK' , HK( , and HK) ),
                                                                    linguistic granularities with various aspects. Table 1
which are then transformed into the pronunciation score
                                                                    summarizes the statistics of the experimental dataset.
sequences with the corresponding word-level regression heads.
Utterance-level Modeling. For the utterance-level assessments,      Implementation Details. For the input feature extraction, the
we first merge HK' , HK( , and HK) with weighted averaging to       energy and the duration statistics follow the processing flow
 Table 2: Experimental results of various methods for
 pronunciation assessments at phoneme and word levels.
   Input                   Phone Score          Word Score (PCC)
               Models
   Feats                  MSE     PCC         Acc.   Stress Total
              LSTM [11]   0.089 0.591         0.514 0.294 0.531
              GOPT [11]   0.085 0.612         0.533 0.291 0.549
   GOP
               GFR [13]   0.079 0.646         0.598 0.334 0.614
              HiPAMA[9]   0.084 0.616         0.575 0.320 0.591                     (a) w/o ConPCO                (b) w/ ConPCO

              GOPT-SSL    0.081 0.640         0.584 0.352 0.603          Figure 3: Visualization of phoneme representations
                3M [8]    0.078 0.656         0.575 0.320 0.591          generated by the proposed hierarchical APA model, where
               HierBFR    0.082 0.639         0.591 0.300 0.609          the blue and orange points are the projected feature
    SSL                                                                  representations derived from the outputs of phoneme
               HierCB     0.076 0.680         0.630 0.355 0.645
              +PCO [12]   0.078 0.688         0.648 0.347 0.622
                                                                         encoder H$ and the phoneme-level textual embeddings E$ .
              +ConPCO     0.071 0.701         0.669 0.437 0.682        aspects of pronunciation assessments at both phoneme and
 Table 3: Experimental results of various methods on                   word granularities. Specifically, compared to Gradformer
 utterance-level pronunciation assessments.                            (GFR), 3M, and GOPT-SSL models, HierCB achieves average
                             Utterance Score (PCC)                     improvements of up to 2.9%, 3.6%, and 3.2% in terms of PCC
     Models
                  Acc.    Comp.    Fluency     Prosody       Total     scores, respectively. In regard to the phoneme-level regularizers
     3M [8]       0.760   0.325     0.753       0.760        0.742     PCO and ConPCO, we can observe that both regularizers can
   GOPT-SSL       0.748    0.290      0.817       0.807      0.778     enhance the phoneme accuracy of the HierCB and bring further
     HierCB       0.772    0.677      0.827       0.823      0.796     benefits to word-level assessments. Moreover, our ConPCO
    +ConPCO      0.780    0.749       0.830        0.823      0.803    method can significantly promote the performance of HierCB,
 Note: Accuracy and Completeness are abbreviated as Acc. and Comp.,    achieving the lowest phoneme-level MSE and delivering the
 respectively.                                                         best performance across all assessment tasks. Second, rendering
                                                                       on the large-scale pretrained acoustic models, GOPT-SSL
suggested in [23][24]. SSL-based features are extracted from           consistently excels the current prevailing approaches, including
three pretrained acoustic models including Wav2vec2.0 [25],            GOPT and LSTM, across pronunciation assessments of various
WavLM [26], and HuBERT [27], where the features are derived            aspects. Finally, in comparison with hierarchical APA models
from outputs of the last layer. These frame-level features are         (HierCB, HierBFR, and HiPAMA), HiPAMA largely falls
extracted and further aggregated into phoneme-level based on           behind the other two APA models, while HierCB exhibits
forced alignment. The multi-head attention layers used in              superior performance over HierBFR. This superiority probably
encoders and attention pooling mechanisms adopt 1 attention            stems from our tactfully designed convolution-augmented
head with 24 hidden units. As to the training configuration, we        Branchformer block, which effectively captures fine-grained
adhere to the settings reported in [12], which conducts 5              information from the SSL-features.
independent trials and each trial consists of 100 epochs with          Performance Evaluation at Utterance Level. Table 3 reports
different random seeds. In this work, the primary evaluation           on the experimental results for utterance-level pronunciation
metric is the Pearson correlation coefficient (PCC), which             assessments. The proposed HierCB outperforms those APA
measures the linear correlation between predicted scores and           models with a parallel architecture in all assessments, notably
ground-truth scores. The MSE value is reported for phoneme-            boosting completeness assessment by 0.352 and 0.387 for 3M
level accuracy.                                                        and GOPT-SSL, respectively. With the phoneme-level
3.1.2. Experimental Results                                            regularizer ConPCO, HierCB consistently improves the
                                                                       performance, highlighting the importance of capturing the
Qualitative Visualization of Feature Space. To qualitatively           hierarchical linguistic structures of an input utterance.
examine whether ConPCO aligns the phoneme representations
derived from an APA model with the embeddings of phoneme-                                       4. Conclusions
level text prompt, we obtain the learned representations H$ and
phoneme embeddings E$ from the proposed HierCB, which are              In this paper, we have proposed a novel training regime,
then projected and visualized via the t-SNE algorithm [28]. In         ConPCO, seeking to learn phoneme-aware representations
Figure 3, we demonstrate that the proposed regularizer                 while preserving the ordinal relationships among the regression
effectively projects these two types of phoneme representations        targets in the learned feature space. In addition, we also
into a shared feature space, exhibiting a more coherent                developed a hierarchical APA model to verify the efficacy of
distribution.                                                          the proposed regularizer. The practical utility of our method has
Performance Evaluation at Lower Granularities. Table 2                 been verified through extensive experiments on speechocen762
presents the experimental results at phoneme- and word-level           benchmark dataset.
granularities, divided into two parts: the first part reports on the   Limitations and Future Work. The proposed method is
results of the APA models that solely rely on GOP-features,            constrained by its dependence on the “reading-aloud” training
while the second part presents the results of the models using         scenario and to some extent lacks explainability for the
the SSL-feature as inputs. For fair comparisons, we reproduce          provided assessment results. In future work, we plan to examine
the GOPT model with the same input features X $ as our                 the proposed method on open-response scenarios, where
proposed approach (GOPT-SSL), and further report a variant of          learners speak freely or respond to a given task or question
HierCB where the encoder layers adopt the Branchformer                 [29][30]. In addition, the issues of explainable pronunciation
blocks (HierBFR). First, a general observation is that the             feedback are also left as a future extension.
proposed HierCB outperforms the previous APA models in all
                                                                          [15] A. Radford et al., “Learning transferable visual models from
5. References                                                                  natural language supervision,” in Proceedings of the International
[1]   S. Bannò, B. Balusu, M. Gales, K. Knill, and K. Kyriakopoulos,           Conference on Machine Learning (PMLR), vol. 139, pp. 8748–
      “View-specific assessment of L2 spoken English,” in Proceedings          8763, 2021.
      of Interspeech (INTERSPEECH), pp. 4471–4475, 2022.
                                                                          [16] B. Elizalde, S. Deshmukh, M. A. Ismail, and H. Wang, “Clap:
[2]   N. F. Chen, and H. Li, “Computer-assisted pronunciation training:        Learning audio concepts from natural language supervision,” in
      From pronunciation scoring towards spoken language learning,”            in Proceedings of the IEEE International Conference on Acoustics,
      in Proceedings of the Asia-Pacific Signal and Information                Speech and Signal Processing (ICASSP), pp. 1–5, 2023.
      Processing Association Annual Summit and Conference
                                                                          [17] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N
      (APSIPA ASC), pp. 1–7, 2016.
                                                                               Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”
[3]   P. M R.-Revell, “Computer-assisted pronunciation training                in Proceedings of the Conference on Neural Information
      (CAPT): Current issues and future directions,” RELC Journal, vol.        Processing Systems (NeurIPS), pp. 5998–6008. 2017.
      52, pp. 189–205, 2021.
                                                                          [18] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han,
[4]   K. Evanini, and X. Wang, “Automated speech scoring for                   S. Wang, Z. Zhang, Y. Wu, R. Pang, “Conformer: Convolution-
      Nonnative middle school students with multiple task types,” in           augmented transformer for speech recognition,” in Proceedings of
      Proceedings of Interspeech (INTERSPEECH), pp. 2435–2439,                 Interspeech (INTERSPEECH), pp. 5036–5040, 2020.
      2013.
                                                                          [19] J. Sakuma, T. Komatsu, and R. Scheibler, “MLP-based
[5]   Y. K. Singla, A. Gupta, S. Bagga, C. Chen, B. Krishnamurthy,             architecture with variable length input for automatic speech
      and R. R. Shah, “Speaker-conditioned hierarchical modelling for          recognition,” in arXiv preprint arXiv:2202.08456, 2022.
      automated speech scoring,” in Proceedings of ACM International
                                                                          [20] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language
      Conference on Information and Knowledge Management (CIKM),
                                                                               modeling with gated convolutional networks,” in Proceedings of
      pp. 1681–1691, 2021.
                                                                               the International Conference on Machine Learning (ICML), pp.
[6]   Z. Li, S. Lloyd, M. Beckman, and R. Passonneau., “Answer-state           933–941, 2017.
      recurrent relational network (AsRRN) for constructed response
                                                                          [21] W. Shang, K. Sohn, D. Almeida, and H. Lee, “Understanding and
      assessment and feedback grouping,” in findings of the
                                                                               improving convolutional neural networks via concatenated
      Association for Computational Linguistics: EMNLP (EMNLP-
                                                                               rectified linear units,” in Proceedings of the International
      findings), pages 3879–3891, 2023.
                                                                               Conference on Machine Learning (ICML), pp. 2217–2225, 2016.
[7]   J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li,
                                                                          [22] S. M. Witt and S. J. Young, “Phone-level pronunciation scoring
      D. Povey, and Y. Wang, “Speechocean762: An open-source non-
                                                                               and assessment for interactive language learning,” in Speech
      native English speech corpus for pronunciation assessment,” in
                                                                               Communication, pages 95–108, 2000.
      Proceedings of Interspeech (INTERSPEECH), pp. 3710–3714,
      2021.                                                               [23] C. Zhu, T. Kunihara, D. Saito, N. Minematsu, N. Nakanishi,
                                                                               “Automatic prediction of intelligibility of words and phonemes
[8]   F. A. Chao, T. H. Lo, T. I. Wu, Y. T. Sung, and Berlin Chen, “3M:
                                                                               produced orally by japanese learners of English,” in IEEE Spoken
      An effective multi-view, multigranularity, and multi-aspect
                                                                               Language Technology Workshop (SLT), pp. 1029–1036, 2022.
      modelling approach to English pronunciation assessment,” in
      Proceedings of the Asia-Pacific Signal and Information              [24] Y. Shen, A. Yasukagawa, D. Saito, N. Minematsu, and K. Saito,
      Processing Association Annual Summit and Conference                      “Optimized prediction of fluency of L2 English based on
      (APSIPA ASC), pp. 575–582, 2022.                                         interpretable network using quantity of phonation and quality of
                                                                               pronunciation,” in Proceedings of IEEE Spoken Language
[9]   H. Do, Y. Kim, and G. G. Lee, “Hierarchical pronunciation
                                                                               Technology Workshop (SLT), pp. 698–704, 2021.
      assessment with multi-aspect attention,” in Proceedings of the
      IEEE International Conference on Acoustics, Speech and Signal       [25] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0:
      Processing (ICASSP), pp. 1–5, 2023.                                      A framework for self-supervised learning of speech
                                                                               representations,” in Proceedings of the International Conference
[10] H. Do, Y. Kim, and G. G. Lee, “Score-balanced loss for multi-
                                                                               on Neural Information Processing Systems (NIPS), pp. 12449–
     aspect pronunciation assessment,” in Proceedings of Interspeech
                                                                               12460, 2020.
     (INTERSPEECH), pp. 4998–5002, 2023.
                                                                          [26] S. Chen et al., “Wavlm: Large-scale self-supervised pre-training
[11] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass,
                                                                               for full stack speech processing,” IEEE Journal of Selected Topics
     “Transformer-based multi-aspect multigranularity non-native
                                                                               in Signal Processing, pp. 1505–1518, 2022.
     English speaker pronunciation assessment,” in Proceedings of the
     IEEE International Conference on Acoustics, Speech and Signal        [27] W.-N. Hsu et al., “HuBERT: Self-Supervised Speech
     Processing (ICASSP), pp. 7262–7266, 2022.                                 Representation Learning by Masked Prediction of Hidden Units,”
                                                                               IEEE/ACM Transactions on Audio, Speech and Language
[12] B.-C. Yan, H.-W. Wang, Y.-C. Wang, J.-T. Li, C.-H. Lin, and B.
                                                                               Processing, pp. 3451–3460, 2021.
     Chen, “Preserving phonemic distinctions for ordinal regression:
     A novel loss function for automatic pronunciation assessment,” in    [28] L. van der Maaten and G. Hinton, “Visualizing data using t- sne,”
     Proceedings of the IEEE Automatic Speech Recognition and                  in Journal of Machine Learning Research (JMLR), vol. 9, pp.
     Understanding Workshop (ASRU), pp. 1–7, 2023.                             2579–2605, 2008.
[13] H.-C. Pei, H. Fang, X. Luo and X.-S. Xu, “Gradformer: A              [29] Y. Wang, M.J.F. Gales, K. M Knill, K. Kyriakopoulos, A. Malinin,
     framework for multi-aspect multi-franularity pronunciation                R. C van Dalen, M. Rashid, “Towards automatic assessment of
     assessment," in IEEE/ACM Trans. on Audio, Speech, and                     spontaneous spoken English,” Speech Communication, vol. 104,
     Language Processing, vol. 32, pp. 554–563, 2024.                          pages 47–56, 2018.
[14] Y. Peng, S. Dalmia, I. Lane, and S. Watanabe, “Branchformer:         [30] J. Park and S. Choi, “Addressing cold start problem for end-to-
     Parallel mlp-attention architectures to capture local and global          end automatic speech scoring,” in Proceedings of Interspeech
     context for speech recognition and understanding,” in                     (INTERSPEECH), pp. 994–998, 2023.
     Proceedings of the International Conference on Machine
     Learning (PMLR), pp. 17627-17643, 2022.
