Towards Efficient and Multifaceted Computer-assisted Pronunciation
   Training Leveraging Hierarchical Selective State Space Model
                and Decoupled Cross-entropy Loss

                                  Fu-An Chao, Berlin Chen
                  Department of Computer Science and Information Engineering
                              National Taiwan Normal University
                               {fuann, berlin}@ntnu.edu.tw


                     Abstract
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
    to this surging demand, we in this work first
    propose HMamba, a novel CAPT
    approach that seamlessly integrates APA
                                                     Figure 1: A running example depicts the evaluation
    and MDD tasks in parallel. In addition, we
                                                     differences between APA and MDD systems in the
    introduce a novel loss function, decoupled
                                                     reading-aloud scenario.
    cross-entropy loss (deXent), specifically
    tailored for MDD to facilitate better-
    supervised      learning    for    detecting     learning. In comparison with traditional curriculum
    mispronounced phones, thereby enhancing          learning, CAPT offers advantages in both time
    overall performance. A comprehensive set         efficiency and cost-effectiveness. More critically, it
    of empirical results on the speechocean762
                                                     shifts the conventional pedagogical paradigm from
    benchmark dataset demonstrates the
    effectiveness of our approach on APA.
                                                     teacher-directed to self-directed learning, thereby
    Notably, our proposed approach also yields       providing a stress-free environment for L2 learners
    a considerable improvement in MDD                (Eskenazi et al., 2009). In addition, CAPT
    performance over a strong baseline,              applications have achieved marked success in
    achieving an F1-score of 63.85%. Our             various commercial sectors and testing services,
    codes       are    made      available     at    such as the APPs of Duolingo (McCarthy et al.,
    https://github.com/Fuann/hma                     2021) and the SpeechRater (Zechner et al., 2009)
    mba                                              developed by Educational Testing Service (ETS).
                                                     Typically, a de-facto archetype system for CAPT
1   Introduction                                     encompasses a “reading-aloud” scenario, where a
                                                     non-native speaker is given a text prompt and
In this era of globalization and technologization,   instructed to pronounce it correctly. In this context,
computer-assisted pronunciation training (CAPT)      previous literature broadly divides applications of
systems have emerged as an appealing alternative     CAPT        into   two     categories:     automatic
to meet the pressing need for second language (L2)   pronunciation       assessment       (APA)        and
mispronunciation detection and diagnosis (MDD),             On these grounds, it is evident that both APA and
with each category dedicated to specific facets of       MDD are indispensable ingredients of CAPT,
pronunciation training. APA aims to evaluate the         playing complementary roles in its success.
spoken proficiency of L2 learners by providing           However, previous studies on APA and MDD
fine-grained feedback on various aspect                  appear to have developed independently, with
assessments (e.g., accuracy and fluency) across          limited research exploring their synergetic use. Ryu
multiple linguistic levels (e.g., word and utterance     et al. (2023) proposed a joint model for APA and
level) (Kheir et al., 2023). To assess L2 learners’      MDD, leveraging knowledge transfer and multi-
spoken proficiency, APA systems typically employ         task learning. Their findings indicate high negative
scoring models that are either jointly trained (Gong     correlations between several assessment scores and
et al., 2022; Chao et al., 2022) or leverage multiple    mispronunciations. This also suggests that the
regressors in an ensemble paradigm (Bannò et al.,        human assessors may be influenced by phonetic
2022a; Bannò and Matassoni, 2022b) to generate           errors when evaluating overall proficiency scores
scores across various aspects. As such, users can        for various aspects, which to some extent gives
receive multi-aspect assessment scores predicted         away the halo effect present in the human
by an APA system, as illustrated in the example          annotations. While the corresponding results show
shown in Figure 1. In contrast to APA, MDD               that jointly modeling both tasks can achieve better
focuses more on non-native speakers’ phonetic            performance than modeling each task in isolation,
pronunciation errors (Chen and Li, 2016). These          only utterance-level holistic assessments are
errors usually have clear-cut distinctions between       considered in their experiments. In order to provide
correct and incorrect ones, and can be easily            more comprehensive and fine-grained feedback for
quantified through deletions, substitutions, and         L2 learners, other granularities, such as the phone
insertions. For instance, a number of MDD models         level or the word level, should also be aptly
are designed to capitalize on classifier-based           modeled. Recognizing this importance, we propose
modeling (Truong et al., 2004; Strik et al., 2009;       HMamba, a more effective approach, for
Harrison et al. 2009), enabling precise                  multifaceted CAPT. Being aware of the linguistic
identification of the exact positions where              hierarchy, HMamba can capture the intrinsic multi-
pronunciation errors occur within an utterance.          layered speech structure, delivering both coarse
This capability provides L2 learners with specific       and fine-grained pronunciation assessments while
feedback on discrepancies between intended               offering more accurate diagnostic feedback of
pronunciation and actual pronunciation.                  mispronunciations. In addition, to address the extra
                                                         computational costs introduced by multi-task
   Albeit the phonetic (segmental) errors are            learning, HMamba leverages a selective state space
crucial in the beginning stages of non-native            model (SSM) that can efficiently tackle both APA
language learning, prosodic (suprasegmental)             and MDD tasks in parallel. The main contributions
errors may often cause a detrimental impact on the       of this paper can be summarized as follows:
perception of fluency and lead to poor intelligibility
(Chen and Li, 2016). This effect may be more             • We introduce HMamba, a unified and
pronounced in learning stress-timed languages like          linguistically hierarchy-aware model that jointly
English, especially for a learner whose mother              tackles APA and MDD tasks, achieving superior
tongue is a syllable-timed language, such as                overall performance compared to prior arts that
Chinese (Ding and Xu, 2016). To tackle this                 employ either single-task or multi-task models.
problem, APA can play a pivotal role by offering         • We propose a novel loss function, decoupled
prosodic or intonation assessment for L2 learners.          cross-entropy loss (termed deXent), which
For example, Lin et al. (2021a) introduced rhythm           effectively addresses the inherent issue of text
rubrics to predict the traits of sentence-level stress      prompt-aware MDD methods. Notably, deXent
in L2 English utterances, demonstrating a strong            is feasible and well-suited for optimizing the
correlation with the prosody scores assessed by the         MDD performance, particularly in striking the
human experts. In addition, Arias et al. (2010)             balance between precision and recall.
proposed text-independent systems for assessing          • To the best of our knowledge, this is the first
intonation and stress, focusing on measuring the            work to adopt and extend Mamba in the APA and
similarity between a test-taker’s intonation or stress      MDD tasks for a more efficient and
curve and that of a reference response.                     comprehensive CAPT application.
Figure 2: An overall architectural overview of HMamba, which consists of a bottom-up hierarchical modeling
structure with several Mamba blocks across three levels (viz. phone, word, and utterance levels) that can perform
multi-granular APA and MDD in parallel.

2     Methodology                                         states 𝐞 = {𝑒0 , 𝑒1 , … , 𝑒𝑁−1 } with respect to 𝐩
                                                          and in turn generate the correct diagnostic output
2.1    Problem Definition                                 𝐲 = {𝑦0 , 𝑦1 , … , 𝑦𝑁−1 } , where 𝑦𝑛 denotes the
Considering an input time sequence of speech              uttered phone of the learner corresponds to 𝑝𝑛 .
signal 𝐮 uttered by an L2 learner and a reference
text prompt 𝐩 that contains 𝑁 -length canonical           2.2    HMamba
phone sequence 𝐩 = {𝑝0 , 𝑝1 , … , 𝑝𝑁−1 }, we adopt
a set of feature extractors along with an aligner to      In this subsection, we shed light on the details of
extract an acoustic feature sequence 𝐗 =                  the proposed model, HMamba, which is devised as
{𝐱0 , 𝐱1 , … , 𝐱𝑁−1 } that aligned with 𝐩 from 𝐮 .        a hierarchical structure built upon the paradigm of
                                                          selective SSM. A schematic illustration of the
Our model aims to address APA and MDD tasks
                                                          complete architecture is depicted in Figure 2.
simultaneously but with separate processing flows:
                                                          Specifically, HMamba synthesizes the APA and
First, we define 𝐺 as a set of linguistic
                                                          MDD modules, each of which contains multiple
granularities, and for each granularity 𝑔 ∈ 𝐺 the
                                                          regressors and a classifier, respectively. These
model manages to predict a set of aspect scores
                                                          modules collectively generate the corresponding
𝐬𝑔 = {𝑠𝑔0 , 𝑠𝑔1 , … , 𝑠𝑔𝑀𝑔 −1 }, where 𝑀𝑔 refers to the
                                                          aspect score sequence 𝐬𝑔 for each linguistic
number of aspect scores of target granularity 𝑔. In       granularity 𝑔, as well as the phonetic error states 𝐞
this work, 𝐺 = {𝑔𝑝ℎ𝑛 , 𝑔𝑤𝑟𝑑 , 𝑔𝑢𝑡𝑡 }, where we have       and diagnosis 𝐲. Furthermore, each classifier and
granularities of 𝑔𝑝ℎ𝑛 (phone level), 𝑔𝑤𝑟𝑑 (word           regressor is implemented with a simple feed-
level), and 𝑔𝑢𝑡𝑡 (utterance level) for the APA task.      forward network (FFN) and jointly optimized
Meanwhile, the model also requires to detect error        through the training.
Acoustic Feature Extraction: In order to portray              extracted. Distinct from 𝐄𝑎𝑏𝑠 , 𝐄𝑟𝑒𝑙 denotes
the non-native speaker’s pronunciation quality,               relative positions of phones in a word using tokens
previous studies on either APA or MDD generally               such as begin [B], internal [I], end [E], and
adopt a pre-trained acoustic model to extract                 single-phone word [S] tokens. For special cases
goodness of pronunciation (GOP)-based features                of silence positions, we explicitly categorize them
(Witt and Young, 2000; Hu et al., 2015; Shi et al.,           as either long silence [LS] or short silence [SS].
2020). However, these features merely offer the               Following the guideline suggested by ETS
segmental-level information that may not capture              (Evanini et al., 2015), positions with a silence
prosodic errors of an L2 learner. Given this                  duration exceeding 0.495 seconds are assigned to
limitation, we first utilize a pre-trained acoustic           [LS]; otherwise, they are assigned to [SS].
model as an aligner to identify phone boundaries              Finally, all these embedding features are point-wise
(including silence), facilitating the extraction of           added to 𝐗 to obtain phone-level input features for
other prosodic features such as the phone duration            subsequent modeling:
and statistics of root mean squared energy (Dong et
al., 2024). To mitigate the low-resourced data                     𝐇𝑔0
                                                                      𝑝ℎ𝑛
                                                                            = 𝐗 + 𝐄𝑝ℎ𝑛 + 𝐄𝑎𝑏𝑠 + 𝐄𝑟𝑒𝑙         (3)
problem (Chao et al., 2022), we also consider other
self-supervised learning (SSL) features including             The details of the complete feature ablations (both
wav2vec 2.0 1 (Baevski et al., 2020), HuBERT 2                acoustic and phonological features) are shown in
(Hsu et al., 2021), and WavLM3 (Chen et al., 2022).           Appendix B.
All these features are concatenated and                       Mamba Blocks: To foster highly efficient multi-
subsequently projected through a linear layer to              task learning, we introduce selective SSMs instead
form a sequence of acoustic features 𝐗 . The                  of attention-based models such as the Transformer
transformation of each time step 𝑡 is given by                (Vaswani et al., 2017). Specifically, we adopt
                                                              Mamba (Gu and Dao, 2023) as our backbone
𝐚𝑡 = [𝐚𝑡𝑔𝑜𝑝 ; 𝐚𝑡𝑑𝑢𝑟 ; 𝐚𝑡𝑒𝑛𝑔 ; 𝐚𝑡𝑤2𝑣 ; 𝐚𝑡ℎ𝑏𝑡 ; 𝐚𝑡𝑤𝑙𝑚 ]   (1)
                                                              model structure in this work. Different from
                                                              previous SSM instantiations, Mamba features an
                𝐱𝑡 = 𝐖𝐚𝑡 + 𝐛                            (2)   input-dependent selection mechanism and a
where 𝐖 and 𝐛 are trainable parameters. Notably,              hardware-aware algorithm, allowing for efficient
a dropout rate of 10% is applied to all SSL features          input information filtering by dynamically
prior to the concatenation due to the discrepancy in          adjusting the SSM parameters based on the input
dimensionality between these and other features.              data. This also facilitate faster recurrent
                                                              computation of the model using scan. Nevertheless,
Phonological Feature Extraction: In addition to               the vanilla Mamba conducts causal computations
acoustic cues, a common practice in CAPT is to                in a unidirectional manner, which prevents it from
inject the phonological information by introducing            capturing global information as effectively as the
the reference text prompt features such as                    multi-head self-attention (MHSA) module
canonical phoneme embeddings (Gong et al.,                    involved in Transformer. To address this problem,
2022), context-aware sup-phoneme embeddings                   we explore a bidirectional variant of Mamba as the
(Chao et al., 2023), and vowel/consonant                      basic modeling block. In this approach, we replace
embeddings (Fu et al., 2021). In contrast to                  the MHSA module in the Transformer encoder
previous studies (Gong et al., 2022; Chao et al.,             with a bidirectional Mamba layer, as depicted in
2022; Do et al., 2023a), we extract the canonical             Figure 2. Specifically, for input 𝐇𝑔𝑖 to the Mamba
phoneme embeddings 𝐄𝑝ℎ𝑛 from 𝐩 using a phone                  block at granularity level 𝑔, the output 𝐇𝑔𝑖+1 of the
embedding layer that includes the silence (SIL)               block is:
information which has been shown to be crucial
when evaluating ones’ spoken proficiency.                     𝐇′𝑔𝑖 = BiMamba(LayerNorm(𝐇𝑔𝑖 )) + 𝐇𝑔𝑖 (4)
Additionally, an absolute positional embedding
𝐄𝑎𝑏𝑠 and a relative position embedding 𝐄𝑟𝑒𝑙 are                   𝐇𝑔𝑖+1 = FFN(LayerNorm(𝐇′𝑔𝑖 )) + 𝐇′𝑔𝑖 (5)

1                                                             3
  https://huggingface.co/facebook/wav2vec                      https://huggingface.co/microsoft/wavlm-
2-large-xlsr-53                                               large
2
  https://huggingface.co/facebook/hubert-
large-ll60k
where BiMamba denotes the bidirectional Mamba                                 𝑔𝑝ℎ𝑛
                                                            Subsequently, 𝐇 𝐿𝑝 are then propagated forward
layer and FFN refers to the feed-forward module,            into the APA module and the MDD module for
respectively. Notably, there are several studies            solving a regression and a sequence classification
investigating the bidirectional processing of               problem, respectively. The APA module contains
Mamba (Liang et al., 2024; Zhang et al., 2024;              one regressor that aims to predict the phone-level
Jiang et al., 2024). In this work, we use a similar                          𝑝ℎ𝑛

structure as Jiang et al. (2024) to implement the           aspect score 𝑠𝑔0 (accuracy). On the other hand,
bidirectional Mamba layer. For input 𝐍𝑔𝑖 from the           the MDD module comprises a classifier and a
output of layer normalization of 𝐇𝑔𝑖 to a                   softmax function that cooperatively learn a
bidirectional Mamba layer, the corresponding                distribution 𝑦̂𝑡 over the phoneme classes 𝐶 for
output 𝐌𝑔𝑖 is computed as follows:                          each time step 𝑡 . The diagnosis 𝑦𝑡 can then be
                                                            identified by applying the argmax function to 𝑦̂𝑡 . In
                 𝐙𝑔𝑖 = Linear(𝐍𝑔𝑖 )                   (6)   this work, we streamline the MDD task by treating
                                                            it as a process of free phone recognition (Li et al.,
𝐒𝑔𝑖 → = Linear(𝐍𝑔𝑖 ), 𝐒𝑔𝑖 ← = Flip(𝐒𝑔𝑖 → ) (7)
                                                            2015). As such, we can directly detect the
                  →                    →
              𝐂𝑔𝑖 = Conv1D→ (𝐒𝑔𝑖 )                          corresponding error state 𝑒𝑡 by comparing 𝑦𝑡 with
         !       ←              ←                     (8)   𝑝𝑡 , eliminating the need for a separate detection
              𝐂𝑔𝑖 = Conv1D← (𝐒𝑔𝑖 )
                                                                                                      𝑔𝑝ℎ𝑛
          𝑔𝑖 →         𝑔𝑖         →          𝑔𝑖 →
                                                            module. Meanwhile, the resulting 𝐇 𝐿𝑝 is served
        𝐎     = 𝜎(𝐙 ) ⨂ SSM (𝐂 )                                   𝑤𝑟𝑑
   {     𝑔𝑖 ←                                         (9)   as 𝐇𝑔0 for subsequent modeling.
        𝐎     = 𝜎(𝐙𝑔𝑖 ) ⨂ SSM← (𝐂𝑔𝑖 ← )
                                                               In word-level modeling, 𝐿𝑤 -layer Mamba
              1       1                                     blocks are first adopted and followed by a 1-D
 𝐌𝑔𝑖 = Linear( 𝐎𝑔𝑖 → + Flip(𝐎𝑔𝑖 ← )) (10)
              2       2                                     convolution layer to capture the local dependencies
where 𝐒𝑔𝑖 → and 𝐒𝑔𝑖 ← denote the forward and                (Lee, 2016). The reason for utilizing the
backward sequence features, respectively.                   convolution layer is that the convolution operation
Specifically, 𝐒𝑔𝑖 ← is derived from 𝐒𝑔𝑖 → by a              can accommodate different realizations of the same
flipping operation Flip(∙). Conv1D(∙), 𝜎(∙), and            underlying phone from various L2 speakers,
SSM(∙) represents the 1-D convolution, activation           thereby mitigating the temporal variability. The
                                                                                            𝑤𝑟𝑑
function, and selective SSM algorithm described in          word-level representations 𝐇𝑔𝐿𝑤 can be derived
Mamba (Gu and Dao, 2023), respectively.                     as follows:
                                                                   𝑤𝑟𝑑                             𝑤𝑟𝑑
Hierarchical Mamba: Since the speech signals                   𝐇′𝑔𝐿𝑤 = MambaBlock𝑤𝑟𝑑 (𝐇𝑔0 )                  (12)
are typically distinguished by the complex                           𝑤𝑟𝑑                        𝑤𝑟𝑑
hierarchical composition, prior studies (Do et al.,               𝐇𝑔𝐿𝑤 = Conv1D𝑤𝑟𝑑 (𝐇′𝑔𝐿𝑤 )                  (13)
2023a; Chao et al., 2023) have suggested that               To obtain word-level aspect scores, we put
hierarchical modeling structures is more amenable               𝑤𝑟𝑑
                                                            𝐇𝑔𝐿𝑤 into the word-level APA module which
than parallel modeling structures (Gong et al.,             contains three regressors to predict the word-level
2022). To capture the linguistic hierarchy while                             𝑤𝑟𝑑    𝑤𝑟𝑑    𝑤𝑟𝑑

retaining the cross-aspect relations within the same        aspect scores 𝑠𝑔0 , 𝑠𝑔1 , 𝑠𝑔2       (accuracy, stress,
linguistic unit, we design and instantiate our model        and total scores), respectively. To facilitate training
with a hierarchical structure and introduce Mamba           efficiency, we propagate the word score to each of
blocks to model the dependencies at each                    its phones during the training stage. In the inference
granularity level. More concretely, our approach            phase, we ensure consistency by averaging the
generates finer granularity scores at the lower             outputs corresponding to each word. In addition,
                                                                𝑤𝑟𝑑                 𝑢𝑡𝑡
layers and coarser granularity scores at the higher         𝐇𝑔𝐿𝑤 is viewed as 𝐇𝑔0 for further modeling.
layers, as exhibited in Figure 2. In phone-level                As for the utterance-level assessments, instead
                            𝑝ℎ𝑛
modeling, we first use 𝐇𝑔0 as the input into 𝐿𝑝 -           of prepending the [CLS] tokens to learn the
layer Mamba blocks to obtain the phone-level                utterance-level representation (Gong et al., 2022),
                                      𝑔𝑝ℎ𝑛                  we explore pooling-based approaches to aggregate
contextualized representations 𝐇 𝐿𝑝 :                       the hidden information. To this end, we utilize an
                                                            attention pooling layer similar to Peng et al. (2022).
       𝑔𝑝ℎ𝑛                             𝑝ℎ𝑛
   𝐇 𝐿𝑝 = MambaBlock𝑝ℎ𝑛 (𝐇𝑔0 )                      (11)    Specifically, assuming that a 𝑑-dimensional input
sequence to the attention pooling layer is
𝐡0 , 𝐡1 … , 𝐡𝑇 −1 , the pooling output is 𝐡 =
   𝑇 −1
∑𝑖=0 𝛼𝑖 𝐡𝑖 , where 𝛼𝑖 is calculated by
                 exp (𝐰𝑇 𝐪𝑖 /𝜏 )
        𝛼𝑖 = 𝑇 −1                        (14)
             ∑𝑗=0 exp (𝐰𝑇 𝐪𝑗 /𝜏 )

where 𝐰 is a learnable vector, 𝐪 is the
                           𝑝ℎ𝑛   𝑤𝑟𝑑   𝑤𝑟𝑑   𝑤𝑟𝑑
concatenated scores of [𝑠𝑔0 , 𝑠𝑔0 , 𝑠𝑔1 , 𝑠𝑔2 ],
and 𝜏 is a controllable temperature hyperparameter.
                                                        Figure 3: Difference between (a) the original cross-
The whole process of utterance-level modeling can
                                                        entropy loss and (b) the decoupled cross-entropy loss,
then be formulated by                                   given the text prompt “crime.”
             𝑢𝑡𝑡                         𝑢𝑡𝑡
      𝐇𝑔𝐿𝑢 = MambaBlock𝑢𝑡𝑡 (𝐇𝑔0 )                (15)
                                                        task, as illustrated in Figure 3. Specifically, we first
       𝑢𝑡𝑡                                 𝑢𝑡𝑡
                                                        decouple the original cross-entropy loss into two
  𝐡𝑔          = AttentionPooling𝑢𝑡𝑡 (𝐇𝑔𝐿𝑢 )      (16)   separate losses, one for mispronunciations and the
                          𝑢𝑡𝑡                           other for correct pronunciations:
After obtaining 𝐇𝑔𝐿𝑢 from 𝐿𝑢 -layer Mamba
blocks, 𝐡𝑔
            𝑢𝑡𝑡
                is derived through the attention                ℒ𝑚𝑖𝑠
                                                                 𝑋𝑒𝑛𝑡 = − ∑ log(𝑦̂𝑡 [𝑦𝑡 ])               (18)
                                                                              𝑡∈ℳ
pooling layer to predict the utterance-level aspect
            𝑢𝑡𝑡   𝑢𝑡𝑡   𝑢𝑡𝑡  𝑢𝑡𝑡    𝑢𝑡𝑡
scores 𝑠𝑔0 , 𝑠𝑔1 , 𝑠𝑔2 , 𝑠𝑔3 , 𝑠𝑔4       (accuracy,              ℒℎ𝑖𝑡
                                                                  𝑋𝑒𝑛𝑡 = − ∑ log(𝑦̂𝑡 [𝑦𝑡 ])              (19)
completeness, fluency, prosody, and total scores)                              𝑡∈ℋ
via an utterance-level APA module which contains        where ℳ and ℋ are mispronunciation and
five regressors corresponding to each score.            correct pronunciation positions, respectively, and
                                                        𝑦̂𝑡 [𝑦𝑡 ] is the predicted probability of the true label
2.3     Optimization
                                                        𝑦𝑡 at time step 𝑡. After obtaining two decoupled
Automatic Pronunciation Assessment Loss: In             losses, the proposed decoupled cross-entropy loss
the proposed model, each APA module is                  (deXent) is obtained by the following formulation:
optimized using Mean Square Error (MSE). The                                          𝜇ℎ 𝛼 𝑚𝑖𝑠
loss for multi-aspect multi-granular assessment,              ℒ𝑀𝐷𝐷 = ℒℎ𝑖𝑡    𝑋𝑒𝑛𝑡 + ( 𝑚 ) ℒ𝑋𝑒𝑛𝑡
                                                                                                          (20)
                                                                                      𝜇
ℒ𝐴𝑃𝐴 , is calculated by assigning weights to each       where 𝜇𝑚 and 𝜇ℎ denote the frequency of the
granularity level 𝑔:                                    mispronunciations and correct pronunciations in
                                  𝑁 −1
                                1 𝑔                     the training set, respectively, and 𝛼 controls the
       ℒ𝐴𝑃𝐴 = ∑ 𝜔𝑔 ∙               ∑ ∙ ℒ𝑔𝑘       (17)
                                𝑁𝑔 𝑘=0                  weight magnitude. After that, we use ℒ𝑀𝐷𝐷 to
                    𝑔∈𝐺
                                                        optimize the MDD module, and the overall loss
where 𝜔𝑔 and 𝑁𝑔 are the tunable hyperparameter          thus can be expressed by
and number of aspect scores at granularity level 𝑔,
                                                                  ℒ = ℒ𝐴𝑃𝐴 + 𝛽 ∙ ℒ𝑀𝐷𝐷                    (21)
respectively. ℒ𝑔𝑘 refers to the MSE loss computed
for 𝑘-th aspect score at granularity level 𝑔.           where 𝛽 is a tunable hyperparameter.
Mispronunciation Detection and Diagnosis Loss:          According to Equation 20, the proposed loss
To be in line with previous MDD studies, our            function can be viewed as one of the loss-balancing
model      incorporates    canonical     phoneme        methods, such as focal loss (Lin et al., 2017) and
embeddings to enhance text prompt-awareness.            class-balanced loss (Cui et al., 2019), to tackle the
Despite some performance improvements, the              imbalance issue in MDD. However, in most end-
mismatch between the L2 learner’s realized phones       to-end MDD methods, where the labels are phones
and canonical phones can still cause some               instead of mispronunciations (0 or 1s), directly
deteriorating effects. This discrepancy can             applying the existing loss-balancing methods on
introduce inaccurate predictions that may               phones is implicit and can be sub-optimal when we
potentially affect the overall quality of phonetic      aim to detect potential mispronunciations. Hence,
analysis. To mitigate this negative impact, we          we believe the proposed deXent provides a better
devise a new loss function tailored for the MDD         alternative to end-to-end MDD.
                           Phone Score      Word Score (PCC)                  Utterance Score (PCC)
      Model       Year
                          MSE↓ PCC↑ Accuracy↑ Stress↑ Total↑ Accuracy↑ Completeness↑ Fluency↑ Prosody↑ Total↑

Deep Feature      2021       -     -        -        -         -     -          -           -           -     0.720

HuBERT Large      2022       -     -        -        -         -     -          -         0.780       0.770     -

Joint-CAPT-L1     2023       -     -        -        -         -   0.719        -         0.775       0.773   0.743

LSTM              2022    0.089 0.591     0.514    0.294 0.531     0.720      0.076       0.745       0.747   0.741

GOPT              2022    0.085 0.612     0.533    0.291 0.549     0.714      0.155       0.753       0.760   0.742

3M                2022    0.078 0.656     0.598    0.289 0.617     0.760      0.325       0.828       0.827   0.796

HiPAMA            2023    0.084 0.616     0.575    0.320 0.591     0.730      0.276       0.749       0.751   0.754

3MH               2023    0.071 0.693     0.682    0.361 0.694     0.782      0.374       0.843       0.836   0.811

HMamba            2024    0.062 0.739     0.708    0.366 0.718     0.807      0.278       0.848       0.843   0.829

    Table 1: APA performance evaluations of our model and all strong baselines on the speechocean762 test set.

3     Experimental Setup                                   which was set to 9e-5. Other implementation
                                                           details are presented in Appendix A.
3.1     Dataset
We conducted experiments on speechocean762, a              Evaluation: The evaluation metrics employed
widely-used open-source dataset curated for APA            include the Pearson Correlation Coefficient (PCC)
and MDD research (Zhang et al., 2021). The                 and Mean Square Error (MSE) for the APA task.
dataset consists of 5,000 English-speaking                 On the other hand, we used precision, recall, F1-
recordings from 250 Mandarin L2 learners, divided          score, and phone error rate (PER) to evaluate the
evenly into training and test sets. For the APA task,      MDD performance, so as to be in accordance with
pronunciation proficiency scores were assessed at          prior studies. To ensure the validity of our
various linguistic granularities and across different      experimental results, we conducted 5 independent
pronunciation aspects. Each score is evaluated by          trials for each experiment, running 20 epochs with
five experts using standardized rubrics. For the           different seeds. The metrics for each task are
MDD task, the dataset provides an extra                    reported as the average of these trials.
mispronunciation transcription annotated using a
                                                           3.3     Compared Baselines
set of 46 phones. This set comprises 39 phones
from the CMU dictionary4, 6 L2-specific phones,            For APA, we compare our proposed approach,
and a [unk] token for unknown phones. Notably,             HMamba, with various cutting-edge baselines
there are no insertion errors in the utterances, and a     which can be categorized into two families: single-
[DEL] token is introduced to mark deletion errors          aspect pronunciation assessment models or multi-
of L2 learners. Therefore, the realized phones can         granular multi-aspect pronunciation assessment
be aligned with canonical phones in this dataset.          models. The first group includes the Deep Feature
                                                           (Lin et al., 2021b), HuBERT Large (Kim et al.,
3.2     Training and Evaluation Details
                                                           2022), and Joint-CAPT-L1 (Ryu et al., 2023). The
Training: We optimized the model with Adam and             second group encompasses LSTM, GOPT (Gong
a tri-phase rate scheduler (Baevski et al., 2020),         et al., 2022), 3M (Chao et al., 2022), HiPAMA (Do
where the learning rate was gradually increased            et al., 2023a), and 3MH (Chao et al., 2023). As for
during the first 40% of steps, held constant for the       MDD, we compare HMamba with Joint-CAPT-L1,
following 40%, and then linearly decayed for the           as to our knowledge it is the only attempt that
remaining steps. The initial learning rate was set to      jointly addresses the APA and MDD tasks with the
2e-3 except for the utterance-level APA module,            speechocean762 dataset.

4
 http://www.speech.cs.cmu.edu/cgi-
bin/cmudict
                  Phone Score          Word Score (PCC)                               Utterance Score (PCC)
      Model
                 MSE↓     PCC↑ Accuracy↑       Stress↑    Total↑    Accuracy↑    Completeness↑     Fluency↑ Prosody↑ Total↑

LMamba           0.071    0.694   0.678        0.299      0.689       0.790           0.234         0.844     0.838    0.816

PMamba           0.068    0.707   0.689        0.320      0.700       0.784           0.142         0.843     0.832    0.817

HMamba           0.062    0.739   0.708        0.366      0.718       0.807           0.278         0.848     0.843    0.829

                   Table 2: Performance comparison between different modeling structures.

                    Mispronunciations                                                         Mispronunciations
      Model                                       PER ↓              Loss       𝛼                                      PER ↓
                Precision ↑ Recall ↑    F1 ↑                                          Precision ↑ Recall ↑     F1 ↑

Joint-CAPT-L1    26.70%     91.40% 41.50%         9.93%            Xent          -     77.07%       38.60%    51.40%   2.53%

HMamba           64.35%     63.41% 63.85% 2.72%                                 0.3    70.06%       54.10%    61.04%   2.61%

Table 3: MDD performance evaluations of our model,                              0.5    67.12%       58.71%    62.62%   2.70%
compared with a representative multi-task approach                 deXent
                                                                                0.7    64.35%       63.41% 63.85%      2.72%
(Ryu et al., 2023) on the speechocean762 test set.
                                                                                0.9    57.74%       71.12%    63.73%   3.14%
4     Experimental Results and Discussion
                                                                   Table 4: Comparison of MDD performance between
4.1    APA Performance                                             the original cross-entropy loss (Xent) and proposed
                                                                   decoupled cross-entropy loss (deXent).
Overall Results: In Table 1, we compare the APA
performance of HMamba with other competitive
                                                                   incompletely articulated words. However, this
baselines, leading to several key observations.
                                                                   limitation could stem from the fact that HMamba
Firstly, it is notable that our approach, HMamba,
                                                                   focuses more on phoneme accuracy and
consistently outperforms all other methods on
                                                                   mispronunciation detection, rather than purely
nearly all assessment tasks, particularly in terms of
                                                                   evaluating word-level completeness.
accuracy scores at phone, word, and utterance
levels. This improvement stems from the joint                      Effects of Hierarchical Structure: To inspect the
modeling paradigm of APA and MDD,                                  hierarchical structure that influences on the APA
highlighting that pronunciation assessments can                    performance of the proposed approach, we
also benefit from phonetic error discovery,                        conducted an experiment to replace the hierarchical
consistent with prior research findings (Ryu et al.,               structure with two other variants, resulting in two
2023). In addition, by adopting SSL features,                      different models: LMamba and PMamba,
HMamba as well as other approaches like HuBERT                     respectively. LMamba has a similar structure to
Large, 3M, and 3MH, achieves significant                           HMamba but outputs all assessment scores in the
improvements over the other APA methods in                         last layers regardless of their granular differences.
terms of utterance-level assessments. In                           On the other hand, PMamba adopts the parallel
comparison to other hierarchical models such as                    structure suggested by previous studies (Gong et al.,
HiPAMA and 3MH, HMamba leverages an SSM                            2022; Chao et al., 2022) that use prepended [CLS]
structure instead of the Transformer structure,                    tokens to predict utterance-level scores. According
demonstrating superior performance on a variety of                 to the results shown in Table 2, HMamba
assessment tasks (further analysis between Mamba                   outperforms PMamba and LMamba across all
and Transformer are shown in Appendix C). In                       assessment aspects, highlighting the advantages of
assessing utterance completeness, while HMamba                     its hierarchical structure for the APA task. This
falls behind 3M and 3MH, it is on par with                         finding aligns with previous research (Chao et al.,
HiPAMA. According to Zhang et al. (2021), the                      2023). In addition, the significant performance
completeness refers to the percentage of the words                 gaps between HMamba and LMamba also suggest
that are actually pronounced. This may imply that                  that phone-level and word-level scores should be
our approach has a weaker ability to detect                        predicted in lower layers.
4.2   MDD Performance                                   5   Conclusion
Overall Results: We evaluate the MDD                    In this paper, we have presented a novel
performance of HMamba by comparing it with              hierarchical selective state space model (dubbed
another celebrated multi-task learning approach,        HMamba) for multifaceted CAPT application.
Joint-CAPT-L1. As shown in Table 3, despite of          Extensive experimental results substantiate the
lower recall rate, HMamba achieves a significant        viability and efficacy of the proposed method
improvement in terms of F1-score over Joint-            compared to several top-of-the-line approaches in
CAPT-L1, with an increase of 22.35%.                    terms of both the APA and MDD performance. In
Additionally, there is a marked reduction in PER by     future work, we envisage mitigating the issue of
7.21%.     These       substantial   enhancements       data imbalance from an optimization perspective.
demonstrate that HMamba not only delivers               In addition, another key area for future research
accurate pronunciation assessments but also             involves tackling the assessment of open-response
produces      more        robust    and      reliable   scenarios in CAPT.
mispronunciation detection and diagnosis results.
Effects of Decoupled Cross-entropy Loss: On the         Limitations
grounds of the distinct improvements in the MDD
performance, we further analyze the underlying          Lack of Accent Diversity. The dataset used in this
effects of our proposed decoupled cross-entropy         study comprises only Mandarin L2 learners, which
loss (deXent). As illustrated in Table 4, training a    would probably limit the generalizability of the
text prompt-aware MDD model using the original          proposed model. As a result, it might be
cross-entropy loss (Xent) often yields high             inapplicable when assessing L2 learners with
precision but low recall. This is because the model     diverse accents. This lack of accent diversity could
primarily relies on input canonical phones, leading     lead to biases and inaccuracies in pronunciation
it to predict prior phones and overlook the actual      assessment for learners from different linguistic
mispronunciations of a learner. Such a model may        backgrounds.
not be suitable for educational settings where
accurately detecting potential mispronunciations is     Limited Interpretability. The proposed model is
critical. To remedy this, the proposed deXent           designed to replicate expert annotations without
method sufficiently provides a feasible solution.       relying on manual assessment rubrics or external
By adjusting the weighting factor 𝛼, we can better      knowledge databases, which would make it
strike the balance between precision and recall,        challenging to provide clear and reasonable
thus optimizing the F1-score. This flexibility is       explanations for the assessment results. This
particularly vital in CAPT, where both over-            insufficiency of interpretability might hinder its
detection        and        under-detection       of    acceptance and trustworthiness among educators
mispronunciations can severely disrupt the              and learners who require transparent and justifiable
learning process—a challenge often neglected by         assessments.
most existing end-to-end methods. While adopting
deXent may result in a minor increase in PER, this      Limited Generalizability This research is
slight performance tradeoff is justifiable for the      centered on the “reading-aloud” pronunciation
significant gains in overall MDD effectiveness.         training scenario, where it is assumed that the L2
                                                        learner accurately pronounces a predetermined text
Limitations of Decoupled Cross-entropy Loss:            prompt. This would narrow the applicability of our
In Table 3, the MDD performance of HMamba is            models to other learning contexts, such as
reported based on maximizing the F1 score using         spontaneous speech or open-ended conversations.
the deXent, as we believe both the precision and
recall metrics are critical for MDD. However, a
                                                        Ethics Statement
potential limitation of using deXent as the loss
function is that, while it may help balance precision   We acknowledge that all co-authors of this work
and recall for MDD, it may not simultaneously           comply with the ACL Code of Ethics and adhere to
improve both metrics. This limitation likely stems      the code of conduct. Our experimental corpus,
from the close relationship between the                 speechocean762, is widely used and publicly
mispronunciation distribution and the loss-             available, and we believe there are no potential
balancing mechanism of deXent.                          risks associated with this work.
Acknowledgments                                         Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu
                                                          Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki
This work was supported by the Language Training          Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu,
and Testing Center (LTTC), Taiwan. Any findings           Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian
and implications in the paper do not necessarily          Wu, Michael Zeng, Xiangzhan Yu, and Furu Wei.
reflect those of the sponsor.                             2022. Wavlm: Large-scale self-supervised pre-
                                                          training for full stack speech processing. IEEE
References                                                Journal of Selected Topics in Signal Processing,
                                                          volume 16, number 6, pages 1505-1518.
Juan Pablo Arias, Nestor Becerra Yoma, and Hiram
   Vivanco. 2010. Automatic intonation assessment for   Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge
   computer aided language learning. Speech               Belongie. 2019. Class-balanced loss based on
   Communication, volume 52, pages 254–267.               effective number of samples. In Proceedings of the
                                                          IEEE/CVF Conference on Computer Vision and
                                                          Pattern Recognition (CVPR), pages 9268-9277.
Stefano Bannò, Bhanu Balusu, Mark Gales, Kate Knill,
   and Konstantinos Kyriakopoulos. 2022a. View-
   specific assessment of L2 spoken English. In         Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023a.
   Proceedings of the Annual Conference of the            Hierarchical pronunciation assessment with multi-
   International Speech Communication Association         aspect attention. In Proceedings of the IEEE
   (INTERSPEECH), pages 4471–4475.                        International Conference on Acoustics, Speech and
                                                          Signal Processing (ICASSP), pages 1–5.
Stefano Bannò and Marco Matassoni. 2022b.
   Proficiency assessment of L2 spoken English using    Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023b.
   wav2vec 2.0. In Proceedings of IEEE Spoken             Score-balanced loss for multi-aspect pronunciation
   Language Technology Workshop (SLT), pages 1088-        assessment. In Proceedings of the Annual
   1095.                                                  Conference of the International Speech
                                                          Communication Association (INTERSPEECH),
                                                          pages 4998–5002.
Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed,
  and Michael Auli. 2020. wav2vec 2.0: A framework
  for    self-supervised    learning   of    speech     Hongwei Ding, Xinping Xu. 2016. L2 English rhythm
  representations. In Proceedings of the Conference       in read speech by Chinese students. In Proceedings
  on Neural Information Processing Systems                of the Annual Conference of the International
                                                          Speech         Communication           Association
  (NeurIPS), pages 12449–12460.
                                                          (INTERSPEECH), pages 2696-2700.
Fu An Chao, Tien Hong Lo, Tzu I. Wu, Yao Ting Sung,
   Berlin Chen. 2022. 3M: An effective multi-view,      Bin Dong, Qingwei Zhao, Jianping Zhang, and
   multigranularity, and multi-aspect modeling            Yonghong Yan. 2004. Automatic assessment of
   approach to English pronunciation assessment. In       pronunciation quality. In Proceedings of IEEE
   Proceedings of the Asia-Pacific Signal and             International Symposium on Chinese Spoken
   Information Processing Association Annual Summit       Language Processing (ISCSLP), pages 137-140.
   and Conference (APSIPA ASC), pages 575–582.
                                                        Maxine Eskenazi. 2009. An overview of spoken
Fu-An Chao, Tien-Hong Lo, Tzu-I Wu, Yao-Ting Sung,        language technology for education. Speech
  Berlin Chen. 2023. A hierarchical context-aware         Communication, volume 51, pages 832–844.
  modeling approach for multi-aspect and multi-
  granular pronunciation assessment. In Proceedings     Keelan Evanini, Michael Heilman, Xinhao Wang, and
  of the Annual Conference of the International           Daniel Blanchard. 2015. Automated scoring for the
  Speech         Communication          Association       TOEFL Junior® comprehensive writing and
  (INTERSPEECH), pages 974–978.                           speaking test. ETS Research Report Series
                                                          2015(1):1–11.
Nancy F. Chen, and Haizhou Li. 2016. Computer-
  assisted pronunciation training: From pronunciation   Kaiqi Fu, Jones Lin, Dengfeng Ke, Yanlu Xie, Jinsong
  scoring towards spoken language learning. In            Zhang, and Binghuai Lin. 2021. A full text-
  Proceedings of the Asia-Pacific Signal and              dependent end to end mispronunciation detection
  Information Processing Association Annual Summit        and diagnosis with easy data augmentation
  and Conference (APSIPA ASC), pages 1–7.                 techniques. arXiv preprint arXiv:2104.08428.
Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang,         Aobo Liang, Xingguo Jiang, Yan Sun, Xiaohou Shi,
  and James Glass. 2022. Transformer-based multi-         and Ke Li. 2024. Bi-Mamba4TS: Bidirectional
  aspect multigranularity non-native English speaker      mamba for time series forecasting. arXiv preprint
  pronunciation assessment. In Proceedings of the         arXiv:2404.15772.
  IEEE International Conference on Acoustics,
  Speech and Signal Processing (ICASSP), pages          Binghuai Lin, Liyuan Wang, Hongwei Ding, Xiaoli
  7262–7266.                                              Feng. 2021a. Improving L2 English rhythm
                                                          evaluation with automatic sentence stress detection.
Albert Gu and Tri Dao. 2023. Mamba: Linear-time           In Proceedings of IEEE Spoken Language
  sequence modeling with selective state spaces.          Technology Workshop (SLT), pages 713-719.
  arXiv preprint arXiv:2312.00752.
                                                        Binghuai Lin and Liyuan Wang. 2021b. Deep feature
Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai,       transfer learning for automatic pronunciation
  Kushal Lakhotia, Ruslan Salakhutdinov, and              assessment. In Proceedings of the Annual
  Abdelrahman Mohamed. 2021. Hubert: Self-                Conference of the International Speech
  supervised speech representation learning by            Communication Association (INTERSPEECH),
  masked prediction of hidden units. IEEE/ACM             pages 4438–4442.
  Transactions on Audio, Speech, and Language
  Processing, volume 29, pages 3451–3460.               Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He,
                                                          Piotr Dollár. 2017. Focal Loss for Dense Object
                                                          Detection. In Proceedings of the IEEE International
Alissa M. Harrison, Wai-Kit Lo, Xiao-Jun Qian, and        Conference on Computer Vision (ICCV), pages
   Helen Meng. 2009. Implementation of an extended        2980-2988.
   recognition network for mispronunciation detection
   and diagnosis in computer-assisted pronunciation     Kun Li, Xiaojun Qian, Shiying Kang, Pengfei Liu, and
   training. In Proceedings of the Workshop on Speech     Helen Meng. 2015. Integrating acoustic and state-
   and Language Technology in Education (SLaTE),          transition models for free phone recognition in L2
   pages 45-48.                                           English speech using multi-distribution deep neural
                                                          networks. In Proceedings of the Workshop on
Wenping Hu, Yao Qian, Frank K. Soong, and Yong            Speech and Language Technology in Education
  Wang. 2015. Improved mispronunciation detection         (SLaTE), pages. 119-124.
  with deep neural network trained acoustic models
  and transfer learning based logistic regression       Arya D. McCarthy, Kevin P. Yancey, Geoffrey T.
  classifiers. Speech Communication, volume 67,           LaFlair, Jesse Egbert, Manqian Liao, and Burr
  pages 154–166.                                          Settles. 2021. Jump-starting item parameters for
                                                          adaptive language tests. In Proceedings of the
Xilin Jiang, Cong Han, and Nima Mesgarani. Dual-          Conference on Empirical Methods in Natural
   path mamba: Short and long-term bidirectional          Language Processing (EMNLP), pages 883–899.
   selective structured state space models for speech
   separation. 2024. arXiv preprint arXiv:2403.18257.   Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji
                                                           Watanabe. 2022. Branchformer: Parallel mlp-
Yassine Kheir, Ahmed Ali, and Shammur Chowdhury.           attention architectures to capture local and global
                                                           context for speech recognition and understanding.
  2023. Automatic pronunciation assessment - a
                                                           In Proceedings of the International Conference on
  review. In Findings of the Association for
                                                           Machine Learning (PMLR), pages 17627–17643.
  Computational Linguistics: EMNLP, pages 8304–
  8324.
                                                        Hyungshin Ryu, Sunhee Kim, and Minhwa Chung.
                                                          2023. A joint model for pronunciation assessment
Eesung Kim, Jae-Jin Jeon, Hyeji Seo, Hoon Kim. 2022.      and mispronunciation detection and diagnosis with
  Automatic pronunciation assessment using self-          multi-task learning. In Proceedings of the Annual
  supervised speech representation learning. In           Conference of the International Speech
  Proceedings of the Annual Conference of the             Communication Association (INTERSPEECH),
  International Speech Communication Association          pages 959-963.
  (INTERSPEECH), pages 1411–1415.
                                                        Jiatong Shi, Nan Huo, and Qin Jin. 2020. Context-
Ann Lee. 2016. Language-independent methods for            aware goodness of pronunciation for computer-
  computer-assisted pronunciation training, Ph.D.          assisted pronunciation training. In Proceedings of
  thesis, Massachusetts Institute of Technology.           the Annual Conference of the International Speech
  Communication Association         (INTERSPEECH),
  pages 3057-3061.

Helmer Strik, Khiet Truong, Febe De Wet, and Catia
  Cucchiarini. 2009. Comparing different approaches
  for automatic pronunciation error detection. Speech
  Communication, volume 51, number 10, pages 845-
  852.

Khiet Truong, Ambra Neri, Catia Cucchiarini, and
  Helmer Strik. 2004. Automatic pronunciation error
  detection:   an    acoustic-phonetic   approach.
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
   interactive   language      learning.    Speech
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
  Mamba in Speech: Towards an alternative to self-
  attention. arXiv preprint arXiv:2405.12609.

Junbo Zhang, Zhiwen Zhang, Yongqing Wang,
   Zhiyong Yan, Qiong Song, Yukai Huang, Ke Li,
   Daniel Povey, and Yujun Wang. 2021.
   Speechocean762: An open-source non-native
   English speech corpus for pronunciation assessment.
   In Proceedings of the Annual Conference of the
   International Speech Communication Association
   (INTERSPEECH), pages 3710 –3714.
                                                       APA                                                    MDD
                     Phone               Word                              Utterance
                                                                                                        F1↑     PER↓
                 MSE↓    Acc.↑   Acc.↑   Stress↑ Total↑    Acc.↑   Comp.↑ Fluency↑ Prosody↑ Total↑

HMamba           0.062   0.739   0.708   0.366     0.718   0.807   0.278    0.848      0.843   0.829   63.85%   2.72%

                                                   Acoustic Features

-wav2vec2        0.062   0.736   0.708   0.326     0.718   0.801   0.185    0.840      0.833   0.823   63.63%   2.79%

-HuBERT          0.063   0.735   0.706   0.344     0.715   0.804   0.216    0.843      0.838   0.825   63.49%   2.90%

-wavLM           0.063   0.731   0.705   0.355     0.715   0.806   0.247    0.844      0.838   0.827   63.39%   2.97%

-duration        0.063   0.734   0.705   0.341     0.715   0.804   0.299    0.844      0.838   0.826   63.70%   2.80%

-energy          0.063   0.735   0.706   0.358     0.716   0.802   0.257    0.840      0.834   0.823   63.28%   2.78%

-GOP             0.066   0.719   0.699   0.293     0.706   0.795   0.228    0.837      0.829   0.817   61.72%   2.79%

                                                 Phonological Features

-absolute-pos    0.063   0.735   0.706   0.332     0.715   0.802   0.261    0.843      0.838   0.825   63.48%   2.79%

-relative-pos    0.063   0.733   0.704   0.352     0.714   0.804   0.220    0.847      0.841   0.825   62.74%   2.81%

-canonical-phn   0.083   0.624   0.604   0.310     0.617   0.775   0.147    0.842      0.836   0.801   28.06%   14.55%

Table 5: Feature ablations of HMamba (MDD performance is reported with F1 and PER as representative metrics).


Appendix                                                      B     Feature Ablations

A Implementation Details                                      In Table 5, we conduct an ablation study on the
                                                              feature extraction to inspect the factors that
Feature Extraction: We adopt an open-source                   influence on APA. Specifically, we removed one
acoustic model 5 to extract GOP features, which               factor at a time to investigate the performance
also serves as an aligner for force alignment.                variations.
Subsequently, the phone-level duration, energy
statistics, and SSL features (average over time               Acoustic Features: According to the ablation
frames) are computed according to the alignment.              experiment, each acoustic feature used in this
The resulting acoustic features 𝑋 and all                     work contributes to specific aspect assessments.
embeddings are 128 dimensions. For all Mamba                  While the model may perform better in assessing
blocks, we set the number of hidden units to 128              utterance completeness without phone duration
and use a kernel size of 4 for the 1-D convolution.           features, the other aspect assessments and MDD
The SSM modules follow the original                           performance        decreases       synchronously.
                                                              Furthermore, among all acoustic features, GOP is
configuration used in Mamba. 𝐿𝑝 , 𝐿𝑤 , 𝐿𝑢 are set
                                                              the most crucial factor in relation to both of the
to 3, 1, 1, respectively. In addition, the word-level         APA and MDD performance.
1-D convolution has 256 kernels, each with a size
of 3.                                                         Phonological Features: As for the phonological
                                                              features, the canonical phoneme embeddings are
Hyperparameters setting: 𝜏 in attention pooling               the most critical features overall, particularly in
layer is set to 1.0. The combining weights 𝜔𝑔 for             MDD. Without canonical phoneme embeddings,
APA loss are uniformly set to 1.0 for each                    the performance dramatically degrades in F1
granularity level 𝑔. Parameters 𝛼 and 𝛽 are tuned             score and PER. Since the number of
to be 0.7 and 0.003, respectively.                            mispronunciations is typically far less than the

5
    https://kaldi-asr.org/models/m13
                                                     APA                                                      MDD
  Block Type       Phone                Word                               Utterance
                                                                                                        F1↑     PER↓
               MSE↓    Acc.↑    Acc.↑   Stress↑ Total↑    Acc.↑    Comp.↑ Fluency↑ Prosody↑ Total↑

Transformer    0.071   0.692    0.689   0.294    0.700    0.797    0.165    0.844      0.839   0.819   60.14%   3.50%


Mamba          0.062   0.739    0.708   0.366    0.718    0.807    0.278    0.848      0.843   0.829   63.85%   2.72%


               Table 6: Performance comparison between Mamba block and Transformer block.



                                    Block Type           Params(M)↓        MACs(G)↓



                               Transformer                 1.469            3.806


                               Mamba                       1.141            2.954


               Table 7: Computational efficiency between Mamba block and Transformer block.




        Figure 4: Comparison of the training curves for models equipped with Mamba and Transformer.

correct pronunciations, canonical phoneme                    C Mamba v.s. Transformer
embeddings can provide ample text-prompt
                                                             To validate the effectiveness of Mamba over
information to complement the acoustic cues. The
                                                             Transformer in different facets, we replace each
impact of canonical phoneme embeddings on APA
                                                             Mamba block in HMamba with a vanilla
is consistent with Gong et al., 2022, and we take
                                                             Transformer block (encoder only). In the
it a step further in this work by demonstrating that
                                                             following, we perform a set of qualitative
they are also pivotal for MDD.
                                                             analyses for comparisons of these two variant
                                                             structures.
Performance comparison: In the first
experiment, we compare the APA and MDD
performance by utilizing Mamba or Transformer
as a basic block, respectively. According to the
results shown in Table 6, Mamba consistently
outperforms Transformer across all assessment
tasks and MDD, especially in phone- and word-
level assessments. The key difference between
Mamba and Transformer is that Mamba leverages
1-D convolution to model the local context
dependency which has been shown crucial in
either APA (Do et al., 2023a) or MDD (Lee, 2016)
task. These findings also align with other research
in the speech community, such as speech
separation (Jiang et al., 2024) and speech
enhancement (Zhang et al., 2024).

Computational        efficiency:    We    further
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
advantage is less pronounced in the PCC curve for
the utterance total score, Mamba still surpasses
Transformer to a moderate extent.
