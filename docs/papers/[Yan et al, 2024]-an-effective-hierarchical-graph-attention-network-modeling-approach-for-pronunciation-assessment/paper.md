3974                                                             IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 32, 2024




  An Effective Hierarchical Graph Attention Network
  Modeling Approach for Pronunciation Assessment
                           Bi-Cheng Yan , Student Member, IEEE, and Berlin Chen , Member, IEEE


   Abstract—Automatic pronunciation assessment (APA) manages                     and informative feedback for L2 learners to improve their
to quantify second language (L2) learners’ pronunciation profi-                  speaking skills [2], [3], but also serve as a handy reference for
ciency in a target language by providing fine-grained feedback with              professionals (e.g., interviewers and examiners) on standard-
multiple aspect scores (e.g., accuracy, fluency, and completeness) at
                                                                                 ized tests to relieve their workload [4].As a crucial ingredient
various linguistic levels (i.e., phone, word, and utterance). Most of
the existing efforts commonly follow a parallel modeling frame-                  of CAPT, automatic pronunciation assessment (APA) aims to
work, which takes a sequence of phone-level pronunciation feature                quantify oral proficiency and provide fine-grained feedback to
embeddings of a learner’s utterance as input and then predicts                   learners by predicting multiple aspect scores at various linguistic
multiple aspect scores across various linguistic levels. However,                levels [5], [6]. An APA system is typically instantiated in a
these approaches neither take the hierarchy of linguistic units into             read-aloud scenario, where an L2 learner is presented with
account nor consider the relatedness among the pronunciation                     a text prompt and instructed to pronounce it correctly. Early
aspects in an explicit manner. In light of this, we put forward
an effective modeling approach for APA, termed HierGAT, which
                                                                                 studies for APA mostly focused on single-aspect assessment,
is grounded on a hierarchical graph attention network. Our ap-                   typically developed by extracting sets of hand-crafted features
proach facilitates hierarchical modeling of the input utterance as               to construct scoring modules accordingly, such as phone-level
a heterogeneous graph that contains linguistic nodes at various                  accuracy [7], [8], [9], word-level lexical stress [10], [11], or
levels of granularity. On top of the tactfully designed hierarchical             various aspects of utterance-level proficiency scores [12], [13],
graph message passing mechanism, intricate interdependencies                     [14]. Although these efforts possess the advantage of being
within and across different linguistic levels are encapsulated and               easily interpretable, they rely solely on surface features and
the language hierarchy of an utterance is factored in as well.
Furthermore, we also design a novel aspect attention module to                   postulate implicitly that scoring aspects of different linguistic
encode relatedness among aspects. To our knowledge, we are the                   levels are independent of each other, often leading to suboptimal
first to introduce multiple types of linguistic nodes into graph-based           performance. More recently, with the synergistic breakthroughs
neural networks for APA and perform a comprehensive qualita-                     in neural model architectures and optimization algorithms [15],
tive analysis to investigate their merits. A series of experiments               [16], research endeavors have been advocated for the notion
conducted on the speechocean762 benchmark dataset suggests the                   of multi-aspect and multi-granular pronunciation assessment,
feasibility and effectiveness of our approach in relation to several
                                                                                 which creates a unified scoring model to jointly evaluate pronun-
competitive baselines.
                                                                                 ciation proficiency at various linguistic levels (i.e., phone, word,
  Index Terms—Automatic pronunciation assessment (APA),                          and utterance) with diverse aspects (e.g., accuracy, fluency, and
computer-assisted pronunciation training, deep regression models,                completeness), as the running example depicted in Fig. 1. Prior
pre-training mechanism.                                                          arts along this line of research usually follow a parallel modeling
                                                                                 paradigm [17], [18], [19], wherein Transformer-based neural
                           I. INTRODUCTION                                       networks serve as the archetype to take as input a sequence
                                                                                 of phone-level pronunciation feature embeddings of a learner’s
         ITH the rising trend of globalization, an ever-growing
W        number of people are willing or being asked to learn
foreign languages. In response to this surging demand for for-
                                                                                 utterance while simultaneously predicting multiple aspect scores
                                                                                 across different linguistic levels without accounting for their
                                                                                 subtle dependency.
eign language learning, computer-assisted pronunciation train-
                                                                                    Albeit effective, such parallel modeling approaches suffer
ing (CAPT) systems have garnered significant research atten-
                                                                                 from at least two weaknesses. First, these approaches fall short
tion, as they can offer L2 (second-language) learners a range
                                                                                 in taking advantage of the hierarchical structure of an utterance,
of stress-free and self-directed scenarios to practicing pronun-
                                                                                 which assumes that all phones within a word are of equal
ciation skills [1]. Among other things, CAPT systems have a
                                                                                 importance and insufficiently capture the word-level structure
broad spectrum of applications, which not only provide timely
                                                                                 cues that are prominent in the composition of an utterance-level
                                                                                 representation when solely based on phone-level pronunciation
  Received 1 February 2024; revised 29 June 2024; accepted 9 August 2024.        features. Second, the relatedness among pronunciation aspects is
Date of publication 26 August 2024; date of current version 9 September 2024.
This work was supported by E.SUN Bank under Grant 202308-NTU-02. The
                                                                                 mostly sidelined. As an illustration, we visualize the correlation
associate editor coordinating the review of this article and approving it for    matrix in Fig. 2, which shows the Pearson Correlation Coeffi-
publication was Dr. Samuel Thomas. (Corresponding author: Berlin Chen.)          cient (PCC) between any pair of expert annotated aspect scores
  The authors are with the Department of Computer Science and Information        on the training set. We can observe that except for the aspects of
Engineering, National Taiwan Normal University, Taipei 11677, Taiwan (e-mail:
80847001s@ntnu.edu.tw; berlin@ntnu.edu.tw).                                      utterance-completeness and word-stress, the remaining aspects
  Digital Object Identifier 10.1109/TASLP.2024.3449111                           present strong correlations not only within the same linguistic

                        2329-9290 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
                                      See https://www.ieee.org/publications/rights/index.html for more information.


             Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
YAN AND CHEN: EFFECTIVE HIERARCHICAL GRAPH ATTENTION NETWORK MODELING APPROACH FOR PRONUNCIATION ASSESSMENT                                         3975



                                                                                 by information aggregation with a dedicated designed mecha-
                                                                                 nism of iterative intra-word, phone-word, inter-word, and word-
                                                                                 utterance message passing. Furthermore, in order to capture the
                                                                                 interactions among aspects, we also design an aspect attention
                                                                                 module [20] to resolve the relatedness among aspects of the
                                                                                 same linguistic level. Comprehensive experiments conducted on
                                                                                 the speechocean762 benchmark dataset show that our proposed
                                                                                 method achieves significant and consistent improvements over
                                                                                 several cutting-edge baselines [21].
                                                                                    Our main contributions of this work can be summarized as
                                                                                 follows:
                                                                                    1) To our knowledge, we are the first to construct a heteroge-
                                                                                        neous graph network for tackling the task of automatic pro-
                                                                                        nunciation assessment, characterizing an input utterance
                                                                                        by its constituent words and the corresponding phones.
                                                                                        The proposed heterogenous graph is able to capture several
                                                                                        types of relations, including utterance-word, word-word,
                                                                                        phone-word, and phone-phone ones.
                                                                                    2) Our proposed modeling framework is highly flexible and
Fig. 1. An example curated from the speechocean762 dataset [21] illustrates
the evaluation flow of an APA system in the reading-aloud training scenario,
                                                                                        can be easily extended to integrate other supra-segmental
which offers an L2 learner in-depth pronunciation feedback.                             linguistic units, such as syllables and intonational phrases.
                                                                                    3) Without resort to any external self-supervised pre-trained
                                                                                        feature extractors, our method is shown to outperform
                                                                                        current cutting-edge methods [22], [23], [24]. Ablation
                                                                                        studies and qualitative analysis confirm its effectiveness
                                                                                        in capturing hierarchical structure of an utterance.
                                                                                    The remainder of this paper is organized as follows. In
                                                                                 Section II, we review related work in the subfields of CAPT,
                                                                                 including mispronunciation detection and diagnosis, as well as
                                                                                 automatic pronunciation assessment. Next, we elaborate on the
                                                                                 proposed modeling framework in Section III. Sections IV and V
                                                                                 detail the experimental setup and results, respectively. Finally,
                                                                                 in Section VI, we conclude the paper with a discussion of our
                                                                                 findings and future work.

                                                                                                           II. RELATED WORK
Fig. 2. Correlation matrix on training set of expert annotations. Each element      Research and development on CAPT date back to pioneering
in the matrix reveals the PCC score of any pair of measuring aspects.            efforts conducted in the 60’s of the last century [7], [25], which
                                                                                 has attracted surging attention in recent years, showing good
                                                                                 promise by leveraging many advanced deep learning technolo-
level but also across different linguistic levels.1 Building on these            gies [26], [27], [28]. According to the types of diagnostic feed-
observations, we in this paper present a novel APA method,                       back being provided, research endeavors of CAPT fall into two
dubbed HierGAT, which leverages hierarchical graph attention                     broad categories: one is phone-level mispronunciation detection
architecture to jointly model the intrinsic structure of an utter-               and diagnosis (MDD), and the other is automatic pronunciation
ance and meanwhile considers the interactions among disparate                    assessment (APA).
aspects at the same and across different linguistic levels. By rep-
resenting an input utterance as a heterogeneous graph, HierGAT
updates and learns meaningful node representations for various                   A. Mispronunciation Detection and Diagnosis
linguistic units through message passing, reformulating the APA                     The goal of mispronunciation detection and diagnosis focuses
task into a node estimation problem. More specifically, HierGAT                  primarily on pinpointing phone-level erroneous pronunciation
first constructs a heterogeneous graph structured hierarchically                 segments and provide L2 learners with the corresponding diag-
with utterance, word, and phone levels, where each level con-                    nostic feedback [28], [29], [30]. Early work relies on pronuncia-
tains its corresponding types of nodes. Subsequently, HierGAT                    tion scoring based approaches, which make use of a well-trained
learns hierarchical representations for various linguistic units                 acoustic model to derive various types of confidence measure-
                                                                                 ments as indicators of mispronunciation. Commonly used indi-
                                                                                 cators include, but is not limited to, phone durations [32], [33],
   1 Both the aspects of utterance completeness and word stress suffer from      likelihood ratios [29], [34], phone posterior probabilities [35],
label imbalance problems, with more than 90% of the assessments receiving        and their combinations [39]. Goodness of pronunciation (GOP)
the highest score [18].                                                          and its descendants are the most iconic instantiations [7]. The

              Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
3976                                                           IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 32, 2024



principal idea behind GOP is to compute the ratio between the                  stress (e.g., unstressed, primary, or secondary) for each syllable
likelihoods of a canonical phone and the most likely pronounced                within a word [10]. Yet another approach is to frame lexical stress
phones predicted by an acoustic model via forced-alignment                     detection and pitch accent detection as a sequence labeling task
of the canonical phone sequence of a given text prompt to                      [47], which created a synergy of canonical lexical stress patterns
the speech signal uttered by a learner. A phone segment is                     and syllable-based prosodic features to evaluate the learner’s
identified as a mispronunciation if the corresponding likeli-                  speaking proficiency and immediately provides diagnostic feed-
hood ratio do not exceed a given phone-dependent threshold.                    back on syllable boundaries. Despite that the aforementioned
However, pronunciation scoring based methods are untenable                     ASR-driven methods have the advantage of being easily inter-
to provide specific diagnostic feedback for the mispronounced                  pretable, their performance is inevitably vulnerable to the errors
phone segments.                                                                made by ASR, potentially leading to an unfaithful rendering
   In order to better obtain informative diagnosis feedback,                   of the linguistic content inherent in a learner’s utterance. This
dictation-based methods alternatively frame MDD as a phone                     issue can be mitigated by replacing hand-crafted features with
recognition task by employing a free-phone recognition process                 automatically derived features, through either an one-stage [48],
to dictate the most likely phone sequence uttered by an L2                     [49], [50] or a multi-stage [51], [52] estimation process.
learner. Consequently, the erroneous pronunciation portions can                   Due to the unprecedented breakthroughs brought about by
be easily identified by comparing the dictation result with the                deep learning, the notion of multi-aspect and multi-granular
corresponding canonical phone sequence. To this end, for exam-                 pronunciation assessment has made inroad into APA with good
ple, Leung et al. made attempts to employ a phone recognizer                   promise. Several neural scoring models have been proposed to
trained with the connectionist temporal classification (CTC)                   jointly evaluate pronunciation proficiency at various linguistic
loss [36]. However, the conditional independence assumption                    levels with diverse aspects. For example, Lin et al. stream-
of the CTC loss may hinder the fidelity of dictation results.                  lined three linguistic-level scoring modules and introduced a
As a workaround, Yan et al. [37] exploited the hybrid CTC-                     single-aspect multi-granular hierarchical APA architecture, uti-
Attention ASR model as the dictation model and sought to                       lizing an attention mechanism to extract and aggregate linguistic
capture deviant (non-categorical) phone productions by aug-                    representations from low to high linguistic levels for multi-
menting the canonical phone dictionary. To integrate historical                granularity proficiency estimation [39]. Gong et al. proposed
mispronunciation patterns of L2 learners, Zhang et al. utilized                a GOP feature-based Transformer (GOPT) to jointly model
a phonetic recurrent neural network Transducer (RNN-T) to                      multi-aspect pronunciation assessment at multiple granularities
transcribe learners’ speech, which synergized RNN-T modeling                   with a multi-task learning mechanism [19]. Since then, several
with weakly supervised data augmentation and diversified beam                  subsequent extensions to the GOPT framework were developed.
search, so as to provide learners with comprehensive diagnostic                For example, Chao et al. integrated prosodic and self-supervised
feedback on erroneous pronunciation segments [38].                             learning (SSL) based features into GOPT to achieve multi-view,
                                                                               multi-granularity, and multi-aspect (3M) pronunciation mod-
                                                                               eling [17]. Do et al. investigated the issue of data imbalance
B. Automatic Pronunciation Assessment                                          incurred by APA and proposed a score-balanced loss function
                                                                               that aims to nudge the prediction bias of a neural model towards
    Automatic pronunciation assessment concentrates more on
                                                                               the majority scores (i.e., high-performing proficiency scores)
assessing and providing a suite of comprehensive pronunciation
                                                                               by assigning higher penalties when the predicted score belongs
scores on a few specific aspects or traits of spoken language
                                                                               to a minority class and vice versa [18]. Departing from the
usage to reflect a learner’s pronunciation quality [39], [40],
                                                                               aforementioned methods, we in this paper propose a novel
[41]. Prior arts on APA focused exclusively on the single-aspect
                                                                               hierarchical APA model based on a hierarchical graph attention
assessment, typically through constructing scoring modules in-
                                                                               Transformer architecture. By representing an input utterance as
dividually to predict a holistic pronunciation proficiency score
                                                                               a hierarchical graph, the proposed method updates and learns the
on a targeted linguistic level or some specific aspect with dif-
                                                                               node representations across several linguistic levels by message
ferent sets of hand-crafted features. These hand-crafted features
                                                                               passing, and aptly turns the pronunciation assessment task into
can be extracted directly from a learner’s input speech signal
                                                                               a node regression problem.
or the associated transcription generated by automatic speech
recognition (ASR), which may consist of acoustic features,
confidence of recognized linguistic units (phones, syllables, or                                         III. METHODOLOGY
words) [43], time-alignment information [44], and other statistic
measures such as fundamental frequency [45], speech rate, and                  A. Problem Formulation
filled pause [46]. As one of the pioneering attempts, Cucchiarini                 In this paper, we explore the task of multi-aspect and multi-
et al. utilized an ASR system to transcribe an input speech signal             granular automatic pronunciation assessment (APA), as illus-
and then derived various statistic measures related to phonation               trated in Fig. 1. Given an input utterance U which consists
quantity, like rate of speech, duration of pauses, and frequency of            of a sequence of audio signals X uttered by an L2 learner
filled pauses, from phone-level alignment to assess the fluency of             and a text prompt T that the learner is expected to pronounce
read speech [46]. Following a similar vein, Ferrer et al. quantified           correctly, the objective of APA is to estimate proficiency scores
word-level stress according to the time-alignment information at               for multiple aspects across various linguistic granularities. The
the syllable nucleus, where Gaussian mixture models were em-                   proposed model (dubbed HierGAT) represents an input utterance
ployed to represent the distributions of prosody- and spectrum-                as a hierarchical graph and formulates automatic pronunciation
related features, aiming to estimate possible manners of lexical               assessment as a node regression task.

           Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
YAN AND CHEN: EFFECTIVE HIERARCHICAL GRAPH ATTENTION NETWORK MODELING APPROACH FOR PRONUNCIATION ASSESSMENT                                                 3977




                                                                                    Fig. 4. An illustration of a hierarchical graph for an input text prompt com-
                                                                                    prising three types of nodes: an utterance node, word nodes, and phone nodes.



                                                                                    its constituent phone nodes, and Ewu describes cross-linguistic
                                                                                    connections between an utterance node and its constituent word
                                                                                    nodes. A schematic depiction of the hierarchical graph is illus-
                                                                                    trated in Fig. 4.
Fig. 3. The overall model architecture of HierGAT. We first construct a                Edge Connection: This hierarchical graph G is an unweighted
hierarchical graph for an input utterance and then learn hierarchy-aware rep-       graph; namely, the connected node pairs have weight 1, and
resentations for three linguistic level nodes (i.e., phone, word, and utterance
nodes). The learned representations of different linguistic nodes are then fed to   disconnected node pairs have weight 0 in the adjacency matrix
the corresponding regressors to access various aspect scores.                       A. For the phone-level connections, an edge epi,j connects phone
                                                                                    nodes vpi and vpj if they are within the same word, facilitating
                                                                                    the aggregation of intra-word information. All word nodes are
                                                                                    fully-connected by word-level edges which seeks to capture
   Formally, we denote a set of linguistic granularities as G =
                                                                                    inter-word information. For the cross-linguistic relations, the
{p, w, u}, where p, w, u stands for the phone, word, and ut-
                                                                                    phone-to-word edge epwi,k connects the phone node vpi to its
terance levels, respectively. For a linguistic level g ∈ G, our
                                                                                    corresponding word node vwk , enabling message passing from
APA model targets to quantify pronunciation skill of an L2
                                                                                    the phone nodes to word nodes. All word nodes are linked to
learner with respect to multiple aspects, represented by Ag =
                                                                                    an utterance supernode vu with word-to-utterance connections,
{ag1 , ag2 , . . . , agNg }, where Ng is the number of aspects, and each
                                                                                    thereby gathering information from the word nodes to an utter-
agj is framed as a regression task that estimates a sequence                        ance node. In the resulting hierarchical graph, each phone node
of aspect score yagj ∈ [0, 2]. The overall model architecture                       can only interact with neighboring phone nodes within the same
of HierGAT is depicted in Fig. 3, which mainly consists of                          word, while interacting indirectly with the phone nodes of other
three parts: 1) node representation initialization, which is re-                    words via word-level node connections.
sponsible for generating node features for phone-, word-, and
utterance-level units; 2) hierarchical graph layer, which learns
                                                                                    C. Node Representation Initialization
hierarchy-aware node representations with iteratively message
passing; 3) aspect assessments on nodes, where regressors are                          Pronunciation Feature Extraction: For an input utterance U,
built upon the learned node (aspect) representations to predict                     we start by converting the text prompt into a canonical phone
the corresponding proficiency score sequence.                                       sequence through looking up a pronunciation dictionary. Next,
                                                                                    various pronunciation features are extracted to assess the L2
                                                                                    learner’s pronunciation quality at the phone level, which are
B. Graph Construction
                                                                                    then concatenated and projected to obtain a sequence of dense
   For an input text prompt T with M words and N phones,                            pronunciation features Xp = (xp1 , xp2 , . . . , xpN ):
we first represent it as a hierarchical graph G = (V, E), where
V stands for the node set and E are edges between nodes. In                                         Xp = Wx · X̃p + bx ,                                     (1)
                                                                                                                     
order to utilize the linguistic structures of the text prompt, the                                  X̃p = [EGOP EEng  EDur ||EFbank ],                   (2)
undirected hierarchical graph G contains phone nodes, word
nodes, and an utterance node, defined by V = Vp ∪ Vw ∪ Vu ,                         where E   GOP
                                                                                                  is goodness of pronunciation-based (GOP) feature
where each phone node vpn ∈ Vp corresponds to a phone pn                            [7], [26], EDur and energy EEng are prosodic features of duration
in the canonical phone sequence of T, vwm ∈ Vw represents a                         and energy statistics, and spectral features EFbank (viz. log Mel-
word wm in T, and vu ∈ Vu is a special supernode that signifies                     filterbank features). Wx and bx are learnable parameters, and ||
the whole utterance. The edge connection of G is defined as                         denotes concatenation operation. Notably, the extracted pronun-
E = Ep ∪ Ew ∪ Epw ∪ Ewu , where Ep denotes the connections                          ciation features include both frame- and phone-level features. To
between phone nodes within a particular word, Ew stands for                         align with the phone-level features, the frame-level features are
the connections between word nodes within the text prompt,                          averaged over time frames based on aligned phone boundaries.
Epw is the cross-linguistic connections between a word node and                     In addition, the word-level pronunciation features are denoted by

               Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
3978                                                           IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 32, 2024



Xw = (xw1 , xw2 , . . . , xwM ), where xwm stands for the features             where σ is an activation function instantiated with rectified linear
of the m-th word, which is the sum of its constituent (connected)              units (ReLU), Ni is the set of neighboring nodes of vi , αij stands
phone-level pronunciation features.                                            for the attention weight between hi and hj , and Wa , Wq , Wk ,
   Node Representation Initialization: We explore to use a                     and Wv are trainable weight matrices. The multi-head attention
convolution-augmented Branchformer (ConvBFR) [56] to ini-                      can be expressed by
                                                                                                             ⎛                  ⎞
tialize node features at both the phone and word levels, with
                                                                                                                
the aim of capturing contextualized pronunciation patterns at                                  ui = ||Tt=1 σ ⎝        t
                                                                                                                     αij Wvt hj ⎠ ,            (10)
their respective granularities. Subsequently, the utterance-level                                                    j∈Ni
node is initialized by summing the features of its connected                                                                                     t
words. More specifically, the proposed ConvBFR comprises two                   where T is the number of independent attention mechanisms, αij
parallel branches to dynamically model various ranged contexts                 are normalized attention weights computed by the t-th attention
at different linguistic granularities, with one branch follow-                 mechanism, and Wvt is the corresponding transformation matrix.
ing the original Transformer network architecture employing                    Next, a residual connection is in turn employed to prevent
self-attention to capture long-range dependencies [53] and the                 gradient vanishing. The updated node representation hi can be
other branch utilizing a convolution module introduced in [55]                 denoted by
to capture local dependencies. Specifically, for the phone-level                                         hi = hi + Wo ui ,                   (11)
nodes, we first map the canonical phone sequence into phone                    where Wo is a linear projection adjusting the dimension of ui to
embeddings Ep via a phone and position embedding layer, which                  align with hi . Finally, stacking on each graph attention layer, we
are then point-wisely added to Xp to provide a rendition of                    introduce a position-wise feed-forward (FFN) layer consisting of
the positional information and phonetic characteristics. Next,                 two linear transformations, in the same vein as Transformer [15].
a phone encoder is followed to initialize the phone-level node                    Hierarchical Massage Passing: The proposed hierarchical
representations H̃p :                                                          graph layer begins by updating representations of phone nodes
                            0                                                  using their locally-neighboring phones within a word via the
                       H̃p = Xp + Ep ,                                 (3)
                                                                               intra-word message passing. Then, the intermediate representa-
                                    
                                                                               tions of a word node Hw are derived by gathering information
                       H̃p = PhnEnc H̃0p ,                             (4)
                                                                               from its constituent phone nodes:
                                                                                                                          
where PhnEnc(·) consists of 3 stacked ConvBFR blocks. Af-                                         Hp←p = GAT H̃p , H̃p ,                      (12)
terward, for the word-level nodes, Xw are enriched with the
textual information Ew , and then a word encoder is employed                                                                
to generate the initial node representations H̃w :                                                  Hw = GAT H̃w , Hp←p ,                    (13)
                        0                                                      where Hp←p is updated representations of phone nodes.
                     H̃w = Xw + Ew ,                                   (5)
                                                                             GAT(H̃p , H̃p ) denotes that H̃p is linear projected to form query,
                     H̃w = WordEnc H̃0w ,                              (6)     key, and value matrices, respectively, while GAT(H̃w , Hp←p )
                                                                               means that H̃w is used as query matrix, and Hp←p serves as the
where Ew is generated by passing the text prompt T into a word                 key and value matrices, respectively. To propagate information
and position layer, and WordEnc(·) encompasses a stack of 3                    from word nodes to the utterance node, we first perform inter-
ConvBFR blocks. For the utterance node representation h̃u , it                 word message passing to update the representations of word
is initialized by summing the representations of its connected
                                                                              nodes for capturing the interactions among words. The repre-
words h̃u = k∈Nu Xwk , where Nu is the set of the neighboring                  sentation of the utterance node is then refined by aggregating
word nodes of the utterance node vu .                                          information from its connected word nodes:
                                                                                                Hw←w = GAT (Hw , Hw ) ,                     (14)
D. Hierarchical Graph Layer                                                                                                   
   After constructing the hierarchical graph G with the adjacency                                hu←w = GAT h̃u , Hw←w ,                      (15)
matrix A and node representations at three linguistic levels                   where Hw←w and hu←w are updated representations of word
(H̃p ∪ H̃w ∪ h̃u ), we use the graph attention network (GAT)                   and utterance nodes, respectively. In GAT(h̃u , Hw←w ), h̃u acts
[54] to update the node representations.                                       as a query vector, and Hw←w is projected to construct the key
   Graph Attention Network: Given a constructed graph G with                   and value matrices. In this way, HierGAT updates and learns
the corresponding hidden representations of input nodes H, a                   hierarchy-aware node representations through the hierarchical
GAT layer updates a node vi with the representation hi as                      graph layer at three linguistic levels.2
follows:
            eij = LeakyReLU(Wa [Wq hi ||Wk hj ],                       (7)     E. Aspect Assessments on Nodes
                    exp (eij )                                                   The proposed HierGAT model is a unified architecture that can
           αij =                  ,                                   (8)     be optimized in an end-to-end manner using the mean square
                   l∈Ni exp (eil )
                   ⎛                 ⎞
                                                                                 2 The proposed hierarchical message passing can be easily generalized to

            ui = σ ⎝      αij Wv hj ⎠ ,                                (9)     k-hop message passing by iterating (11) to (15) k times. However, since the pro-
                                                                               posed hierarchical graph G is undirected, it may include redundant information
                            j∈Ni
                                                                               in the k-hop neighbors.


           Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
YAN AND CHEN: EFFECTIVE HIERARCHICAL GRAPH ATTENTION NETWORK MODELING APPROACH FOR PRONUNCIATION ASSESSMENT                                             3979



error (MSE) loss for each aspect at different linguistic levels.                                                 TABLE I
                                                                                               STATISTICS OF THE SPEECHOCEAN762 DATASET
Once the aspect representations are obtained, a fully-connected
layer acting as the regressor is in turn employed to calculate the
corresponding aspect score sequence.
   Aspect Assessments via Phone-level Nodes: For the phone-
level node aspect assessment, we first concatenate phone node
representations with their corresponding word node representa-
tions, which are then activated by the ReLU function to derive
the aspect representations Hp for phone nodes:
                 Hp = σ(Wp [Hp←p ||H w←w ]) ,                         (16)
  
H w←w is a sequence of augmented word-level node representa-
tions, repeated for each phone node based on the phone-to-word
connections. Next, the regression head is built on top of Hp to
access phone accuracy scores.
   Aspect Assessments via Word-level Nodes: For the word-level
node aspect assessments, the word node representations are
first concatenating with the average representations of their                   which are then passed through regression heads to derive the
constituent phone nodes:                                                        utterance-level proficiency scores.
                  Hw = σ(Ww Hw←w ||H̄w ] ,                             (17)        Model Optimization: The total loss is computed as a weighted
                                                                                sum of the MSE losses from different levels, where the loss
where H̄w = (h̄w1 , h̄ww , . . . , h̄wM ) with h̄wm being the av-               at each linguistic level is calculated as an average of multiple
erage vector of constituent phone-level representations derived                 aspects:
from Hp←p for the m-th word. Afterward, an aspect attention                                  λp                λw              λu 
mechanism is introduced to capture the relatedness among the                       LAP A =            Lpip +            Lwiw +          Luiu ,
                                                                                             Np i              Nw i             Nu i
aspects [20], [41]. Specifically, for the j-th word-level aspect, the                                  p                    w                      u

intermediate aspect representations H̃wj are linearly projected                                                                                (23)
from Hw , and a multi-head cross-attention (MHCA) with a                        where Lpip , Lwiw , and Luiu are phone-level, word-level, and
masking strategy is followed to derive word-level aspect rep-                   utterance-level losses at disparate aspects, respectively; λp , λw ,
resentations Hwj from a collection of all intermediate represen-                and λu are adjustable parameters controlling the influence of
tations Cw = [H̃w1 , H̃w2 , . . . , H̃wNw ] . The following equations           different granularities; and Np , Nw , and Nu refer to the numbers
illustrate the operations of aspect attention:                                  of aspects at phone, word, and utterance levels.
                  H̃wj = Wwj · Hw + bwj ,                              (18)
                                                                                                     IV. EXPERIMENTAL SETUPS
                                         
                  Hwj = MHCA H̃wj , Cw ,                               (19)     A. Experimental Data
where Wwj and bwj are aspect specific projection weights. In                       We conducted APA experiments on the speechocean762
the operation of MHCA, H̃wj is linearly projected as query                      dataset, a publicly available open-source dataset specifically
matrix, while Cw serves as key and value matrices. The masking                  designed for multi-aspect and multi-granular pronunciation as-
strategy ensures that the output representation at a specific                   sessment [21]. This dataset contains 5000 English-speaking
position is only influenced by the other aspects of the word.                   recordings spoken by 250 Mandarin L2 learners. The training
Lastly, the aspect representations Hwj are taken as the input to                and test sets are of equal size, each of which has 2500 utterances.
the corresponding regressor for evaluating the j-th word-level                  This corpus contains comprehensive annotation information,
pronunciation aspect.                                                           and the pronunciation proficiency scores were evaluated at mul-
   Utterance-level Node Aspect Estimations: For the utterance-                  tiple linguistic granularities alongside disparate aspects. Table I
level node aspect assessments, the node representations Hp←p                    summarizes the detailed statistics of used speech corpus. Each
and Hw←w are individually fed into an attention pooling mech-                   score was independently assigned by five experts using the same
anism to obtain holistic vector representations h̄p←p and h̄w←w                 rubrics, and the final score was determined by selecting the
at the phone and word levels, respectively. The utterance node                  median value from the five scores.
representation hu is then generated by packing these vectors
together via concatenation and projection:                                      B. Pronunciation Feature Extractions
      h̄p←p = AttPoolp (Hp←p ) ,                                       (20)        GOP Feature: To extract the GOP feature, we first aligned
                                                                                audio signals X with the text prompt T by using an ASR
      h̄w←w = AttPoolw (Hw←w ) ,                                       (21)     model3 to obtain the timestamps for each phone in the canonical
                                                                           phone sequence. Next, frame-level phonetic posterior proba-
         hu = σ Wu hu←w h̄p←p  h̄w←w + bu ,                        (22)     bilities were produced by the ASR model and then averaged
where σ is the ReLU function, and Wu and bu are trainable
parameters. After that, an aspect attention mechanism is per-                      3 A public-assessable ASR model trained with English speech corpus: https:
formed on hu to derive various aspect representations huj ,                     //kaldi-asr.org/models/m13.


             Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
3980                                                            IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 32, 2024



over time based on the phone-level timestamps. The resulting                    D. Compared Methods
phone-level posterior probabilities are converted into a GOP                       We first report the inter-annotator agreement for the five
feature vector as a combination of log phone posterior (LPP)                    annotators (Human-agreement), and compare the proposed
and log posterior ratio (LPR). Owing to the used ASR model                      model with the following top-of-the-line methods:
containing 42 phones, the GOP feature of a canonical phone p                       Lin2021 [14]: This method uses a single-aspect multi-
was thus represented by an 84-dimensional vector:                               granular pronunciation scorer with a hierarchical architecture
   [LPP (p1 ) , . . . , LPP (p42 ) , LPR (p1 |p) . . . , LPR (p42 |p)] ,        which takes phone-level surface features as inputs and assesses
                                                                                the learner’s utterance-level accuracy score.
                                                                       (24)        Kim2022 [28]: This approach employs a single-aspect pro-
                                                                                nunciation assessment model designed to separately measure
  LP P (pi ) = logp (pi |o; ts , te ) ,                                         oral skills on the utterance level. Each aspect-specific scorer
                                 t
                                                                                is implemented as a Bi-LSTM network, with the input fea-
                        1         e
                                                                                tures extracted from a self-supervised learning model (HuBERT
               =                     logp (pi |ot ) ,                  (25)
                   te − ts + 1 t =t                                             Large [23]).
                                     s
                                                                                   LSTM [19]: This method frames multi-aspect and multi-
  LPR (pi |p) = logp (pi |o; ts , te ) − logp (p|o; ts , te ) , (26)            granular pronunciation assessment as sequential labeling tasks,
                                                                                deriving a sequence of phone-level features and utilizing a
where LPR is the log posterior ratio between phones pi and p;                   3-layer LSTM to generate the representations across different
ts and te are the start and end timestamps of phone p, and ot is                linguistic units based on distinct timestamps.
the input acoustic observation of the time frame t.                                GOPT [19]: This model extends the sequential modeling
   Energy Feature: The energy feature is a 7-dimensional vector                 strategy by replacing the backbone model of LSTM with a
comprised of statistics (viz. [mean, std, median, mad, sum,                     3-stacked Transformer block and performs pronunciation as-
max, min]) over phone segments, where the root-mean-square                      sessment at various granularities with diverse aspects.
energy (RMSE) is employed to compute energy value for each                         Ryu2023 [40]: This method leverages is a unified model
time frame, with 25-millisecond windows and a stride of 10                      architecture that adopts a self-supervised model as the
milliseconds.                                                                   backbone model, which is optimized with phone recog-
   Duration Feature: The duration feature is a 1-dimensional                    nition and utterance-level pronunciation assessment tasks
vector indicating the length of each phone segment in seconds.                  jointly.
   Log Mel-filterbank Feature: The log Mel-filterbank feature is                   Gradformer (GFR) [42]: This model approaches multi-
an 80-dimensional vector computed over 25-millisecond win-                      aspect and multi-granular pronunciation assessment tasks with
dows with 10-millisecond strides, which are then averaged over                  a granularity-decoupled Transformer network, which decouples
each phone segment to from the corresponding phone-level                        the linguistic units of an utterance into two sub-groups: phone
feature.                                                                        and word levels, and utterance level. A Conformer encoder is
                                                                                employed to jointly model pronunciation aspects at phone and
                                                                                word levels, while a Transformer decoder takes a sequence of
C. Implementation Details
                                                                                aspect vectors as the input and interacts with the encoder outputs
   Model Configurations: In accordance with [19], we normal-                    for utterance-level pronunciation scoring.
ized utterance-level and word-level scores to the same scale as                    HiPAMA [41]: This model is built on top of a hierarchical
the phone-level score [0, 2] for training APA models. Both the                  architecture for multi-aspect and multi-granular pronunciation
feature encoders at the phone and word levels consist of three                  assessment, which more resembles our model in relation to all
blocks, each with a single-head attention mechanism and 24                      the other methods. In contrast to our model, HiPAMA extracts
hidden units. The proposed hierarchical graph layer consists of                 high-level pronunciation features from low-level features based
3 stack graph attention layers, each with a single attention head               on a simple average pooling mechanism. In addition, the aspect
and a hidden size of 24.                                                        attention mechanism used in HiPAMA performs on the internal
   Training Configurations: In the training phase, we use a batch               logistics, while our model operates on the intermediate repre-
size of 25 and apply Adam optimizer with a learning rate 1e-3. To               sentations.
ensure the reliability of our experimental results, we repeated 5                  3M [17]: This approach is a state-of-the-art APA model with
independent trials, each consisting of 100 epochs using different               a paralle pronunciation modeling technique, which enhances the
random seeds with a learning rate scheduler that warms up at                    input features of GOPT with three types of SSL-based features
the beginning and cuts in half every five epochs after the 20-th                to capture supra-segmental pronunciation cues, while also in-
epoch. The experimental results are reported by averaging 100                   tegrating vowel and consonant features to enhance phone-level
experiments with the minimum phone-level MSE values, where                      textual embeddings.
the mean and standard deviation values for different evaluation                    HierCB [56]: This approach is aslo a cutting-edge APA
metrics, as described below, are reported.                                      model with a hierarchical neural structure, stacking multiple
   Evaluation Metrics: The primary evaluation metric is PCC,                    ConvBFR blocks at three linguistic granularities (phone, word,
which measures the linear correlation between predicted scores                  and utterance) for pronunciation modeling and a combination
and ground-truth scores. In addition, mean squared error value                  of mean pooling and attention pooling mechanisms is further
(MSE) is used to assess phone-level accuracy.                                   employed to capture cross-granularity relationships.


            Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
YAN AND CHEN: EFFECTIVE HIERARCHICAL GRAPH ATTENTION NETWORK MODELING APPROACH FOR PRONUNCIATION ASSESSMENT                                          3981



                                                                                      style, rhythm, and intonation. Consequently, our model con-
                                                                                      siders multiple aspects when evaluating the total and prosody
                                                                                      aspects. Interestingly, the aspect of completeness is primarily
                                                                                      influenced by fluency and its own. Interestingly, the aspect of
                                                                                      completeness is primarily influenced by fluency and its own.
                                                                                      We postulate that our model reflects the halo effect present in
                                                                                      the human annotations on the training set. Specifically, when
                                                                                      assessing the completeness score, it seems that the decisions of
                                                                                      human annotators might be influenced by the score pertaining
                                                                                      to the prosody, making word-level pronunciation clarity maybe
                                                                                      psychometrically redundant.


                                                                                      B. Main Results
                                                                                         Table II presents the APA results on the speechocean762
                                                                                      dataset, organized into two groups, where the first group includes
Fig. 5. Score distributions for the aspects across different linguistic granular-
                                                                                      the results of models built upon the GOP-based features, while
ities on the training and test sets: (a) utterance-level aspects (accuracy, com-      the second group for other models utilizing the SSL-based
pleteness, fluency, prosody, and total score), (b) word-level aspects (accuracy,      features. Furthermore, for fair comparisons, we report on the
stress, and total score), and (c) phone-level accuracy score. The first row of each   performance of GOPT and HierGAT variants, where the in-
figure shows the distribution on the training set, while the second row shows the
distribution on the test set.                                                         put features of these models are enhanced by concatenating
                                                                                      GOP features with three types of SSL-based features (i.e.,
                                                                                      Wav2vec2.0, HuBERT, WavLM), following the processing flow
                                                                                      suggested in [17].
                      V. EXPERIMENTAL RESULTS                                            With respect to the models built on the GOP-based fea-
                                                                                      tures (the first group of Table II), we can make the following
A. Qualitative Analysis                                                               observations. First, on the whole, our model (HierGAT) con-
   Distributions of Aspect Scores: Before launching into a se-                        sistently outperforms human-human agreement on all assess-
ries of experiments on the APA tasks, we perform quantitative                         ment tasks, expect for the aspect of utterance-completeness.
analysis on the score distributions of aspects across different                       Second, Lin2020, a single-aspect assessment method, fails to
linguistic granularities on both the training and test sets. As                       harness the dependency between aspects through the multi-task
shown in Fig. 5, the speechocean762 is a well-curated dataset,                        learning scheme, resulting in inferior performance compared
where though the majority of aspect scores skew towards high                          to other multi-aspect and multi-granular pronunciation assess-
proficiency scores, the distributional trends are consistent be-                      ment models. Third, compared to the baseline methods with
tween the training and test sets. Furthermore, both the distri-                       the parallel modeling techniques, HierGAT excels on most as-
butions of utterance-completeness and word-stress demonstrate                         sessment tasks, particularly for assessments of higher linguistic
a notable high-score-biased phenomenon. The scores for these                          granularities (utterance and word levels), achieving average
two aspects are densely distributed on high-performing labels,                        improvements of 9.94%, and 8.28% over LSTM, and GOPT,
and a plurality of the data instances belong to a small number                        respectively. This performance gain underscores the significance
of labels. This label imbalance problem poses a challenge for                         of capturing the hierarchical structure of an utterance when
regression models to determine proficiency scores accurately.                         modeling cross-linguistic relationships with the proposed hi-
   Qualitative Visualization of Attention Weights in the Aspect                       erarchical graph layer. In terms of the hierarchical modeling
Attention Mechanisms: In the second set of experiments, we                            architecture, our model outperforms HiPAMA across various
examine the relatedness among disparate aspects at both word                          pronunciation assessment tasks with an average improvement of
and utterance levels on the training set by analyzing the attention                   up to 5.47%, while maintaining a top-performing phone-level
weights of the aspect attention mechanisms when assessing a                           accuracy score. HiPAMA generates high-level pronunciation
specific aspect score. Fig. 6(a) presents attention weights among                     features from phone-level pronunciation features with a simple
the word-level aspects, which reveals the attention weights for                       average operation, which can be seen as fully-connected rela-
the assessments on the accuracy and total aspects are influenced                      tionships with uniform weights. In contrast, our graph structures
by various other aspects. In contrast, the aspect of stress is a                      effectively prune unnecessary connections between phones or
specific evaluation task concerned with identifying emphasis on                       words when modeling cross-linguistic relationships. Finally, our
particular syllables within a word, resulting in attention weights                    model outperforms the state-of-the-art APA model, GFR, in
being focused on itself [52]. We then move to analyzing the                           most word- and utterance-level assessment tasks, achieving an
relatedness among the utterance-level pronunciation aspects.                          average PCC score improvement of 3.56%, while maintaining
As shown in Fig. 6(b), the attention weights for the prosody                          comparable results at the phone-level pronunciation aspect. As
and the total aspects are more uniformly distributed, whereas                         opposed to GFR, our model streamlines the pronunciation as-
the fluency aspect is primarily complemented by the prosody                           sessment tasks at three linguistic levels with tactfully designed
aspect. This could be attributed to the fact that the total and                       hierarchical architecture and a newly proposed aspect attention
prosody scores measure holistic oral skills, including speaking                       mechanism to capture the relatedness among aspects.

               Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
3982                                                             IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 32, 2024




Fig. 6. Qualitative visualization of model parameters when predicting each aspect score on the speechcoean762 training data set. We show (a) the averaged
attention values for word-level aspects, and (b) the averaged attention weights for utterance-level aspects.

                                                              TABLE II
            PERFORMANCE EVALUATIONS OF OUR MODEL AND ALL COMPARED METHODS ON SPEECHOCEAN762, WHERE ACC., AND COMP. REFER
                                      TO THE ASPECTS OF ACCURACY AND COMPLETENESS, RESPECTIVELY




   When we pair the GOP-based features with the SSL-based                        in the PCC score. Furthermore, compared to the cutting-edge
features, the assessment results are consistently improved across                hierarchical APA model, HierCB, our model significantly ex-
all pronunciation aspects. By combining the SSL-based features,                  cels for most assessments at word and utterance levels. This
HierGAT obtains average performance gains of 4.30%, 3.20%                        superiority stems from the efficacy of the proposed hierarchical
and 3.26% on phone, word, and utterance levels, respectively,                    graph attention layer in modeling cross-linguistic relationships.
compared to its base form. Second, as the focus is shifted to the
single-aspect pronunciation scorer, we find that Kim2022 addi-
tionally with the SSL-based features can boost the assessment                    C. Ablation Studies
results for the utterance-level pronunciation assessments, reach-                   To better understand the contributions of different modules
ing almost the same performance level with other multi-aspect                    to the performance of HierGAT, we conduct here a series of
and multi-granularity pronunciation scorers developed based on                   ablation studies for in-depth analysis. First, we compare the
the GOP-features.                                                                prediction distributions of HierGAT with different input features
   On a separate front, Ruy2023 further boots the performance                    thorugh boxplots for various aspects, as shown in Fig. 7. Next,
of utterance-level assessments by jointly training the APA model                 we remove different model components of HierGAT and report
with the MDD task. This highlights the potential of combining                    the corresponding PCC scores for the accuracy evaluations at
the APA and MDD tasks, as it encourages the APA model to pro-                    three granularities in Table III. Finally, we examine the learned
duce more phonetic-aware representations. Finally, compared                      weights in both the phone and word encoders when using the
to the existing APA models with parallel neural architectures,                   weighted combination mechanism in ConvBFR.
our model demonstrates remarkable performance improvements                          Comparison of Input Features: Fig. 7 shows the distribution
across various aspects at three granularities, outperforming                     of predicted scores estimated by HierGAT with different input
GOPT and 3M with lifts of 5.97% and 5.11%, respectively,                         features for various pronunciation aspects in the training set.


             Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
YAN AND CHEN: EFFECTIVE HIERARCHICAL GRAPH ATTENTION NETWORK MODELING APPROACH FOR PRONUNCIATION ASSESSMENT                                         3983




                                                                                Fig. 8. Visualization of branch weights with respect to various layers of
                                                                                ConvBFR at different linguistic levels (phone level and word level).


                                                                                additionally resorts to the SSL-based features, the corresponding
                                                                                predictions are more accurate for most aspects, resulting in
                                                                                smaller interquartile ranges compared to the base form of the
                                                                                model.
                                                                                   Comparison of Model Components: The first part of Table III
                                                                                presents an ablation study with the following settings: 1) replac-
                                                                                ing the concatenation operator with a weight average mechanism
                                                                                for merging two branches in both phone and word feature
Fig. 7. Boxplots of the HierGAT’s predictions for various pronunciation
aspects in the development set using different input features.
                                                                                encoders [53], 2) removing the aspect attention mechanism, and
                                                                                3) replacing the hierarchical graph layer with a simple attention
                             TABLE III                                          pooling. First, we can observe that the weighted average mech-
              THE ABLATION STUDIES ON SPEECHOCEAN762                            anism is slightly worse than the concatenation operator, where
                                                                                we see performance drops at phone and utterance levels and a
                                                                                modest improvement at the word level. Next, we notice the per-
                                                                                formance significantly declines at the utterance level and slightly
                                                                                drops at the word-level when the aspect attention mechanisms
                                                                                are removed from the proposed hierarchical architecture. The
                                                                                proposed aspect attention mechanism can effectively leverage
                                                                                the relatedness among aspects, as evident by the proportional
                                                                                decrease in performance corresponding to the number of aspects
                                                                                at different linguistic granularities. Finally, the employ of the
                                                                                hierarchical graph layer is indispensable for HierGAT, as the
                                                                                removal of such a layer leads to performance degrades for all
                                                                                linguistic granularities.
                                                                                   Depth and Width of GAT Layers: In the second and third parts
                                                                                of Table III, we investigate the impact on the performance of
                                                                                HierGAT when varying the width or depth of the GAT layer in
                                                                                the proposed hierarchical graph layer. We observe that the model
                                                                                performance is gradually improved as the number of layers
                                                                                increases; however, this improvement is limited. Meanwhile,
                                                                                there is a tendency of performance degradation when with an
                                                                                increase in the number of heads. One possible reason is that
                                                                                Speechocean762 by itself is not a large-scale dataset, and our
                                                                                model is capable of sufficiently learning the data characteristics
First, we focus on the predictions at lower-level granularities                 when equipped with a single GAT head. As such, to strike a
(viz. phone and word levels). In Fig. 7(a) and (b), we observe that             balance between performance and computational efficiency, the
the predicted scores of these two models (HierGAT and HierGAT                   proposed model is configured as a 3-layer GAT with a single
with SSL-based features) are more accurate for instances of                     attention head.
high and low annotation scores, which are tightly concentrated                     Learned Weights for Merging Operation in the Phone and
around the high and low score intervals but are more scattered                  Word Encoders: To examine the learned weights for merging
for the intermediate score interval. A possible reason is that the              two branches at the phone and word encoders, we visualize
annotated scores of training instances for phone-accuracy and                   the average weights on the training set while accessing pro-
word-accuracy are densely located at high scores, leading to the                nunciation aspects in the variant of HierGAT which replaces the
model’s predictions for instances with intermediate scores to                   concatenation operator with a weighted average mechanism in
be biased towards higher scores. Next, as we look at Fig. 7(c)                  the ConvBFR blocks. As shown in Fig. 8, we can observe several
and (d) for the utterance-level aspect assessments, we notice                   certain patterns in the learned weights. For example, in the initial
that the predicted scores for accuracy and total aspects have                   layers of the phone- and word-level ConvBFR modules, the two
the tendency of consistently decreasing from high to low scores                 types of branches are utilized in an interleaving fashion, with
for both models, which closely aligns with the score intervals                  the learned weights being distributed almost uniformly between
provided by human experts. This is evident that when the model                  the two branches. This indicates that both models use local

             Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
3984                                                                IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 32, 2024



and global relationships to learn hidden representations. In the                   [12] E. Coutinho et al., “Assessing the prosody of non-native speakers of
following layers of the phone encoder, the attention block and                          English: Measures and feature sets,” in Proc. Lang. Resour. Eval. Conf.,
                                                                                        2016, pp. 1328–1332.
the convolution block are utilized, showing that global context                    [13] C. Cucchiarini et al., “Quantitative assessment of second language learn-
and local relationships are equally important in the phone-level                        ers’ fluency by means of automatic speech recognition technology,” J.
modeling. On the other hand, the word encoder leveraging                                Acoustical Soc. Amer., vol. 107, no. 2, pp. 989–999, 2000.
consecutive attention blocks is observed, highlighting the im-                     [14] B. Lin and L. Wang, “Deep feature transfer learning for automatic pro-
                                                                                        nunciation assessment,” in Proc. Annu. Conf. Int. Speech Commun. Assoc.,
portance of global dependencies in word-level pronunciation                             2021. pp. 4438–4442.
modeling.                                                                          [15] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf.
                                                                                        Process. Syst., 2017, pp. 5998–6008.
                                                                                   [16] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretrain-
                            VI. CONCLUSION                                              ing of deep bidirectional transformers for language understanding,” in
                                                                                        Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics, 2019,
   In this paper, we have proposed HierGAT, a hierarchical                              pp. 4171–4186.
                                                                                   [17] F.-A. Chao, T.-H. Lo, T.-I. Wu, Y.-T. Sung, and B. Chen, “3M: An effec-
graph-based architecture for automatic pronunciation assess-                            tive multi-view, multigranularity, and multi-aspect modeling approach to
ment. Notably, we are the first to explore constructing a het-                          English pronunciation assessment,” in Proc. IEEE Asia-Pac. Signal Inf.
erogeneous graph network to streamline the three linguistic                             Process. Assoc. Annu. Summit Conf., 2022, pp. 575–582.
units for the pronunciation assessment. Evaluation on the spee-                    [18] H. Do, Y. Kim, and G. G. Lee, “Score-balanced loss for multi-aspect
                                                                                        pronunciation assessment,” in Proc. Annu. Conf. Int. Speech Commun.
chocean762 benchmark datasets proves the effectiveness of                               Assoc., 2023, pp. 4998–5002.
HierGAT and demonstrates capturing the language hierarchy                          [19] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass, “Transformer-based
and interactions between pronunciation aspects are beneficial to                        multi-aspect multigranularity non-native English speaker pronunciation
                                                                                        assessment,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process.,
the assessments.                                                                        2022, pp. 7262–7266.
   Limitations and Future Work: In this research, the proposed                     [20] R. Ridley, L. He, X.-Y. Dai, S. Huang, and J. Chen, “Automated cross-
model focus on the “reading-aloud” pronunciation training sce-                          prompt scoring of essay traits,” in Proc. AAAI Conf. Artif. Intell., 2021,
nario, where the assumption is that the L2 learner pronounces                           vol. 35, pp. 13745–13753.
                                                                                   [21] J. Zhang et al., “Speechocean762: An open-source non-native English
a predetermined target sentence correctly. This assumption re-                          speech corpus for pronunciation assessment,” in Proc. Annu. Conf. Int.
stricts the applicability of our models to other learning scenarios,                    Speech Commun. Assoc., 2021, pp. 3710–3714.
such as freely-speaking or open-ended conversations. We leave                      [22] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A
this for a future extension. We further plan to delve into resolving                    framework for self-supervised learning of speech representations,” in Proc.
                                                                                        Adv. Neural Inf. Process. Syst., 2020, pp. 12449–12460.
the data imbalance issue for the proposed model to enhance its                     [23] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and
generalizability to unseen learners.                                                    A. Mohamed, “HuBERT: Self-supervised speech representation learning
                                                                                        by masked prediction of hidden units,” IEEE Trans. Audio Speech Lang.
                                                                                        Process., vol. 29, pp. 3451–3460, 2021.
                                                                                   [24] S. Chen et al., “WavLM: Large-scale self-supervised pre- training for full
                                  REFERENCES                                            stack speech processing,” IEEE J. Sel. Topics Signal Process., vol. 16,
 [1] A. Van Moere and R. Downey, “Technology and artificial intelligence                no. 6, pp. 1505–1518, Oct. 2022.
     in language assessment,” in Handbook of Second Language Assessment.           [25] E. B. Page, “Statistical and linguistic strategies in the computer grading
     Boston, MA, USA: De Gruyter Mouton2016, pp. 341–357.                               of essays,” in Proc. Conf. Comput. Linguistics, 1967, pp. 1–13.
 [2] M. Eskenazi, “An overview of spoken language technology for education,”       [26] W. Hu, Y. Qian, F. K. Soong, and Y. Wang, “Improved mispronunciation
     Speech Commun., vol. 51, no. 10, pp. 832–844, 2009.                                detection with deep neural network trained acoustic models and transfer
 [3] K. Evanini and X. Wang, “Automated speech scoring for nonnative middle             learning based logistic regression classifiers,” Speech Commun., vol. 67,
     school students with multiple task types,” in Proc. Annu. Conf. Int. Speech        pp. 154–166, 2015.
     Commun. Assoc., 2013, pp. 2435–2439.                                          [27] Y. Qian et al., “Neural approaches to automated speech scoring of mono-
 [4] K. Evanini, M. C. Hauck, and K. Hakuta, “Approaches to automated                   logue and dialogue responses,” in Proc. IEEE Int. Conf. Acoust., Speech,
     scoring of speaking for K–12 English language proficiency assessments,”            Signal Process., 2019, pp. 8112–8116.
     ETS Res. Rep. Ser., vol. 2017, pp. 1–11, 2017.                                [28] E. Kim, J.-J. Jeon, H. Seo, and H. Kim, “Automatic pronunciation assess-
 [5] K. Li, X. Wu, and H. Meng, “Intonation classification for L2 English               ment using self-supervised speech representation learning,” in Proc. Annu.
     speech using multi-distribution deep neural networks,” Comput. Speech              Conf. Int. Speech Commun. Assoc., 2022, pp. 1411–1415.
     Lang., vol. 43, pp. 18–33, 2017.                                              [29] J. Shi, N. Huo, and Q. Jin, “Context-aware goodness of pronunciation
 [6] S. Banno, B. Balusu, M. J. F. Gales, K. M. Knill, and K. Kyriakopoulos,            for computer-assisted pronunciation training,” in Proc. Annu. Conf. Int.
     “View-specific assessment of L2 spoken English,” in Proc. Annu. Conf.              Speech Commun. Assoc., 2020, pp. 3057–3061.
     Int. Speech Commun. Assoc., 2022, pp. 4471–4475.                              [30] B.-C. Yan, H.-W. Wang, Y.-C. Wang, and B. Chen, “Effective graph-based
 [7] S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and                 modeling of articulation traits for mispronunciation detection and diag-
     assessment for interactive language learning,” Speech Commun., vol. 30,            nosis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2023,
     no. 2/3, pp. 95–108, 2000.                                                         pp. 1–5.
 [8] K. Li, X. Qian, and H. Meng, “Mispronunciation detection and diag-            [31] C. Richter and J. Guðnason, “Relative dynamic time warping comparison
     nosis in L2 English speech using multi-distribution deep neural net-               for pronunciation errors,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal
     works,” IEEE/ACM Trans. Audio Speech Lang. Process., vol. 25, no. 1,               Process., 2023, pp. 1–5.
     pp. 193–207, Jan. 2017.                                                       [32] Q.-T. Truong, T. Kato, and S. Yamamoto, “Automatic assessment of
 [9] S. Mao, F. Soong, Y. Xia, and J. Tien, “A universal ordinal regression             L2 English word prosody using weighted distances of F0 and intensity
     for assessing phone-level pronunciation,” in Proc. IEEE Int. Conf. Acoust.         contours,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2018,
     Speech, Signal Process., 2022, pp. 6807–6811.                                      pp. 2186–2190.
[10] L. Ferrer, H. Bratt, C. Richey, H. Franco, V. Abrash, and K. Precoda,         [33] C. Graham and F. Nolan, “Articulation rate as a metric in spoken language
     “Classification of lexical stress using spectral and prosodic features for         assessment,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2019,
     computer-assisted language learning systems,” Speech Commun., vol. 69,             pp. 3564–3568.
     pp. 31–45, 2015.                                                              [34] S. Sudhakara, M. K. Ramanathi, C. Yarra, and P. K. Ghosh, “An improved
[11] D. Korzekwa et al., “Detection of lexical stress errors in non-native (L2)         goodness of pronunciation (GOP) measure for pronunciation evaluation
     English with data augmentation and attention,” in Proc. Annu. Conf. Int.           with DNN-HMM system considering hmm transition probabilities,” in
     Speech Commun. Assoc., 2021, pp. 3915–3919.                                        Proc. Annu. Conf. Int. Speech Commun. Assoc., 2019, pp. 954–958.



              Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
YAN AND CHEN: EFFECTIVE HIERARCHICAL GRAPH ATTENTION NETWORK MODELING APPROACH FOR PRONUNCIATION ASSESSMENT                                                       3985



[35] S. Mao, Z. Wu, R. Li, X. Li, H. Meng, and L. Cai, “Applying multitask          [52] D. Korzekwa, J. Lorenzo-Trueba, T. Drugman, and B. Kostek, “Computer-
     learning to acoustic-phonemic model for mispronunciation detection and              assisted pronunciation training—Speech synthesis is almost all you need,”
     diagnosis in L2 English speech,” in Proc. IEEE Int. Conf. Acoust., Speech,          Speech Commun., vol. 142, pp. 22–33, 2022.
     Signal Process., 2018, pp. 6254–6258.                                          [53] Y. Peng, S. Dalmia, I. Lane, and S. Watanabe, “Branchformer: Parallel
[36] W.-K. Leung, X. Liu, and H. Meng, “CNN-RNN-CTC based end-to-end                     MLP-attention architectures to capture local and global context for speech
     mispronunciation detection and diagnosis,” in Proc. IEEE Int. Conf.                 recognition and understanding,” in Proc. Int. Conf. Learn. Representations,
     Acoust., Speech, Signal Process., 2018, pp. 8132–8136.                              2022, pp. 17627–17643.
[37] B.-C. Yan, M.-C. Wu, H.-T. Hung, and B. Chen, “An end-to-end mis-              [54] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio,
     pronunciation detection system for L2 English speech leveraging novel               “Graph attention networks,” in Proc. Int. Conf. Learn. Representations,
     anti-phone modeling,” in Proc. Annu. Conf. Int. Speech Commun. Assoc.,              2018.
     2020, pp. 3032–3036.                                                           [55] A. Gulati et al., “Conformer: Convolution-augmented transformer for
[38] D. Y. Zhang, S. Saha, and S. Campbell, “Phonetic RNN-transducer for                 speech recognition,” in Proc. Annu. Conf. Int. Speech Commun. Assoc.,
     mispronunciation diagnosis,” in Proc. IEEE Int. Conf. Acoust., Speech,              2020, pp. 5036–5040.
     Signal Process., 2023, pp. 1–5.                                                [56] B.-C. Yan, Y.-C. Wang, J.-T. Li, H.-W. Wang, W.-C. Chao, and B.
[39] B. Lin, L. Wang, X. Feng, and J. Zhang, “Automatic scoring at multi-                Chen, “ConPCO: Preserving phoneme characteristics for automatic pro-
     granularity for L2 pronunciation,” in Proc. Annu. Conf. Int. Speech Com-            nunciation assessment leveraging contrastive ordinal regularization,”
     mun. Assoc., 2020, pp. 3022–3026.                                                   2024, arXiv:2406.02859.
[40] H. Ryu, S. Kim, and M. Chung, “A joint model for pronunciation as-
     sessment and mispronunciation detection and diagnosis with multi-task
     learning,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2023,
     pp. 959–963.                                                                                              Bi-Cheng Yan (Student Member, IEEE) received the
[41] H. Do, Y. Kim, and G. G. Lee, “Hierarchical pronunciation assessment                                      M.S. degree in computer science and information
     with multi-aspect attention,” in Proc. IEEE Int. Conf. Acoust., Speech,                                   engineering in 2017 from National Taiwan Normal
     Signal Process., 2023, pp. 1–5.                                                                           University, Taipei, Taiwan, where he is currently
[42] H.-C. Pei, H. Fang, X. Luo, and X.-S. Xu, “Gradformer: A framework                                        working toward the Ph.D. degree in computer science
     for multi-aspect multi-granularity pronunciation assessment,” IEEE/ACM                                    and information engineering. He was with ASUSTeK
     Trans. Audio, Speech, Lang. Process., vol. 32, pp. 554–563, 2024.                                         Computer Inc., Beitou, Taiwan, from 2017 to 2020.
[43] P. Muller, F. De Wet, C. Van Der Walt, and T. Niesler, “Automatically                                     He is the author/coauthor of more than 20 academic
     assessing the oral proficiency of proficient L2 speakers,” in Proc. Workshop                              publications. His research interests include computer-
     Speech Lang. Technol. Educ., 2009, pp. 29–32.                                                             assisted language learning, speech recognition, and
[44] H. Franco et al., “EduSpeak: A speech recognition and pronunciation                                       speech enhancement.
     scoring toolkit for computer-aided language learning applications,” Lang.
     Testing, vol. 27, no. 3, pp. 401–418, 2010.
[45] K. Laskowski, J. Edlund, and M. Heldner, “An instantaneous vector
     representation of delta pitch for speaker-change prediction in conversa-                                 Berlin Chen (Member, IEEE) received the B.S. and
     tion dialogue system,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal                                  M.S. degrees in computer science and information
     Process., 2008, pp. 5041–5044.                                                                           engineering from National Chiao Tung University,
[46] C. Cucchiarini, H. Strik, and L. Boves, “Quantitative assessment of second                               Hsinchu, Taiwan, in 1994 and 1996, respectively, and
     language learners’ fluency by means of automatic speech recognition                                      the Ph.D. degree in computer science and information
     technology,” J. Acoust. Soc. Amer., vol. 107, no. 2, pp. 989–999, 2000.                                  engineering from National Taiwan University, Taipei,
[47] K. Li, S. Mao, X. Li, Z. Wu, and H. Meng, “Automatic lexical stress and                                  Taiwan, in 2001. He was with the Institute of Infor-
     pitch accent detection for L2 English speech using multi-distribution deep                               mation Science, Academia Sinica, Taipei, from 1996
     neural networks,” Speech Commun., vol. 96, pp. 28–36, 2018.                                              to 2001, and then with the Graduate Institute of Com-
[48] L. Chen, J. Tao, S. Ghaffarzadegan, and Y. Qian, “End-to-end neural net-                                 munication Engineering, National Taiwan University,
     work based automated speech scoring,” in Proc. IEEE Int. Conf. Acoust.,                                  from 2001 to 2002. In 2002, he joined the Graduate
     Speech, Signal Process., 2018, pp. 6234–6238.                                  Institute of Computer Science and Information Engineering, National Taiwan
[49] W. Liu et al., “An ASR-free fluency scoring approach with self-supervised      Normal University, Taipei. He is currently a Professor with the Department of
     learning,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2023,    Computer Science and Information Engineering of the same university. He is the
     pp. 1–5.                                                                       author/coauthor of more than 200 academic publications. His research interests
[50] K. Fu, S. Gao, S. Shi, X. Tian, W. Li, and Z. Ma, “Phonetic and prosody-       include speech recognition and natural language processing, multimedia infor-
     aware self-supervised learning approach for non-native fluency scoring,”       mation retrieval, computer-assisted language learning, and artificial intelligence.
     in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2023, pp. 949–953.
[51] S. Cheng, Z. Liu, L. Li, Z. Tang, D. Wang, and T. F. Zheng, “ASR-free
     pronunciation assessment,” in Proc. Annu. Conf. Int. Speech Commun.
     Assoc., 2020, pp. 3047–3051.




               Authorized licensed use limited to: Chi Ejimofor. Downloaded on March 03,2026 at 06:18:52 UTC from IEEE Xplore. Restrictions apply.
