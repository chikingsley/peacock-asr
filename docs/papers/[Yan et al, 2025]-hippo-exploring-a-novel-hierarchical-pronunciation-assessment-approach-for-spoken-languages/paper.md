    HiPPO: Exploring A Novel Hierarchical Pronunciation Assessment
                   Approach for Spoken Languages
                Bi-Cheng Yan1, Hsin-Wei Wang1, Fu-An Chao1, Tien-Hong Lo1,
                              Yung-Chang Hsu2, Berlin Chen1
                               1
                                   National Taiwan Normal University, 2EZAI
                                        {bicheng, berlin}@ntnu.edu.tw


                      Abstract
                                                                          • Assessment in the Reading-aloud Scenario
                                                                                         Reference Text:
     Automatic pronunciation assessment (APA)                        I like Europe. (Please read the reference text correctly)
     seeks to quantify a second language (L2)
     learner's pronunciation proficiency in a
                                                                                       (I like Europe.)
     target language by offering timely and fine-
                                                                                The fluency score for your
     grained diagnostic feedback. Most existing                                   scripted speech is 5.
     efforts on APA have predominantly                           (L2 Learner)                                   (APA Model)
                                                                        (a) Appling APA Model to the Scripted Speech
     concentrated on highly constrained
                                                                        • Assessment in the Free-speaking Scenario
     reading-aloud tasks (where learners are                                    Reference Text (Optional):
     prompted to read a reference text aloud);                      Have you been to Europe? What would you do there?

     however, assessing pronunciation quality in
     unscripted speech (or free-speaking                                            (I’d like to buy a bag.)
     scenarios)         remains          relatively                                Sorry, please read the
     underexplored. In light of this, we first                   (L2 Learner)
                                                                                  reference text out loud.
                                                                                                                (APA Model)
     propose       HiPPO,        a    hierarchical                 (b) Grafting APA Model to Evaluate Unscripted Speech

     pronunciation assessment model tailored
     for spoken languages, which evaluates an
                                                                                   (Oh, the City of Music!)
     L2 learner’s oral proficiency at multiple                                                                    ASR Trans
     linguistic levels based solely on the speech                               You said “the city of music”.
                                                                                  The fluency score is 3.
     uttered by the learner. To improve the                      (L2 Learner)                                      (HiPPO)
                                                                   (c) Employing HiPPO to Evaluate Unscripted Speech
     overall accuracy of assessment, a
     contrastive ordinal regularizer and a                Figure 1: Outlines our motivations. (a) Existing
     curriculum learning strategy are introduced          APA models are primarily tailored for read-aloud
     for model training. The former aims to               tasks. (b) Directly applying APA models to free-
     generate score-discriminative features by            speaking scenarios struggles to quantify oral skills
     exploiting the ordinal nature of regression          based on speech signals. (c) HiPPO integrates a
     targets, while the latter gradually ramps up         speech recognizer to generate transcriptions from
     the training complexity to facilitate the            the learner’s speech, effectively reformulating free-
     assessment task that takes unscripted                speaking assessment as a task akin to read-aloud.
     speech as input. Experiments conducted on
     the Speechocean762 benchmark dataset               Davis, 2025; Moere and Downey, 2016). To meet
     validates the feasibility and superiority of
                                                        this pressing demand, CAPT systems have become
     our method in relation to several cutting-
                                                        ubiquitous and appealing learning tools,
     edge baselines.
                                                        transitioning the conventional pedagogical
1    Introduction                                       approach from teacher-led instruction to self-
                                                        directed learning (Rogerson-Revell, 2021; Chen
Spurred by the global demand for foreign language       and Li, 2016; Singla et al., 2021).
proficiency in both the workforce and academia,             Automatic pronunciation assessment (APA)
computer-assisted pronunciation training (CAPT)         aims to evaluate L2 learners’ speaking proficiency
has gained significant attention, which facilitates     and provide fine-grained feedback on specific
second-language (L2) learners to practice               pronunciation aspects pertaining to a target
pronunciation skills with near-instant, instructive,    language, figuring prominently in the field of
and potentially diagnostic feedback (Norris and         CAPT. Prior studies on APA have primarily drawn
attention to highly constrained speaking tasks           articulatory traits across multi-granular linguistic
(such as listening and then repeating words or           units, HiPPO capitalizes on a tailor-made Conv-
sentences). As exemplified in Figure 1(a), a de-         LLaMA block to stack a hierarchical neural
facto archetype system for APA is instantiated in        architecture, which augments the LLaMA block
reading-aloud (or scripted) learning scenarios,          (Touvron et al., 2023) with a convolutional branch
where an L2 learner is provided with a reference         and rotary position encoding (Su et al., 2024).
text and instructed to pronounce it correctly.           Moreover, during training, a contrastive ordinal
Methods in this line of research typically rely on an    regularizer is put forward to modulate feature
input reference text paired with the learner’s           distances through the absolute differences between
speech to derive timestamps of linguistic units (i.e.,   regression targets. By exploiting the ordinal
phones or words) via an automatic speech                 constraints, the proposed regularizer serves as a
recognition (ASR) system, which are then used for        promising approach to generate score-
either pronunciation feature extraction (Gong et al.,    discriminative features, mitigating the detrimental
2022; Chao et al., 2022; Do et al., 2023; Yan et al.,    effects of ASR errors on pronunciation assessments.
2024) or for neural modeling (Lin and Wang, 2021;        We further introduce a simple yet effective
Wang et al., 2025). Albeit achieving competitive         curriculum learning strategy for HiPPO that
performance in relation to inter-rater agreement         progressively increases the training complexity,
(Yan and Chen, 2024; Pei et al., 2024), scripted-        transforming the assessment tasks from the read-
speech assessments fail to reflect learners’             aloud scenario to the free-speaking counterpart. An
speaking abilities in real-world communication. In       extensive set of experiments conducted on
contrast, pronunciation assessment of spoken             Speechocean762 benchmark dataset (Zhang et al.,
languages introduces new challenges to CAPT, as          2021), consisting of both read-aloud and simulated
it attempts to quantify an L2 learner’s oral skills in   free-speaking scenarios, demonstrates substantial
spontaneous speech or elicit authentic responses         and consistent performance gains of the proposed
through short questions (Zechner and Evanini,            methods over several strong baselines.
2019; Kheir et al., 2023). Directly grafting existing        In summary, our contributions are at least four-
APA models to use cases of spoken language               fold: (1) to our knowledge, HiPPO is the first
assessment, however, confronts at least two major        attempt to assess oral skills for unscripted speech
issues. First, as shown in Figure 1(b), the utterances   with multi-faceted scores from phone to utterance
of an L2 learner are produced in an unscripted           levels, opening a new avenue for CAPT; (2) we
manner, which makes APA models struggle to               propose a novel Transformer block, Conv-LLaMA
extract      correct      pronunciation       features   block, as the backbone of HiPPO, elaborately
encompassing time-alignment information (Shen            designed to handle the free-from speech uttered by
et al., 2021; Deng et al., 2020; Witt and Young,         L2 learners; (3) to alleviate the negative effects of
2000). What is more, owing to the free-form nature       ASR errors, a contrastive ordinal regularizer is
of unscripted speech, the desired APA models are         proposed to reflect the ordinality of regression
required to accommodate speech input of varying          targets within the feature space; and (4) a simple
lengths.                                                 yet effective curriculum learning strategy is
    Building on these observations, this paper           explored to boost the performance of pronunciation
presents HiPPO, a novel hierarchical pronunciation       assessment in the free-speaking scenario.
assessment model for spoken languages that
evaluates L2 learners’ oral proficiency based on         2   Methodology
unscripted speech (or free-speaking scenarios) and
provides analytical scores on various pronunciation      This section sets out with a problem definition for
aspects across multi-granular linguistic levels.         pronunciation assessments on unscripted speech
Specifically, HiPPO strategically employs a speech       (or free-speaking scenarios) and then sheds light on
foundation model along with a grapheme-to-               the proposed methods, encompassing the
phoneme (G2P) converter to derive the most likely        assessment model, training objectives, and learning
phone sequence produced by an L2 learner, thereby        strategy. Due to the space limit, the overview of
bringing the assessment task closer to its scripted-     related work will be given in Appendix A.
speech counterpart, as illustrated in Figure 1(c). To
overcome sequence length constraints and preserve
                                                                                                     Word                     Word                    Word
                                                                                                                                                                                                           Utterance
                                                                                                     [The]                   [City]                   [Of]
                                                                                                                         Acc/Stress/Total        Acc/Stress/Total                                   Acc/Flu/Comp/Pros/Total
                                                                                                Acc/Stress/Total
      Phone Phone          Phone Phone Phone Phone                  Phone Phone
      [DH] [AH]             [S]   [IH]  [T] [IY]                    [AH]   [V]
       Acc   Acc            Acc   Acc Acc Acc                        Acc   Acc
                                                                                                                      Word-level Regressors                                                         Utterance-level Regressors

                         Phone-level Accuracy Regressor
                                                                                                            CNN                       CNN                   CNN                                            Atten Pooling
                                         Phone Encoder                                                                                                                                                                      ……
                                                                                                                         Word Encoder
                                                                                                                                                                                                        Utterance Encoder



                                                                                                                                                                       Weighted Merge
                                                                                                                                                                                                                            ……
                                                                                                   Cat & Proj              Cat & Proj              Cat & Proj                                                Cat & Proj
                                     Feature Extraction                                                                                                                                                                      ……

                                                                                                 Atten Pooling           Atten Pooling            Atten Pooling                                                                        CNN
                                                                                                                                                                                                                            ……

        DH       AH              S        IH       T      IY           AH        V                                                                                                                                          ……
       Word 1 (The)                       Word 2 (City)               Word 3 (Of)                        Phone-level Feature Representations (X ! , H ! )                                                                                % ")
                                                                                                                                                                                        Phone-level and Word-level Features (X ! , H ! , H
                         Phone-level Modeling                                                                        Word-level Modeling                                                            Utterance-level Modeling
            Phone-level Textual             CTC-GOP Feature             Word-level Textual       Projected SSL-                       Input                         Converted Phone                                       Transcribed Word
             Embeddings (E ! )               Sequence (X ! )            Embeddings (E ")         Features (# ##$ )
                                                                                                                                                   DH … V                                             The …      Of
                                                                                                                                    Speech (X)                       Sequence (&)                                           Sequence (!)
                                                                                                              Input Descriptions


         Figure 3: The overall architecture of the proposed hierarchical pronunciation assessment model (HiPPO).

                                 Speech
                                 Signal        Speech Foundation
                                                                      Transcribed
                                                                     Word Sequence      G2P Model
                                                                                                                                2.2          Hierarchical Pronunciation Assessment
       (An L2 Learner)
                                     X          Model (Whisper)            )             (G2PE)
                                                                                                                                             Model for Spoken Languages (HiPPO)
                                                                                               Perceived Phone
                                                        CTC-based GOP Feature

    A Set of Aspect Scores
                                                        Sequence and SSL-based
                                                                                                 Sequence *
                                                                                                                                Figure 3 depicts the model architecture of HiPPO,
                                                            Feature Vectors            Pronunciation
                                                                                                                                which encompasses three major modeling stages:
             ! !          !
    A ! = {$" , $# , … , $$! }                                                        Feat. Extraction
                                                          E %&' , ,(#) , ,*+ , ,(,-
                                            (HiPPO)
                                                                                                                                phone-, word-, and utterance-level modeling. In
Figure 2: Processing flow of HiPPO for qualifying                                                                               each of these modeling stages, the corresponding
the oral skills in unscripted speech.                                                                                           encoder is constructed with a newly proposed
                                                                                                                                Conv-LLaMA block. After obtaining the
2.1              Problem Definition                                                                                             representations of all pronunciation aspects, a
                                                                                                                                distinct regressor is used to generate the
To assess speaking skills across different linguistic                                                                           pronunciation score of each aspect.
granularities for unscripted speech, as illustrated in
Figure 2, we first employ a speech foundation                                                                                   Pronunciation Feature Extraction. To portray
model1 to transcribe a speech signal X produced by                                                                              the pronunciation quality of X , we extract
an L2 learner into a sequence of 𝑀 words 𝐰 =                                                                                    connectionist temporal classification (CTC)-based
(𝑤! , 𝑤" , … , 𝑤# ) and subsequently a G2P                                                                                      goodness pronunciation (GOP) features for each
converter2 to generate the corresponding phonetic                                                                               phone in 𝐩 , where the pronunciation quality is
transcription of 𝑁 phones 𝐩 = (𝑝! , 𝑝" , … , 𝑝$ ) ,                                                                             measured as the likelihood ratio of all valid CTC
where 𝐰 and 𝐩 collectively serve as a proxy for the                                                                             alignments of 𝐩 to that of the deviated phonetic
textual and phonetic realizations perceived by                                                                                  transcripts (Cao et al., 2024). Compared to
                                                                                                                                previous studies on the GOP feature extraction
human raters. Let G = {𝑔%&' , 𝑔()*+ , 𝑔,-- }
                                                                                                                                (Witt and Young, 2000; Hu et al., 2015; Shen et al.,
denotes the set of linguistic granularities, where
                                                                                                                                2021), the CTC-based method computes GOP
𝑔%&' , 𝑔()*+ and 𝑔,-- mark the phone-, word-,
                                                                                                                                scores without explicit timestamps of phone
and utterance-level linguistic granularities,
                                                                                                                                segments and inherently tackles alignment errors
respectively. HiPPO is trained under a multi-task
                                                                                                                                by accounting for insertions and/or deletions in the
learning paradigm to estimate a set of aspect score
                       . .        .                                                                                             deviated phonetic transcriptions. Additionally, to
sequences A. = {𝐚! , 𝐚" , … , 𝐚$! } for each
                                                                                                                                capture supra-segmental articulation cues and
granularity 𝑔 ∈ G , where 𝑁. is the number of                                                                                   mitigate the data-sparsity issue frequently
pronunciation aspects.                                                                                                          occurring in L2 speech corpora (Lo et al., 2024;
                                                                                                                                Bannò and Matassoni, 2022), we leverage self-
                                                                                                                                supervised learning (SSL)-based features for
                                                                                                                                utterance-level pronunciation modeling. The SSL-

1                                                                                                                               2
    https://huggingface.co/openai/whisper-large-v3                                                                                  https://github.com/Kyubyong/g2p
based features are extracted at the frame-level and
then aggregated to the utterance-level via simple                              MHSA Module                                                      SwiGLU Operation


mean pooling over time (Chao et al., 2022; Kim et                                                                                                             Linear
                                                                                                  Self-Attention
                                                                    RMSNorm    Feed-forward                        Feed-forward              RMSNorm
al., 2022). A bit of terminology: the pronunciation                                                                                                                                       Dropout
                                                                                                    w/ RoPE
                                                                                                                                                         Linear        SiLU



                                                                                                                                                                                                    Weighted Combine
feature extraction of HiPPO produces a phone-
level pronunciation feature sequence X / ∈ ℝ+" ×$

                                                                                                                          Depthwise Conv                                Pointwise Conv
and a projected SSL-based feature vector 𝐱 112 ∈
                                                                                 Pointwise Conv
                                                                    RMSNorm                                                                RMSNorm                                       Dropout
                                                                                                        SiLU                                           SiLU
ℝ+# ×! , where 𝑁 is the length of the phone
sequence, and 𝑑/ and 𝑑3 represent the hidden                                                                       CNN Module

dimension of phone- and utterance-level modeling.
The processing flow summarized as follows:                    Figure 4: A schematic illustration of the proposed
                     !
                   X = Lin!    (E"#! ),             (1)       Conv-Llama block.

          𝐱 $$% = Lin$$% +,𝐞&'( ; 𝐞)* ; 𝐞&%+ /0),   (2)   multiplication on the key and query vectors in the
where Lin/ (∙) and Lin112 (∙) are linear projections,     multi-head self-attention layer (Su et al., 2024).
and [; ] is a concatenation operation. E45/ ∈             Hierarchical APA Modeling. For the phone-level
ℝ6!×$ refers to the CTC-based GOP features                assessment, we first combine the pronunciation
extracted from a well-trained CTC-based ASR               features X / with the textual embeddings E< ∈
model 3 , while 𝐞7"8 , 𝐞93 , and 𝐞72: ∈ ℝ!;"6×!           ℝ+" ×$ in a point-wise manner, followed by a
are utterance-level SSL-feature vectors derived           phone encoder to obtain aspect representations H/ :
from pre-trained acoustic models, viz. wav2vec-                                                                !
                                                                                                       H, = X ! + E! ,                                                                                                 (3)
2.0, Hubert, and WavLM, respectively.
                                                                                                                                                              !
                                                                                              H! = PhnEnc+H, 0,                                                                                                        (4)
Convolution-augmented LLaMA Block (Conv-                              /
LLaMA). To model a pronunciation feature                  where E is generated by passing phonetic
sequence of arbitrary length and capture nuanced          transcription 𝐩 into a phone embedding layer, and
articulation traits across linguistic units, we           PhnEnc(∙) is a stack of 3 Conv-LLaMA blocks.
introduce a Conv-LLaMA block to stack a                   Subsequently, a regressor is built on top of H/ to
hierarchical assessment model, which enhances the         produce phone-level accuracy scores.
model component of LLaMA (Touvron et al., 2023)               For word-level assessments, we begin by
with a convolutional branch and rotary position           deriving a word representation vector from its
encoding. As depicted in Figure 4, the proposed           constituent phones with a dedicated attention
block comprises two branches: one branch captures         pooling, implemented with a 1-D depth-wise
supra-segmental articulation cues via a multi-head        convolution layer, an MHA layer, and an average
self-attention (MHSA) module followed by a                operation. The word-level input features X 7 ∈
swish-gated linear unit (SwiGLU) operation                ℝ+$ ×# are obtained by feeding X / and H/
(Touvron et al., 2023), while the other focuses on        through word-level attention pooling, and then
capturing local pronunciation traits via a                packing their pooled counterparts together via a
convolutional neural network (CNN) module.                linear projection:
Subsequently, these two branches are combined via                             6 & = AttPool& (X ! ),
                                                                              X                                                                                                                                        (5)
                                                                                            !
a weighted average operation (Peng et al., 2022).
                                                                              6
                                                                              H& = AttPool&" (H! ),                                                                                                                    (6)
The proposed CNN module is equipped with two
key components, i.e., a point-wise convolution for                                         6&; 6
                                                                              X & = Lin& +,X   H& /0,                                                                                                                  (7)
capturing information across feature dimensions
                                                          where 𝑀 denotes the length of transcribed word
and a depth-wise convolution layer for extracting
                                                          sequence, and 𝑑7 symbolizes the hidden
local spatial patterns. On the other hand, the MHSA
module incorporates rotary position encoding              dimension of word-level modeling4. Following the
(RoPE), a relative position encoding method               integration of word-level textual embeddings E7
developed for extrapolating feature sequence              with X 7 , a word encoder is employed to generate a
lengths, which operates through channel-wise
3
    https://github.com/frank613/CTC-based-GOP.git         4
                                                            For efficient parallel computation, a word-level
                                                          representation is duplicated to length of constituent phones.
sequence of contextualized representations H7 ∈
                                                                                      Score:1
ℝ+$ ×# :                                                                         !%             !%
                                                                  Attract with
               H,& = X & + E& ,                   (8)         Tightness Term (ℒ( )
                                                                                                          Score:2
                                                                                           !!""'                     !%
            H& = WordEnc(H,& ),                   (9)                                                    !!""$

where E7 are obtained by mapping the transcribed                                                                      !%
                                                                       Score:0
word sequence 𝐰 through modernBERT (Warner
                                                                                               Repel with
et al., 2024), and WordEnc(∙) consists of 2 Conv-                      !!!"#               Diversity Term (ℒ&)
                                                                !%
LLaMA blocks. Consequently, three distinct 1-D
depth-wise convolution layers are performed on                                                       Tightness Term (ℒ( )
top of H7 to generate aspect representations (viz.                             !%                    Diversity Term (ℒ&)
H7% , H7& , and H7' ). The word-level
pronunciation scores (accuracy, stress, and total)          Figure 5: Illustration of contrastive ordinal
are generated by passing the aspect representations         (CONO) regularizer, which preserves inter-score
into the corresponding regressors.                          discrepancies with diversity term ℒ; and maintains
    For the utterance-level assessments, we first           intra-score compactness via the tightness term ℒ< .
fuse H7% , H7& , and H7' with a weighted average
operation to produce J  H7 ∈ ℝ+$ ×# . After the
distinct forward propagation through 1-D depth-           Constative Ordinal Regularizer. To mitigate the
wise convolution layers on X / , H/ , and JH7 , the       detrimental effects of ASR errors on assessment
corresponding outputs are combined via a linear           performance, we devise a contrastive ordinal
projection, and then fed into an utterance encoder        (CONO) regularizer to extract score-discriminative
to generate contextualized representations H3 :           features. As phone-level representations are
       >                                                  essential for constructing a hierarchical assessment
       H& = Merge(H&! , H&" , H&# ),              (10)
                                                          model, we first extract an utterance-level feature 𝐳
                                          > & )]), (11)
 H,* = Lin* ([DC. (X ! ); DC' (H! ); DC/ (H               by averaging the outputs of the phone-level
                                                          encoder H% over time. For a training batch of 𝐿
              H* = UttEnc(H,* ),                  (12)
                                                          utterances, the corresponding feature vectors are
where UttEnc(∙) is a single Conv-LLaMA block,             aggregated to form a sequence Z = (𝐳! , 𝐳" , . . . , 𝐳> ).
and DC! (∙), DC" (∙), and DC= (∙) are distinct 1-D            As depicted in Figure 5, the CONO regularizer
depth-wise convolution layers, each of which has a        encourages the feature vectors Z to render the
kernel size of 3. Afterward, five separate attention      ordinal relationship of the utterance-level accuracy
pooling layers are stacked on top of H3 and then          scores 𝐲 = (𝑦! , 𝑦" , … , 𝑦> ) via the synergy of a
integrated with the projected SSL-based feature           diversity term ℒ+ and a tightness term ℒ- :
vector 𝐱 112 via separate residual connections.
                                                                        ℒ9:5: = 𝜆; ℒ; + 𝜆< ℒ< ,                           (14)
These aspect representation vectors are processed
by the corresponding regressors to derive the             where 𝜆+ and 𝜆- are trade-off parameters. The
utterance-level aspect scores (viz. accuracy,             diversity term ℒ+ preserves inter-score
fluency, completeness, prosody, and total).               discrepancies by minimizing the negative distances
                                                          between score centroid vectors 𝐳?) with a penalty:
2.3   Training Objectives                                                              A
                                                                     1
For the proposed model, we first consider a                ℒ; = −          H H 𝑤=> P𝐳?& − 𝐳?' P ,                         (15)
                                                                  𝑀(𝑀 − 1)                     '
                                                                                      =8. =@>
weighted sum of mean squared error (MSE) losses
                                                          where 𝐾 is the number of score centers, and
as the training objective, collected from multiple
                                                          penalty 𝑤@A = |𝑦@ − 𝑦A | signifies the absolute
aspects across granularities:
                              5% 6.                       differences between the regression targets. The
                      1                                   score centroid vectors 𝐳?) and 𝐳?* are computed
        ℒ010 = H 𝜆2 ×    H ℒ2$ ,                  (13)
                      𝑁2
                2∈4           78,                         from Z by averaging all feature vectors whose
where 𝜆. denotes adjustable parameter, 𝑁. is              utterance-level accuracy scores are 𝑦@ and 𝑦A ,
number of aspects at granularity 𝑔 , and ℒ.(              respectively. The tightness term ℒ- regulates intra-
represents the MSE loss computed for the 𝑘 -th            score    compactness       by    pulling    feature
aspect score sequence.
                             Phone Scores        Word Score (PCC)         Utterance Score (PCC)
         Models
                            MSE↓     PCC↑       Accuracy↑ Total↑ Accuracy↑ Fluency↑ Prosody↑            Total↑
         Liu2023              -         -           -          -          -        0.795        -           -
                                                                        0.692      0.757      0.757       0.714
       VanillaSSL             -         -           -          -
                                                                       (±0.006)   (±0.010)   (±0.009)   (±0.006)
                                                  0.427      0.436      0.705      0.772      0.763       0.730
         MultiPA              -         -
                                                 (±0.008)   (±0.010)   (±0.009)   (±0.010)   (±0.016)   (±0.006)
                            0.240     0.330       0.416      0.417      0.717      0.797      0.791       0.741
       Parallel-TFR
                          (±0.003)   (±0.009)    (±0.016)   (±0.019)   (±0.014)   (±0.003)   (±0.003)   (±0.010)
                            0.237     0.345       0.426      0.428      0.726      0.799      0.791       0.748
     Parallel-LLaMA
                          (±0.001)   (±0.004)    (±0.012)   (±0.011)   (±0.006)   (±0.006)   (±0.005)   (±0.004)
                            0.238     0.328       0.412      0.418      0.692      0.786      0.780       0.724
       Hier-LLaMA
                          (±0.001)   (±0.008)    (±0.011)   (±0.012)   (±0.012)   (±0.008)   (±0.006)   (±0.008)
                            0.202     0.480       0.520      0.521      0.733      0.806      0.797       0.754
          HiPPO
                          (±0.003)   (±0.013)    (±0.016)   (±0.016)   (±0.006)   (±0.003)   (±0.002)   (±0.006)
                            0.213     0.448       0.513      0.516      0.720      0.797      0.791       0.743
        w/o CONO
                          (±0.004)   (±0.012)    (±0.007)   (±0.007)   (±0.005)   (±0.003)   (±0.002)   (±0.005)
                            0.241     0.331       0.404      0.404      0.698      0.790      0.785       0.728
         w/o CL
                          (±0.002)   (±0.011)    (±0.012)   (±0.014)   (±0.010)   (±0.011)   (±0.011)   (±0.007)

    Table 1: The performance evaluations of our model and all compared methods on Speechocean762 test set in
    simulated free-speaking scenarios.

representations 𝐳@ towards their score centroid               ℒJ*HH with a probability of 𝒫(𝜏), where 𝒫(𝜏) =
vectors 𝐳?) :                                                 𝜏⁄𝑇 is a scheduling function, with 𝑇 being the total
                      B
                 1                                            number of training iterations and 𝜏 ∈ [0, 𝑇]. The
             ℒ< = HT𝐳𝑖 − 𝐳𝑐𝑖 T .                     (16)     training strategy at iteration 𝜏 is defined by
                 𝐿            '
                      =8.
The training objective of HiPPO is designed as a                     `1 − 𝕀(𝜏)aℒ*HI+ + 𝕀(𝜏)ℒJ*HH ,            (18)
linear combination of the pronunciation                       with the indicator function 𝕀(𝜏) given by
assessment task ℒDED and the CONO
                                                                        1, learning hard task (𝑤. 𝑝. 𝒫(𝜏))
regularization ℒFG$G :                                        𝕀(𝜏) = X                                      . (19)
                                                                      0, learning easy task (𝑤. 𝑝. 1 − 𝒫(𝜏))
            ℒ = ℒ010 + 𝜆9:5: ℒ9:5: ,                 (17)
where 𝜆FG$G is a tunable hyperparameter.                      3      Experimental Settings
2.4      Curriculum Learning                                  This section describes the benchmark dataset and
Drawing inspiration from education systems,                   metrics used in this paper. Implementation details
curriculum learning techniques improve model                  and descriptions of comparative methods are
performance by progressively escalating training              elaborated in Appendices B and C. Furthermore,
complexity from simple to hard (Bengio et al.,                HiPPO and the experimental dataset are publicly
2009; Castells et al., 2020; Vakil and Amiri, 2023).          available to ensure the reproducibility of our work,
The proposed curriculum training strategy starts              accelerate CAPT research, and facilitate
from assessing pronunciation in a reading-aloud               standardized evaluation5.
scenario ℒ*HI+ , and gradually shifts towards                 Benchmark Dataset. A series of experiments were
assessing pronunciation in the free-speaking                  carried out on the Speechocean762 dataset, a
counterpart ℒJ*HH . In ℒ*HI+ , the pronunciation              publicly available corpus specifically designed for
features are extracted from the learner’s speech              CAPT research (Zhang et al., 2021). This dataset
alongside the corresponding reference text, while             comprises 5,000 English-speaking recordings
in ℒJ*HH the transcribed word sequence serve as an            collected from 250 Mandarin L2 learners, with
alternative for pronunciation feature extraction. At          training and test sets of equal size, each containing
each training iteration 𝜏, HiPPO selects a task from          2,500 utterances. Speechocean762 was collected in
ℒ*HI+ with a probability of 1 − 𝒫(𝜏), or from the             a reading-aloud scenario (reading reference texts

5
    https://github.com/bicheng1225/HIPPO/tree/main
                                                                                   HiPPO
                                                                                   HiPPO        HiPPO
                                                                                                w/o    w/o ℒ!"#"
                                                                                                    Cono
                                                                    0.8              HiPPO      w/o Cono
                                                                             (a) Utterance-level Accuracy Scores
                                                                     0.8
                                                                    0.6
                                                                     0.6
                                                              PCC
                                                                    0.4
                                                                     0.4
                                                                    0.2
                                                                     0.2   0.00%     19.61%     22.75%    26.12%
                                                                                (b) Word-level
                                                                            0.00%      19.61% Accuracy
                                                                                                 22.75%Scores
                                                                                                           26.12%
           (a) Vanilla Model            (b) ℒ!!
                                                                     0.6

                                                               PCC
                                                                     0.5
                                                                     0.4
                                                                     0.3
                                                                           0.00%      19.61% Accuracy
                                                                               (c) Phone-level 22.75% Scores
                                                                                                         26.12%

                                                                     0.6
                                                                     0.6
                                                               PCC   0.5
                                                                     0.5
                (c) ℒ!                 (d) ℒ! + ℒ"                   0.4
                                                                     0.4
                                                                     0.3
                                                                     0.3
    Figure 6: Visualization of utterance-level                             0.00%
                                                                           0.00%      19.61%
                                                                                      19.61%   22.75%        26.12%
                                                                                                             26.12%
                                                                           0.00%       19.61%   22.75%        26.12%
    representations Z , where the orange, blue, and                                        WER%
    green points indicate accuracy scores of 4.0, 6.0,        Figure 7: A comparison of PCC scores for
    and 8.0, respectively. The plots display feature          pronunciation accuracy at the phone, word, and
    points for: (a) vanilla model, (b) vanilla model with     utterance levels between HiPPO and HiPPO w/o
    a modified diversity term ℒ;( where the penalty is        ℒ9CDC under varying word error rate (WER)
    removed, (c) vanilla model with diversity term ℒ; ,       conditions. These WERs are calculated based on
    and (d) vanilla model with CONO regularizer               the reference text and different input transcriptions
    ℒ9:5: .                                                   which are reference text and outputs of Whisper
                                                              models (viz., large-v3, medium-en, small-en).

aloud) with accessible reference texts and                  different pronunciation aspects and linguistic
corresponding canonical phones (phone-level                 granularities. 2) As to the ASR-free models, both
reference text). To simulate a free-speaking                VanillaSSL and Liu2023 are limited to utterance-
scenario for possible use cases of spoken language          level assessment, lacking finer-grained aspect
assessment, we exclude these reference texts from           scores at the phone or word level. Moreover,
the model input and rely instead on the ASR                 Liu2023 outperforms VanillaSSL in assessing the
transcribed words and their associated phones. The          utterance-level fluency, where the gains stem from
detailed pronunciation score assignments for the            the integration of frame-level phonetic information
free-speaking scenario are provided in Appendix D.          via k-means clustering. Note also that effectively
                                                            using phonetic information to boost assessment
Evaluation Metrics. 1) Pearson correlation
                                                            performance has been verified in prior work (Gong
coefficient (PCC, ↑ ) measures the linear
                                                            et al., 2022). Subsequently, compared to MultiPA,
correlation between predicted and ground-truth
                                                            our method extracts pronunciation feature at the
scores for disparate pronunciation aspects. 2) Mean
                                                            phone-level and then qualifies pronunciation
squared error (MSE, ↓ ) evaluates score
                                                            aspects      hierarchically     across     linguistic
discrepancy of the phone-level accuracy. The mean
                                                            granularities, resulting in superior assessment
and standard deviation are reported for both
                                                            performance. 3) In comparison among the variants
metrics.
                                                            of HiPPO, Parallel-CTC and Parallel-LLaMA
                                                            outperform Hier-LLaMA in most assessment tasks.
4      Experimental Results
                                                            This observation suggests that, when pronunciation
Assessments in the Free-speaking Scenarios. At              features are extracted from the transcripts
the outset, we compare our HiPPO with several               containing ASR errors, the parallel design offers a
current top-of-the-line APA models in the                   more flexible and robust neural architecture for
simulated free-speaking scenarios. From the results         assessments in free-speaking scenarios compared
shown in Table 1, we make the following                     to the hierarchical one. Notably, HiPPO stands out
observations. 1) Our HiPPO achieves better PCC              in assessment performance via the synergy of
scores than all other competitive methods across
                      Phone Scores        Word Score (PCC)                Utterance Score (PCC)
      Models
                    MSE↓      PCC↑       Accuracy↑ Total↑        Accuracy↑ Fluency↑ Prosody↑       Total↑
     AzurePA          -         -          0.623       -           0.700     0.715      0.842      0.782
                    0.085     0.612        0.533     0.549         0.714     0.753      0.760      0.742
      GOPT
                   (±0.001)   (±0.003)    (±0.004)    (±0.002)    (±0.004)   (±0.008)   (±0.006)   (±0.005)
                    0.078      0.656       0.598       0.617       0.760      0.828      0.827      0.796
        3M
                   (±0.001)   (±0.005)    (±0.005)    (±0.005)    (±0.004)   (±0.006)   (±0.008)   (±0.005)
                    0.084      0.616       0.575       0.591       0.730      0.749      0.751      0.754
     HiPAMA
                   (±0.001)   (±0.004)    (±0.004)    (±0.004)    (±0.002)   (±0.001)   (±0.002)   (±0.002)
                    0.081      0.644       0.622       0.634       0.735      0.801      0.795      0.764
     HierTFR
                   (±0.000)   (±0.000)    (±0.002)    (±0.002)    (±0.008)   (±0.004)   (±0.002)   (±0.002)
                    0.078      0.650       0.575       0.589       0.754      0.816      0.806      0.772
   Parallel-TFR
                   (±0.001)   (±0.009)    (±0.018)    (±0.013)    (±0.011)   (±0.006)   (±0.007)   (±0.010)
                    0.074      0.658       0.598       0.610       0.774      0.837      0.829      0.796
  Parallel-LLaMA
                   (±0.002)   (±0.007)    (±0.012)    (±0.009)    (±0.009)   (±0.006)   (±0.004)   (±0.009)
                    0.082      0.656       0.622       0.634       0.789      0.844      0.832      0.811
   Hier-LLaMA
                   (±0.002)   (±0.006)    (±0.006)    (±0.008)    (±0.006)   (±0.003)   (±0.003)   (±0.005)
                    0.080      0.657       0.630       0.643       0.791      0.845      0.837      0.816
     HiPPO*
                   (±0.001)   (±0.001)    (±0.009)    (±0.009)    (±0.002)   (±0.001)   (±0.001)   (±0.001)

 Table 2: The performance evaluations of our model and all compared methods on Speechocean762 test set in
 the read-aloud scenarios. HiPPO* refers to the model trained without curricular strategy and CONO
 regularizer.

CONO regularizer and curriculum learning                 the utterance-level accuracy score (Yan et al.,
strategy.                                                2024). By comparing Figures 6(b) with 6(c), it is
                                                         evident that both diversity terms (ℒ;( and ℒ; ) can
Ablation Studies in Free-speaking Scenarios. As
                                                         capture subtle differences between utterance-level
shown in the last two columns of Table 1, we ablate
                                                         scores, where feature points are clustered by their
HiPPO with following settings: removing the
                                                         respective accuracy scores. The integration of
CONO regularizer (w/o CONO) and substituting
                                                         ordinal penalty, as shown in Figure 6(c), further
the curriculum learning strategy with training on a      facilitates a clearer scattering of feature
combined dataset of reading-aloud and free-
                                                         representations, with blue and green points more
speaking scenarios (w/o CL). From these ablation
                                                         distinctly spread out. Finally, the impact of the
studies we can observe that both the CONO
                                                         tightness term ℒ- is verified in Figure 6(d), where
regularizer and the curriculum strategy are crucial
                                                         the feature points exhibit tighter clustering in
to HiPPO. Removing either one of them leads to a
                                                         comparison with other subfigures.
decline in performance across several aspects and
granularities. Second, the curriculum learning           Effectiveness of CONO Regularizer across
strategy makes a substantial contribution to the         Different ASR Word Error Rate Settings. Figure
performance. Training HiPPO with the combined            7 examines the effectiveness of CONO regularizer
dataset, in contrast, results in lower performance       ℒF)') for the assessment accuracy at different
across all assessment tasks.                             granularities across various ASR word error rates
                                                         (WERs), by comparing the HiPPO and its ablated
Qualitive Analysis on the CONO Regularizer in
                                                         version (HiPPO w/o CONO). Notably, in this set of
the Free-speaking Scenarios. In Figure 6, we
                                                         experiments, our models were trained on the
qualitatively examine the effectiveness of the
                                                         reference text and transcripts generated by
additional training regularizer on the proposed
                                                         whisper-large-v3 (achieving a WER of 19.6%) via
hierarchical model. As depicted in Figure 6, the
                                                         proposed curricular learning strategy. First, with
feature points in these subfigures display ordinal
                                                         reference text as the input transcript, the
relationships, which are sorted by their utterance-
                                                         assessment performance of both models seems
level scores, with blue points being located
                                                         comparable across granularities (phone, word, and
between red points and green points. This result
                                                         utterance levels). Second, at the utterance-level
can be attributed to the aggregation of
                                                         assessment, the PCC scores of these two models
representations Z from the phone-level
                                                         appear relatively immune to WER degradation. A
representations, which are highly correlated with
                                                         possible reason is that utilization of SSL-based
features in utterance-level modeling, as the SSL         we explored a simple yet effective curriculum
models are often pre-trained on complex acoustic         learning strategy for the spoken language
environments. Finally, the benefits of the CONO          assessment. Extensive experimental results
regularizer are more prominent at finer-grained          validate the feasibility and effectiveness of the
linguistic levels. Specifically, the performance         proposed methods, obtaining superior assessment
degrades substantially at the phone and word levels;     performance compared to several state-of-the-art
however, the performance of HiPPO exhibits a             methods in both reading-aloud and stimulated free-
more attenuated decline in comparison to other           speaking scenarios. In future work, we plan to
variants, which highlights the robustness of the         explore more robust assessment models under
proposed regularizer to ASR errors.                      various word error rate conditions for unscripted
                                                         pronunciation assessments.
Assessments in the Read-aloud Scenario. In
Table 2, the proposed HiPPO is evaluated in a read-
                                                         6   Limitations
aloud setting, where reference texts are employed
in training and test. The main findings are              Spoken language assessment gauges language
presented as follows. 1) HiPPO markedly                  competence across three sub-dimensions:
outperforms other methods in most pronunciation          pronunciation (fluency and delivery), language use
aspects. Notably, in contrast to prior studies, i.e.,    (vocabulary and grammar), and topic development
parallel models (GOPT and 3M) and hierarchical           (content and discourse). In this paper, however,
ones (HiPAMA and HierTFR), our model assesses            HiPPO focuses exclusively on pronunciation
pronunciation quality without explicit phone-level       assessment within the broader context of spoken
timestamps and achieves superior performance             language evaluation. The following are several
across various pronunciation aspects. 2) AzurePA         limitations of HiPPO in real-world applications:
stands out at the assessment of utterance-level
prosody, whereas its performance on the other            Transcriptions Containing ASR Errors.
pronunciation aspects trails behind that of the other    Although speech foundation models have achieved
methods. These inferior results probably stem from       near-human accuracy on public benchmark
that AzurePA is a commercial system that might           datasets, transcribing non-native English speech
has not been finetuned on Speechocean762. 3) As          remains challenging. In our experiments, the word
to the comparison between the variants of HiPPO          error rate (WER) for Speechocean762, transcribed
(Parallel-LLaMA, Parallel-TFR, and Hier-                 using Whisper-large-v3, is 19.22% for the training
LLaMA),        Hier-LLaMA        attains     superior    set and 17.49% for the test set. Examining the
performance in most pronunciation aspects,               performance of HiPPO through the lens of different
particularly at the word and utterance levels, with a    WER conditions, we observed a significant
slight sacrifice in performance at the phone-level.      degradation when ASR errors were severe, even
These results are in line with the findings from         with the proposed CONO regularizer.
previous studies (Do et al., 2023; Chao et al., 2023).   Lack of Accent Diversity. The used dataset merely
By comparing HiPPO with Hier-LLaMA, we can               contains Mandarin L2 learners, hindering the
verify that the proposed Conv-LLaMA block                generalizability of the proposed model and could
brings consistent improvements to pronunciation          be untenable when assessing the L2 learners with
assessments.                                             diverse accents.
                                                         The Lack of Interpretability. The model of the
5   Conclusion                                           proposed method simply trains to mimic expert’s
In this paper, we have proposed a novel                  annotations without resorting to manual
hierarchical pronunciation assessment model              assessment rubrics or other external knowledge,
(dubbed HiPPO) for the spoken languages. To              making it not straightforward to provide reasonable
address arbitrarily long pronunciation feature           explanations for the assessment performance.
sequences and capturing articulation traits across       Ethics Statement
various linguistic granularities, we designed a
Conv-LLaMA block for the proposed model. A               We hereby acknowledge that all of the co-authors
contrastive ordinal regularizer is put forward to        of this work compile with the provided ACL Code
enhance robustness against ASR errors. Moreover,         of Ethics and honor the code of conduct. Our
                                                         experimental corpus, Speechocean762, is widely
used and publicly available. We think there are no      Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023.
potential risks for this work.                           Hierarchical pronunciation assessment with multi-
                                                         aspect attention. In Proceedings of the IEEE
References                                               International Conference on Acoustics, Speech and
                                                         Signal Processing (ICASSP), pages 1–5.
Stefano Bannò and Marco Matassoni. 2022.
 Proficiency assessment of L2 spoken English using      Keelan Evanini, Maurice Cogan Hauck, and Kenji
 wav2vec 2.0. In Proceedings of the IEEE Spoke            Hakuta. 2017. Approaches to automated scoring of
 Language Technology Workshop (SLT), pages 1088–          speaking for K–12 English language proficiency
 1095.                                                    assessments. ETS Research Report Series, pages 1–
                                                          11.
Yoshua Bengio, Jérôme Louradour, Ronan Collobert,
 and Jason Weston. 2009. Curriculum learning. In        Nancy F. Chen and Haizhou Li. 2016. Computer-
 Proceedings of the International Conference on          assisted pronunciation training: From pronunciation
 Machine Learning (ICML), pages 41–48.                   scoring towards spoken language learning. In
                                                         Proceedings of the Asia-Pacific Signal and
Xinwei Cao, Zijian Fan, Torbjørn Svendsen,               Information Processing Association Annual Summit
 Giampiero Salvi. 2024. A framework for phoneme-         and Conference (APSIPA ASC), pages 1–7.
 level pronunciation assessment using CTC. In
 Proceedings of Interspeech (INTERSPEECH), pages        Luciana Ferrer, Harry Bratt, Colleen Richey, Horacio
 302–305.                                                Franco, Victor Abrash, Kristin Precoda. 2015.
                                                         Classification of lexical stress using spectral and
Thibault Castells, Philippe Weinzaepfel, and Jerome      prosodic features for computer-assisted language
 Revaud. 2020. Superloss: A generic loss for robust      learning systems. Speech Communication, volume 69,
 curriculum learning. In Proceedings of Advances in      pages 31–45.
 Neural Information Processing Systems (NeurIPS).
                                                        Horacio Franco, Harry Bratt, Romain Rossier, Venkata
Fu-An Chao, Tien-Hong Lo, Tzu-I. Wu, Yao-Ting            Rao Gadde, Elizabeth Shriberg, Victor Abrash, and
 Sung, Berlin Chen. 2022. 3M: An effective multi-        Kristin Precoda. 2010. EduSpeak: A speech
 view, multigranularity, and multi-aspect modeling       recognition and pronunciation scoring toolkit for
 approach to English pronunciation assessment. In        computer-aided language learning applications.
 Proceedings of the Asia-Pacific Signal and              Language Testing, volume 27, pages 401–418.
 Information Processing Association Annual Summit
 and Conference (APSIPA ASC), pages 575–582.            Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang,
                                                         and James Glass. 2022. Transformer-based multi-
Fu-An Chao, Tien-Hong Lo, Tzu-I. Wu, Yao-Ting            aspect multigranularity non-native English speaker
 Sung, Berlin Chen. 2023. A Hierarchical Context-        pronunciation assessment. In Proceedings of the
 aware Modeling Approach for Multi-aspect and            IEEE International Conference on Acoustics, Speech
 Multigranular Pronunciation Assessment. In              and Signal Processing (ICASSP), pages 7262–7266.
 Proceedings of Interspeech (INTERSPEECH), pages
 974–978.                                               Wenping Hu, Yao Qian, Frank K. Soong, and Yong
                                                         Wang. 2015. Improved mispronunciation detection
Yu-Wen Chen, Zhou Yu, and Julia Hirschberg. 2024.        with deep neural network trained acoustic models and
 MultiPA: A multi-task speech pronunciation              transfer learning based logistic regression classifiers.
 assessment model for open response scenarios. In        Speech Communication, volume 67, pages 154–166.
 Proceedings of Interspeech (INTERSPEECH), pages
 297–301.                                               Yassine Kheir, Ahmed Ali, and Shammur Chowdhury.
                                                         2023. Automatic Pronunciation Assessment–A
Eduardo Coutinho, Florian Hönig, Yue Zhang, Simone       Review. In Findings of the Association for
 Hantke, Anton Batliner, Elmar Nöth, and Björn           Computational Linguistics: EMNLP, pages 8304–
 Schuller. 2016. Assessing the Prosody of Non-Native     8324.
 Speakers of English: Measures and Feature Sets. In
 Proceedings of the Tenth International Conference on   Eesung Kim, Jae-Jin Jeon, Hyeji Seo, Hoon Kim. 2022.
 Language Resources and Evaluation (LREC), pages         Automatic pronunciation assessment using self-
 1328–1332.                                              supervised speech representation learning. In
                                                         Proceedings of Interspeech (INTERSPEECH), pages
Huaijin Deng, Youchao Lin, Takehito Utsuro, Akio         1411–1415.
 Kobayashi, Hiromitsu Nishizaki, and Junichi
 Hoshino. 2020. Automatic fluency evaluation of         Yaman Kumar Singla, Avyakt Gupta, Shaurya Bagga,
 spontaneous speech using disfluency based features.     Changyou Chen, Balaji Krishnamurthy, Rajiv Ratn
 In Proceedings of the IEEE International Conference     Shah. 2021. Speaker-conditioned hierarchical
 on Acoustics, Speech and Signal Processing              modelling for automated speech scoring. In
 (ICASSP), pages 9239-9243,                              Proceedings of the ACM International Conference on
 Information & Knowledge Management (CIKM),              Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng
 pages 1681–1691.                                         Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer:
                                                          Enhanced Transformer with rotary position
Binghuai Lin and Liyuan Wang. 2021. Deep feature
                                                          embedding. Neurocomputing, volume 568.
 transfer learning for automatic pronunciation
 assessment. In Proceedings of Interspeech               Hugo Touvron, Louis Martin, Kevin Stone, Peter
 (INTERSPEECH), pages 4438–4442.                          Albert, Amjad Almahairi, et al. 2023. Llama 2: Open
                                                          foundation and fine-tuned chat models. arXiv preprint
Wei Liu, Kaiqi Fu, Xiaohai Tian, Shuju Shi, Wei Li,
                                                          arXiv:2307.09288.
 Zejun Ma, and Tan Lee. 2023. An ASR-free fluency
 scoring approach with self-supervised learning. In      Nidhi Vakil and Hadi Amiri. 2023. Curriculum
 Proceedings of the IEEE International Conference on      Learning for Graph Neural Networks: A Multiview
 Acoustics, Speech and Signal Processing (ICASSP),        Competence-based Approach. In Proceedings of the
 pages 1–5.                                               Annual Meeting of the Association for Computational
                                                          Linguistics (ACL), pages 7036–7051.
Tien-Hong Lo, Fu-An Chao, Tzu-I Wu, Yao-Ting Sung,
 and Berlin Chen. 2024. An effective automated           Alistair Van Moere and Ryan Downey. 2016.
 speaking assessment approach to mitigating data            Technology and artificial intelligence in language
 scarcity and imbalanced distribution. In Findings of       assessment. Handbook of second language
 the Association for Computational Linguistics:             assessment, pages 341–358.
 NAACL, pages 1352–1362.
                                                         Yihao Wang, Zhongdi Wu, Joseph Nese, Akihito
Pamela M Rogerson-Revell. 2021. Computer-assisted         Kamata, Vedant Nilabh, Eric C. Larson. 2025. A
  pronunciation training (CAPT): Current issues and       unified model for oral reading fluency and student
  future directions. RELC Journal, volume 52, pages       prosody. In Proceedings of the IEEE International
  189–205.                                                Conference on Acoustics, Speech and Signal
                                                          Processing (ICASSP), pages 1–5.
John M. Norris and Larry Davis. 2025. Assessing
   second language speaking at ETS: Introduction. In:    Ke Wang, Lei He, Kun Liu, Yan Deng, Wenning Wei,
   Challenges and Innovations in Speaking                 Sheng Zhao. 2025b. Exploring the potential of large
   Assessment. Routledge, pages 1–18.                     multimodal models as effective alternatives for
                                                          Pronunciation assessment. in arXiv preprint
Silke M. Witt and S. J. Young. 2000. Phone-level
                                                          arXiv:2503.11229.
 pronunciation scoring and assessment for interactive
 language learning. Speech Communication, volume         Benjamin Warner, Antoine Chaffin, Benjamin Clavié,
 30, pages 95–108.                                        Orion Weller, Oskar Hallström, et al. 2024. Smarter,
                                                          better, faster, longer: A modern bidirectional encoder
Pieter Mülller, Febe de Wet, Christa van der Walt, and
                                                          for fast, memory efficient, and long context
 Thomas Niesler. 2009. Automatically assessing the
                                                          finetuning       and    inference.   arXiv     preprint
 oral proficiency of proficient L2 speakers. In
                                                          arXiv:2412.13663.
 Workshop on Speech and Language Technology in
 Education (SLaTE), pages 29–32.                         Bi-Cheng Yan and Berlin Chen. 2024. An effective
                                                          hierarchical graph attention network modeling
Hao-Chen Pei, Hao Fang, Xin Luo, Xin-Shun Xu. 2024.
                                                          approach for pronunciation assessment. IEEE/ACM
 Gradformer: A framework for multi-aspect multi-
                                                          Transactions on Audio, Speech, and Language
 granularity pronunciation assessment. IEEE/ACM
                                                          Processing, volume 32, pages 3974–3985.
 Transactions on Audio, Speech, and Language
 Processing, volume 32, pages 554–563.                   Bi-Cheng Yan, Jiun-Ting Li, Yi-Cheng Wang, Hsin-
                                                          Wei Wang, Tien-Hong Lo, Yung-Chang Hsu, Wei-
Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji
                                                          Cheng Chao, and Berlin Chen. 2024. An Effective
 Watanabe. 2022. Branchformer: Parallel mlp-
                                                          Pronunciation Assessment Approach Leveraging
 attention architectures to capture local and global
                                                          Hierarchical Transformers and Pre-training Strategies.
 context for speech recognition and understanding. In
                                                          In Proceedings of the Annual Meeting of the
 International Conference on Machine Learning
                                                          Association for Computational Linguistics (ACL),
 (PMLR), pages 17627–17643
                                                          pages 1737–1747.
Yang Shen, Ayano Yasukagawa, Daisuke Saito,
                                                         Klaus Zechner, and Keelan Evanini. 2019. Automated
 Nobuaki Minematsu, and Kazuya Saito. 2021.
                                                          speaking assessment: Using language technologies to
 Optimized prediction of fluency of L2 English based
                                                          score spontaneous speech. Routledge.
 on interpretable network using quantity of phonation
 and quality of pronunciation. In Proceedings of IEEE    Biao Zhang and Rico Sennrich. 2019. Root mean
 Spoken Language Technology Workshop (SLT),               square layer normalization. In Proceedings of
 pages 698–704.                                           Advances in Neural Information Processing Systems
                                                          (NeurIPS), volume 32.
Junbo Zhang, Zhiwen Zhang, Yongqing Wang,               Drawing on this research trend, various neural
 Zhiyong Yan, Qiong Song, Yukai Huang, Ke Li,           models capitalizing on hand-crafted features and/or
 Daniel Povey, and Yujun Wang. 2021.                    self-supervised features have been extensively
 Speechocean762: An open-source non-native English
                                                        investigated in existing literature, including
 speech corpus for pronunciation assessment. In
 Proceedings of Interspeech (INTERSPEECH), pages        parallel (Gong et al., 2022; Chao et al., 2022),
 3710 –3714.                                            hierarchical (Do et al., 2023, Yan et al., 2024), and
                                                        linguistic-decoupled structures (Pei et al., 2024).
Jian Zhu, Cong Zhang, and David Jurgens. 2022.
 Phone-to-audio alignment without text: A semi-         Unscripted-speech Assessment. Unscripted-
 supervised approach. In Proceedings of the IEEE        speech assessment is an emerging research field
 International Conference on Acoustics, Speech and      and has gained increasing attention in recent years,
 Signal Processing (ICASSP), pages 8167–8171.           as it attempts to qualify learners’ speaking abilities
                                                        in real-world communication. The corresponding
A Related Work                                          developments target free-speaking scenarios, in
Automatic Pronunciation Assessment (APA)                which an L2 learner receives a reference text (with
quantifies L2 learners' pronunciation proficiency in    short questions) and is expected to respond or share
a target language, offering either analytic scores      opinions grounded in their personal experiences.
(continuous numerical values for specific aspects)      Based on the free-form and spontaneous speech
or an overall score (discrete categorical values for    inputs, the assessment models then qualify oral
speaking competence). We categorize the related         skills and provide instructive feedback at various
APA works into the following two groups for             linguistic levels. As one of the initial attempts, Liu
discussion, differentiated by their reliance on         et al. (2023) proposed an ASR-free method which
reference text.                                         leveraged a pre-trained self-supervised learning
                                                        (SSL) model (viz., wav2vec2.0) to estimate
Scripted-speech Assessment. The developments            fluency scores for L2 learners without resorting to
of scripted-speech assessment are typically             the reference texts (or ASR transcriptions). Their
designed in read-aloud learning scenarios, where        method extracted frame-level acoustic features
an L2 learner is provided with a reference text and     with a self-supervised learning (SSL) model and
instructed to pronounce it verbatim. Early efforts in   generated phonology features by assigning
scripted speech assessment predominantly focused        proximity phone labels (cluster index) to each
on single-aspect assessment, which predicted            frame via K-means clustering. To assess word-
proficiency scores at specific linguistic levels with   level speaking skills, a pioneering effort, Chen et al.
various sets of hand-crafted features by separate       (2024) proposed MultiPA which extracted
scoring modules, such as phone-level accuracy           pronunciation features at word-level with two
(Witt and Young, 2000), word-level stress (Ferrer       speech recognizers and employed a bottom-up
et. al., 2015), and utterance-level fluency             neural structure to examine the learner’s
(Coutinho et. al., 2016). Furthermore, the              pronunciation skills at both word and utterance
commonly used hand-crafted features were derived        levels. Specifically, one recognizer employs a
from the reference text in conjunction with the         high-performing ASR model (whisper-large-v3) to
learner's speech via an ASR model (hybrid DNN-          approximate the ground-truth word sequence,
HMM system), where the extracted pronunciation          while the other utilizes an ASR model trained with
features included acoustic features, confidence         native speaker speech (whisper-medium-en) to
scores of recognized linguistic units, time-            emulate how a native speaker would process the
alignment information, and statistical measures,        learner's speech. Compared to previous works, our
but were not limited to these (Mülller et al., 2009;    model assesses oral skills from phone-level to
Franco et al., 2010). To provide comprehensive          utterance-level by working in tandem with a speech
pronunciation feedback for language learners, a         recognizer and a G2P converter. Furthermore, to
flurry of recent work has advocated multi-aspect        mitigate the detrimental effects of ASR errors, we
and multi-granular pronunciation assessment,            proposed the extract score-discriminative features
which evaluates pronunciation proficiency across        by leveraging the contrastive ordinal regularizer.
multiple linguistic levels (viz. phoneme, word, and
utterance), with diverse aspects (e.g., accuracy,
fluency, and completeness) with a unified model.
                                                       model, first extracts frame-level SSL features,
                                                       subsequently assigning phonetic information to
B Implementation Details                               each frame with k-means clustering and then
                                                       evaluating utterance-level pronunciation aspects
This section illustrates the implementation details    via a simple mean pooling mechanism. 2) ASR-
of our experiments, and we plan to make our source     based method: MultiPA (Chen et al., 2024) extracts
code and datasets publicly accessible after the        pronunciation features based on two speech
reviewing process.                                     recognizers and constructs an assessment model to
Training Hyperparameters. Our implementation           qualify oral skills at word and utterance levels. 3)
follows previous studies (Gong et al., 2022; Chao      Variants of HiPPO: Parallel-CTC, Parallel-LLaMA,
et al., 2022), employing the Adam optimizer, with      and Hier-LLaMA adopt the same inputs as HiPPO
a learning rate of 0.001, and a batch size of 25. To   (i.e., X ! and 𝐱 112 ), while exploring different neural
stabilize the training process, the aspect scores at   architectures. Parallel-CTC and Parallel-LLaMA
both the utterance and word levels are normalized      adopt a parallel architecture, with the former using
to match the scale of the phone-level score, ranging   Transformer blocks and the latter stacking of
from 0 to 2. We conducted 5 independent trials,        LLaMA blocks. Hier-LLaMA replaces the Conv-
with each trial running for 100 epochs and using a     LLaMA blocks of HiPPO with standard LLaMA
different random seed to reduce the impact of          blocks.
randomness. The evaluation metrics are reported as
                                                       Comparative         Models     for       Read-aloud
the average of the best-performing epochs across
                                                       Assessment. For read-aloud scenario, we first
these trials, selected based on the minimum phone-
                                                       report the performance of Azure Pronunciation
level MSE values.
                                                       Assessment (AzurePA) service (Wang et al.,
Model Configurations. In the Conv-LLaMA                2025b), followed by a comparison with several
block, the multi-head self-attention (MHSA)            APA models. 1) Parallel neural structure: GOPT
module is configured with 1 head and 24 hidden         (Gong et al., 2022) adopts pronunciation features
units for different granularities (𝑑/ = 𝑑7 = 𝑑3 =      derived from phone-level timestamps and models
24). The attention pooling mechanisms at the word      the pronunciation aspects with Transformer blocks;
and utterance levels share the same configuration,     3M (Chao et al., 2022) extends GOPT by
which use a single-layer multi-head attention          incorporating acoustic features, i.e., phone duration
mechanism with 3 heads and 24 hidden units.            statistics and SSL-based features, and phonology
Furthermore, in each modeling stage, the               features, i.e., vowel and consonant embeddings. 2)
regressors for various pronunciation aspects are       Hierarchical neural structure: HiPAMA (Do et al.,
implemented as feed-forward networks, each             2023) is a language hierarchy-aware APA model
consisting of two linear transforms with a non-        equipped with the trait attention mechanisms;
linear activation in between, and the second           HierTFR (Yan et al., 2024) stacked a hierarchical
transform of each projects the hidden dimension to     neural structure via Transformer blocks and
a single scalar output.                                proposed mask prediction to strengthen the
                                                       relationships across granularities for model
C Comparative Methods                                  initialization.

We compare HiPPO with several top-of-the-line          D Score Assignments for Speechocean762
methods in both simulated free-speaking and read-        Corpus in the Simulated Free-speaking
aloud scenarios.                                         Scenario
Comparative Models for Simulated Free-
                                                       Speechocean762 was curated in a read-aloud
speaking Assessment. First, for the free-speaking
                                                       learning scenario, where human annotators
scenario, we compare three categories of methods.
                                                       provided pronunciation scores at the utterance,
1) ASR-free methods: VanillaSSL (Chen et al.,
                                                       word, and phone levels. These scores were
2024) qualifies utterance-level pronunciation
                                                       assigned based on the reference text (for scoring
aspects by fine-tuning a pre-trained self-supervised
                                                       utterance-level and word units) and the
learning (SSL) model (viz., wav2vec2.0); similarly,
                                                       corresponding canonical phone sequence (for
Liu et al. (2023), based on a SSL-based acoustic       scoring     phone    units).   To      reorganize
Speechocean762 in the simulated free-speaking                 • Reference Text and                    • ASR Transcription and
scenario, we first use a speech foundation model                Canonical phones                          the G2P outputs

(whisper-large-v3) to transcribe the learners’         Ref: A               C             E    Trans: A            F                 G
speech and convert the transcriptions into the
corresponding phone sequences via a G2P               Cano: p1         p2       p3        p4    G2P: p1           p6            p4       p7

converter (g2pE). Next, for each recording, we             (a) A Reference Text and an ASR Transcription, Each with Their
                                                                           Corresponding Phone Sequence
align the ASR transcription to reference text and
the converted phone sequence to canonical phone         • Word-level Alignment                         • Phone-level Alignment
                                                                  ⦰
sequence, respectively. Based on the resulting            A                 C        E           p1         p2      p3          p4       ⦰

alignments, we first assigned pronunciation scores
                                                                            ⦰                               ⦰
from human annotators to correctly recognized             A        F                 G           p1                 p6          p4       p7

segments (i.e., including phone and word units).          C        I        D        S           C          D           S       C           I
For subsequent score assignments, we handled            (b) Word Score Assignments                    (c) Phone Score Assignments
ASR errors at both levels (viz., word and phone                                          Alignment Operations
levels) as follows:                                               Correctly
                                                              C Recognize        S Substitution         I   Insertion       D    Deletion

• Deletion Errors: Ignored due to there are no
    corresponding segments in the transcribed        Figure 8: Illustration of score assignments for
    words (or converted phones).                     Speechocean762 in simulated free-speaking
• Substitution Errors: Assigned scores based on      scenarios. For a sample recording, we demonstrate
    aligned segments, as most substitution error     the assignment process: (a) shows the reference
                                                     text and an ASR transcription, along with their
    cases reflect subtle acoustic differences.
                                                     respective canonical and G2P converted phone
• Insertion Errors: Assigned a score of zero.
                                                     sequences; (b) alignment of ASR transcription to
Figure 8 presents the alignment process for a        reference text for human scoring; (c) alignment of
sample recording. Note that insertion errors are     G2P outputs to canonical phones for pronunciation
retained, owing to the maintenance of phone-to-      scoring.
word mappings for developing hierarchical neural
structure. Figure 8(c) highlights how the score
assignment process maintains phone-to-word
relationships for converted phones (i.e., the
mapping of phone segments p6 and pK to word G).
