               End-to-End Mispronunciation Detection
                and Diagnosis From Raw Waveforms
                                                     Bi-Cheng Yan, Berlin Chen
                                                   Dept. of Computer Science and
                                                      Information Engineering
                                                 National Normal Taiwan University
                                                           Taipei, Taiwan
                                                 {80847001s, Berlin}@ntnu.edu.tw

    Abstract—Mispronunciation detection and diagnosis (MDD)         text prompt, ERN can readily offer appropriate diagnosis
is designed to identify pronunciation errors and provide            feedback on mispronunciations [9][10]. However, it is
instructive feedback to guide non-native language learners,         practically difficult to enumerate and include sufficient
which is a core component in computer-assisted pronunciation        phonological rules into the decoding network for different L1-
training (CAPT) systems. However, MDD often suffers from the        L2 language pairs. Furthermore, inclusion of too many
data-sparsity problem due to that collecting non-native data and    phonological rules would incur ASR accuracy drop and in turn
the associated annotations is time-consuming and labor-             lead to poor MD performance. Apart from the above, a
intensive. To address this issue, we explore a fully end-to-end     common thought is that we can evaluate learners’
(E2E) neural model for MDD, which processes learners’ speech
                                                                    pronunciations based on free phone recognition. An MDD
directly based on raw waveforms. Compared to conventional
hand-crafted acoustic features, raw waveforms retain more
                                                                    system is thus trained to directly recognize the possible
acoustic phenomena and potentially can help neural networks         sequence of phones pronounced by a non-native learner,
discover better and more customized representations. To this        which can be compared to canonical phone sequence of a
end, our MDD model adopts a co-called SincNet module to take        given prompt [11][12][13]. Problems facing this approach
input a raw waveform and covert it to a suitable vector             include scarce annotated training data for deviated
representation sequence. SincNet employs the cardinal sine (sinc)   pronunciations and the variability in how different speakers
function to implement learnable bandpass filters, drawing           articulate each phone. Alternatively, we can conduct MDD by
inspiration from the convolutional neural network (CNN). By         recognizing articulatory features instead of phone symbols
comparison to CNN, SincNet has fewer parameters and is more         [14][15] or employing a multi-task learning strategy to
amenable to human interpretation. Extensive experiments are         leverage additionally speech data compiled from L2 speakers’
conducted on the L2-ARCTIC dataset, which is a publicly-            native languages [16].
available non-native English speech corpus compiled for
research on CAPT. We find that the sinc filters of SincNet can          Recently, several efforts have been made to utilize deep
be adapted quickly for non-native language learners of different    neural networks to learn complex and abstract representations
nationalities. Furthermore, our model can achieve comparable        directly based on speech waveforms for ASR [17][18].
mispronunciation detection performance in relation to state-of-     Compared with using traditional hand-crafted acoustic
the-art E2E MDD models that take input the standard hand-           features, raw waveform modelling allows the incorporation of
crafted acoustic features. Besides that, our model also provides    more information cues, such as the phase spectrum
considerable improvements on phone error rate (PER) and             information that is typically ignored in Mel-scale frequency
diagnosis accuracy.                                                 cepstral coefficients (MFCCs) derived from Fourier transform
                                                                    magnitude-based spectra. Furthermore, such human-
    Keywords—computer assisted pronunciation training (CAPT),       engineered features, in fact, are originally designed from
mispronunciation detection and diagnosis (MDD), raw waveforms,      perceptual evidence and there is no guarantee that such
sincnet                                                             representations are really suitable for all speech-related tasks.
                     I. INTRODUCTION                                Directly processing the raw waveform allows neural network
                                                                    based methods to learn low-level acoustic representations that
With accelerating globalization, more and more people are           are possibly more tailored to a specific task. Palaz et al. [17]
willing or required to learn second languages (L2).                 investigated the usefulness of raw waveform-based models on
Developments of computer-assisted pronunciation training            the TIMIT phone recognition task, which showed that CNNs
(CAPT) systems open up new possibilities to enable L2               could deliver superior performance over fully connected
learners to practice L2 pronunciation skills in an effective and    networks. Ravanelli et al. [18] introduced a parametrized sinc
stress-free manner [1][2]. As an integral component of a            functions in replace of the convolution operators of CNNs and
CAPT system, mispronunciation detection and diagnosis               proposed the so-called SincNet architecture, which has shown
(MDD) manages to provide different kinds of information,            promising results on various speaker identification,
such as pronunciation scores [5] and diagnosis feedback [6],        verification and phone recognition tasks. In the context of
to guide non-native language learners to practice their             MDD, Yang et al. [15] proposed an unsupervised approach
pronunciations. Pronunciation scores reflect how similar an         and argued that the incorporation of large-scale unlabeled
L2 learner’s pronunciation is to that of native speakers. In        native speech data in the training stage can overcome the data
practice, pronunciation scoring based methods detect errors         sparsity problem for MDD. The authors employed a
using confidence measures that are derived from the automatic       contrastive predictive coding (CPC) model trained with
speech recognizer (ASR), e.g., phone durations, phone               language-adversarial training criteria to align the feature
posterior probability scores and segment duration scores            distributions between the L1 and L2 speech datasets, yielding
[7][8]. In order to obtain informative diagnosis feedback, the      accent-invariant speech representations for MDD.
extended recognition network (ERN) method augments the
decoding network of ASR with phonological rules. By                 In this paper, we develop a fully end-to-end (E2E) neural
comparison between an ASR output and the corresponding              model architecture that streamlines the MDD process by



XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE
                                              …    '$%!    #$                       𝑝(#( (𝐲|X) = C 𝑝(𝐲|𝒛, X)𝑝(𝒛|X),                 (2)
                                                                                                       𝒛
                    CTC
                                                                                                ≈ C 𝑝(𝐲|𝒛)𝑝(𝒛|X),                   (3)
                                  Attention
                                                                                                       𝒛
             "!     "" … "#                                                                             %
                                              … $$%!        %$
                                    &$%!                                             𝑝)** (𝐲|X) = F 𝑝(𝑦$ |𝑦!:$'! , X).              (4)
                  Encoder                            Decoder
                                                                                                       $,!
             SincNet/CNN                                                 where the frame-wise latent variable sequences 𝒛 belongs to a
                                              … #$%"      #$%!           canonical phone set 𝒰 augmented with and the additional
                     X                                                   <blank> label, which facilitates CTC to enforce a monotonic
  Fig. 1. Mispronunciation detection and diagnostic with a hybrid CTC-   behavior of phone-level alignments. Eq. (3) is a neat
            Attention ASR architecture from raw waveforms.               formulation of the CTC model, which is derived from the
directly taking input the waveform signals of L2 learners’               assumption that the symbol translation model 𝑝(𝐲|𝒛) is
speech. Specifically, an E2E free phone recognition system is            conditionally independent of the input sequence X [20]. The
                                                                         linear combination weight 𝛼 in Eq. (1) is a hyperparameter
adopted to generate the possible phone sequence uttered by an
                                                                         used to linearly interpolate the two posterior probabilities. In
L2 learner, which is instantiated by a hybrid connectionist
temporal classification/attention (CTC-ATT) model [19].                  our experiments, the 𝛼 is set equal to 0.5.
Further, we insert a SincNet (or CNN) module prior to the                The attention-based model can be expressed by
encoder of CTC-ATT model, which aims to covert a raw
waveform signal to a suitable vector representation sequence.                               H = Encoder(X),                         (5)
The overall model architecture is depicted in Fig. 1. By                                𝛂$,. = Attention(𝐪$ , 𝐡. ),                 (6)
comparison to CNN, SincNet has fewer parameters and is
more amenable to human interpretation. Empirical
                                                                                             𝐜$ = C 𝛂$,. 𝐡. ,                       (7)
evaluations conducted on the L2-ARCTIC dataset
                                                                                                        .
demonstrate that our proposed SincNet-based E2E MDD
model significantly reduces the phone error rate (PER).                     𝑝(𝑦$ |𝑦!:$'! , H) = Decoder(𝐪$'! , [𝑦$'! ; 𝐜$'! ]),     (8)
Furthermore, our best model can achieve comparable
mispronunciation detection performance in relation to these              In order to constrain the range of the attention operations, we
state-of-the-art models.                                                 choose the location-aware attention [19] as the attention
                                                                         mechanism in our experiments, as indicated in Eq. (7).
           II. SINCNET-BASED E2E MDD MODEL
                                                                         B. The SincNet Architecture
A. Hybrid CTC-ATT ASR Model                                              SincNet [18] is a parametric counterpart used to replace the
                                                                         convolution operations of CNN. Each impulse response of
The basic hybrid CTC-ATT MDD model typically consists
                                                                         SincNet’s filters is a subtraction of two cardinal sine (sinc)
of four modules, as depicted in Fig. 1: 1) an encoder module
                                                                         functions, resulting in an ideal bandpass filter. The standard
that is shared across CTC and the attention-based model. The             CNN perform a set of time-domain convolution operations on
encoder module extracts 𝑆-length high-level encoder state                the input with some finite impulse response filters defined as:
sequence H = (𝐡! , … , 𝐡" ) from a 𝑇-length acoustic features
X = (𝐱! , … , 𝐱 # ) through a stack of convolutional and/or                     𝑠[𝑡] = 𝑥[𝑡] ∗ ℎ[𝑡] = ∑%'!
                                                                                                      $,/ 𝑥[𝑙] ∙ ℎ[𝑡 − 𝑙].          (9)
recurrent networks, where 𝑆 ≤ 𝑇 is due to downsampling; 2)
an attention module that calculates a fixed-length context               where 𝑥[𝑡] is a chunk of the speech signal, ℎ[𝑡] is a filter of
vector 𝐜$ by summarizing the output of the encoder module                𝐿-length, and 𝑠[𝑡] is the filtered output. In this case, all the
at each output step for 𝑙 ∈ [1, … , 𝐿], finding out relevant             elements of ℎ[∙] are learnable parameters (i.e., all the L
parts of the encoder state sequence to be attended for                   elements of each filter are learned from data). SincNet
predicting an output phone symbol y$ , where the output                  proposes to replace filters ℎ with a sinc function 𝑔 that only
symbol sequence 𝐲 = (𝑦! , … , 𝑦% ) belongs to a canonical                depends on two variables: low and high cut-off frequencies. A
                                                                         filter in SincNet with impulse response 𝑔[𝑡, 𝑓! , 𝑓0 ] and
phone set 𝒰; 3) Given the context vector 𝐜$ and the history of
                                                                         frequency response 𝐺[𝑓, 𝑓! , 𝑓0 ] can be respectively express by:
partial diagnostic results 𝑦!:$'! , a decoder module updates its
hidden state 𝐪$ autoregressively and estimates the next phone
                                                                           𝑔[𝑡, 𝑓! , 𝑓0 ] = 2𝑓0 𝑠𝑖𝑛𝑐(2𝜋𝑓0 𝑡) − 2𝑓! 𝑠𝑖𝑛𝑐(2𝜋𝑓! 𝑡),   (10)
symbol 𝑦$ ; 4) The CTC module offers another diagnostic
results based on the frame-level alignment between the input                                                 1    1
sequence X and the canonical phone symbol sequences 𝐲 by                            𝐺[𝑓, 𝑓! , 𝑓0 ] = Π e01 f − Π e01 f.            (11)
                                                                                                             !        "
introducing a special <blank> token. It can substantially                                    234 (7)
reduce irregular alignments during the training and test                 where 𝑠𝑖𝑛𝑐(𝑥) = 7 , 𝑓! and 𝑓0 are two learnable
phases.                                                                  parameters that describe low and high cutoff frequencies, 𝑓 is
   The training objective function of the hybrid CTC-ATT                 a frequency index, and Π(∙) is a rectangular bandpass filter
                                                                         function in the magnitude frequency domain.
model is to maximize a logarithmic linear combination of the
posterior probabilities predicted by CTC and the attention-
based model, i.e., 𝑝(#( (𝐲|X) and 𝑝)** (𝐲|X):
     ℒ = 𝛼log𝑝(#( (𝐲|X) + (1 − 𝛼)log𝑝)** (𝐲|X),                    (1)
      TABLE I.           STATISTICS OF THE EXPERIMENTAL SPEECH
                                CORPORA.
                                                                #Marked
   Subsets       #Spks       #Utters    #Hours     #Phones
                                                                  errors
                                                                13K
   Train         477         6237       8.81       234K
                                                                (0.56%)
                                                                4K                           (a)                                (b)
   Test          6           900        0.87       15K
                                                                (13.65%)

     TABLE II.           CONFUSION MATRIX OF MISPRONUNCIATION
                         DETECTION AND DIAGNOSIS.

                                                 Ground truth
           Total condition
                                                                                              (c)                               (d)
                                            CP                  MP

                                       True positive      False positive
                              CP
                                           (TP)               (FP)
   Model prediction
                                       False negative     True negative
                              MP
                                            (FN)          (TN=CD+DE)

                                                                                             (e)                                (f)
    III. EXPERIMENTAL SETTINGS AND PERFORMANCE
                                                                            Fig. 2. Visualization of learned filters in the conventional CNN and
                 EVALUATION METRICS                                         SincNet. Two filters were randomly chosen and highlighted with
A. Speech Corpora and Model Configuration                                   different colors; the remaining filters were plotted with shade grey. (a)
                                                                            and (b) are impulse responses for CNN and SincNet; (c) and (d) are
We carried out experiments using the L2-ARCTIC [21] and                     frequency responses for CNN and SincNet; and (e) and (f) show the
TIMIT corpus [22] for MDD tasks. The L2-ARCTIC dataset                      normalized average frequency response for CNN and SincNet.
is a publicly-available non-native English speech corpus
compiled for research on CAPT, accent conversion, and others.              B. Performance Evaluation Metrics
It contains correctly pronounced utterances and                            For the mispronunciation detection task, we follow the
mispronounced utterances of 24 non-native speakers (12                     hierarchical evaluation structure adopted in [5], while the
males and 12 females), whose L1 languages include Hindi,                   corresponding confusion matrix for four test conditions is
Korean, Mandarin, Spanish, Arabic and Vietnamese. Apart                    illustrated in Table II. Based on the statistics accumulated
from that, a suitable quantities of native (L1) English speech             from the four test conditions, we calculate the values of
datasets compiled from the TIMIT corpus (composed of 630                   different metrics like recall, precision and the F-1 measure
speakers) was used to bootstrap the training of the various E2E            (the harmonic mean of the precision and recall), so as to
MDD models. To unify the phone sets of these two corpora,                  evaluate the performance of mispronunciation detection.
we followed the definition of the CMU pronunciation
                                                                           Those metrics are defined as follows:
dictionary to obtain an inventory of 39 canonical phones. Next,
we divided these two corpora into training, development and                                                     TN
test sets, respectively; in particular, the setting of the                                   Precision =             ,                          (12)
                                                                                                              TN + FN
mispronunciation detection experiments on L2-ARCTIC
followed the recipe provided by [13]. Table I summarizes                                                      TN
detail statistics of these speech datasets.                                                     Recall =           ,                            (13)
                                                                                                            TN + FP
    Our baseline E2E MDD models built on the hybrid CTC-
ATT model. The encoder network is composed of the VGG-                                               Precision ∗ Recall
                                                                                          F-1 = 2                       .                       (14)
based deep CNN component plus a bidirectional LSTM                                                   Precision + Recall
component with 1024 hidden units [19], which takes input the
                                                                           Furthermore, to calculate the diagnostic accuracy rate (DAR),
hand-crafted acoustic features, such as Mel-filterbank outputs
(FBANK) or MFCCs. The decoder network consists of two-                     we focus on the cases of TN and consider it as combination
layer unidirectional-LSTM with 1024 cells. As to the hand-                 of diagnostic errors (DE) and correct diagnosis (CD). The
crafted acoustic features, FBANK is 80-dimensional while                   accuracies of mispronunciation diagnosis rate (DAR) are
MFCCs 40-dimensional. Both of them were extracted from                     calculated by:
waveform signals with a hop size of 10 ms and a window size                                                  CD
of 25 ms, and further normalized with the global mean and                                          DAR =          .                            (15)
variance. When taking input raw waveform signals                                                           CD + DE
alternatively, the SincNet module is tacked in front of the
encoder network. SincNet module was based on the                                            IV. EXPERIMENTAL RESULTS
configuration suggested in [18], which is made of an array of              A. Interpretation of the learned filters
parametrized sinc-functions in the first layer, followed by two
one-dimensional convolutional layers. Further, each layer of               At the outset, we analyze the dynamic properties of the learned
SincNet has 80, 128 and 128 filters and kernel size are set to             filters in the first layer of the CNN (or SincNet) module,
be 251, 3 and 3, respectively.                                             respectively. Fig. 2 illustrates the impulse and frequency
                                                                           responses of the learned filters, which were trained on the
                                                                           correctly-pronounced training utterances of L2 learners. In the
   TABLE III.     %PER FOR CORRECT PORNUNCIATION UTTERENCES
           WITH DIFFERENT FRONTEDN PROCESSING SCHEMES.

                  MFCC          FBANK              CNN         SincNet

    %PER            9.25           8.45             6.44        5.50


     TABLE IV.       MISPRONUNCIATION DETECTION RESULTS FOR
                         DIFFERENT MODELS.

                  Input
     Model                    %Recall             %Precision      %F1
                 Feature

    GOP          FBANK          52.88               35.42         42.42

    CTC-          MFCC          53.54               53.64         53.59
                                                                               Fig. 3. Normalized average frequency response for different L1-
    ATT          FBANK          52.43               55.31         53.83          dependent pronunciation distributions extracted by SincNet.

    CTC-
                                                                          of SincNet capture mainly the property of pitch regions (the
    ATT           RAW           47.60               55.15         51.10   average pitch is 133 Hz for a male and 234 for a female) and
    +CNN                                                                  cover various English vowels regions (the first and second
                                                                          formant of English vowels approximately located at 500 Hz
    CTC-                                                                  and 1,300 Hz, respectively).
    ATT           RAW           50.09               55.31         52.57
    +SincNet                                                                  Next, in order to examine the properties of SincNet filters
                                                                          learned for the L1-dependent pronunciation distributions of
    TABLE V.    DETAILS OF MISPRONUNCIATION DETECTION AND                 L2 learners with different nationalities, we use all of training
           DIAGNOSIS RESULTS FOR DIFFERENT MODELS.                        data to build a multilingual SincNet module and then transfer
                                                                          it with the L1 speech data of L2 learners to obtain their
                    Correct
                 pronunciations
                                        Mispronunciations                 respective L1-dependent SincNet module. The corresponding
     Models                                                      %DAR     normalized average frequency response be demonstrated in
                 %TP       %FN            %TN         %FP                 Fig. 3. As can be observed, all of the L1-dependent filters
                                                                          operate in frequency bins below 2,000 Hz, which is consistent
    Leung et
    al.* [11]
                 67.81     32.19          65.04       32.96      32.10    with the perceptual scales that take inspiration from the human
                                                                          auditory system. This implies that the learned filters
    CTC-
                 89.79     10.21          52.43       47.57      59.84    essentially were L1-dependent and selective in processing
    ATT†                                                                  those spectral components. Further, each filter seemingly
    CTC-                                                                  learns to capture different English vowels, in terms of their
    ATT          90.67      9.33          47.60       52.40      62.08    first or second formants. For example, the filters of American,
    +CNN                                                                  Arabic, Hindi and Spanish clearly manifest the vowel /a/
                                                                          according to its second formant (the second formant of vowel
    CTC-
    ATT          90.25      9.75          50.09       49.91      60.96
                                                                          /a/ centers around 1,100Hz). Furthermore, the filters of
    +SincNet                                                              Mandarin and Korean tend to highlight the frequencies below
  Note: *We reproduced the model architecture in the framework of Leung
                                                                          800 Hz. We argue that these filters learn to represent the vowel
  et al by adopting FBANK features for the CNN-RNN-CTC model, so          /u/ or /ɔ/ (the second formant of vowel /u/ and /ɔ/ are below
  there may exists some slight differences with [11]. †We report on the   800 Hz). It is worth mention that the L1 speech dataset for
  CTC-ATT model that takes input the FBANK features.                      each nationality of the L2 learners are approximate 4 hours.
first raw, we depict the learned filters in the time domain. The          This indicates that the filters involved in SincNet can be
learned filters of CNN are a set of filters with 𝐿 parameters             adapted quickly for different non-native language learners.
(i.e., 𝐿 = 251 ). Thus, the impulse response for CNN                      B. Comparison of Phone Recognition Performation
seemingly learn to represent temporal property of a waveform
                                                                          We report the phone error rate (PER) for CTC-ATT model
and its corresponding frequency response looks without
particular patterns, which is uniformly distributed in low and            trained with raw waveform modeling (CNN, SincNet) or
high frequency bins (cf. Fig. 2 (a) and (c)). SincNet, instead            traditional hand-crafted acoustic features (MFCC, FBANK).
making use of a set of sinc functions which are designed to               In this experiment, the testing utterances were subset of testing
implement rectangular bandpass filters, demonstrate                       data set, which adopt correct part of pronounced utterances
regularity in the frequency domain. Looking at Fig. 2 (d), we             only. The results are summarized in Table III. We can observe
                                                                          that both of raw waveform models significantly outperform
can find that most the filters of SincNet lie on frequencies
below 2,000 Hz. This implies that SincNet learns to reflect the           the traditional features models. SincNet can lead to lowest
properties of human’s auditory system. We then turn to                    PER, at least have 3% absolute PER reduction while
examine the normalized average frequency responses of CNN                 compared to the traditional hand-crafted acoustic features.
                                                                          Next. The performance degraded when SincNet was swapped
and SincNet. Fig. 2 (e) and (f) depict the normalized average
frequency responses of the filters learned by CNN and                     with a standard CNN.
SincNet, respectively. There are several obvious peaks                    C. Performance of Mispronunciation Detection
standing out in the plot of SincNet. The observation to some              We assess the performance levels for various different input
extent is consistent with [18], which mentioned that the filters          formats for E2E MDD, i.e., FBANK features, MFCCs, and
raw waveform signals, with respect to mispronunciation                                           REFERENCES
detection. We also report the pronunciation scoring based
MDD method, namely the GOP method building on the                   [1]  K. Shi, et al., “Computer-Assisted Language Learning System:
hybrid DNN-HMM ASR model. Specifically, the DNN                          Automatic Speech Evaluation for Children Learning Malay and Tamil,”
component of GOP is a 5-layer time-delay neural network                  in Proc. INTERSPEECH, pp. 1019–1020, 2020.
(TDNN) with 1,280 neurons in each layer. The corresponding          [2] Y. Xie, et al., “A Mandarin L2 Learning APP with Mispronunciation
results are shown in Table IV. As can be seen, the E2E ASR               Detection and Feedback,” in Proc. INTERSPEECH, pp. 1015–1016,
based methods clearly surpass the GOP-based method, with at              2020.
least 10% absolute improvements in terms of the F-1 measure.        [3] X. Feng, et al., “A Dynamic 3D Pronunciation Teaching Model Based
                                                                         on Pronunciation Attributes and Anatomy” in Proc. Interspeech, pp.
This implies that the free phone recognition method can boost            1023–1024, 2020.
the performance on mispronunciation detection. Next, as             [4] S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and
traditional hand-crafted acoustic features are taken as the input        assessment for interactive language learning,” Speech Communication,
to the encoder network of the E2E MDD model, we can find                 pp. 95–108, 2000.
that the performance of FBANK is on par with MFCCs.                 [5] K. Li, X. Qian and H. Meng, “Mispronunciation detection and
FBANK model stands out in performance when precision is                  diagnosis in l2 english speech using multidistribution deep neural
used as the evaluation metric, whereas the situation is reversed         networks,” IEEE/ACM Transactions on Audio, Speech, and Language
                                                                         Processing, vol. 25, 193–207, 2016.
when recall is used as the metric. Finally, we compare the
performance of CNN and SincNet, both of which consume               [6] W. Li et al., “Improving non-native mispronunciation detection and
                                                                         enriching diagnostic feedback with DNN-based speech attribute
raw waveform signals to the encoder network. As can be seen              modeling,” in Proc. ICASSP, pp. 6135–6139, 2016.
from Table IV, SincNet is slightly better than CNN by 1.47%,        [7] W. Hu, et al., “Improved mispronunciation detection with deep neural
in terms of the F-1 measure. Interestingly, SincNet achieves             network trained acoustic models and transfer learning based logistic
comparable MD results with the E2E MDD model that takes                  regression classifiers,” Speech Communication, vol. 67, pp.154–166,
input the FBANK features.                                                2015.
                                                                    [8] S. Sudhakara et al., “An improved goodness of pronunciation (GoP)
D. Performance of Mispronunciation Detection Diagnosis                   measure for pronunciation evaluation with DNN-HMM system
                                                                         considering HMM transition probabilities,” in Proc. Interspeech, pp.
In the third of set of experiments, we turn to evaluating the            954–958, 2019.
mispronunciation diagnosis performance of different E2E             [9] W.-K. Lo, S. Zhang and H. Meng, “Automatic derivation of
MDD models. The corresponding results are shown in Table                 phonological rules for mispronunciation detection in a Computer-
V, where the true positive rate (TP) and true negative rate (TN)         Assisted pronunciation training system,” in Proc. Interspeech, pp. 2010.
are two important criteria for evaluating the performance of        [10] K. Kyriakopoulos, K. Knill and M. Gales, “Automatic detection of
CAPT systems. From this Table V, we can observe that our                 accent and lexical pronunciation errors in spontaneous non-native
                                                                         English speech,” in Proc. Interspeech, pp. 3052–3056, 2020.
CTC-ATT model can boots CNN-RNN-CTC model
                                                                    [11] W.-K. Leung, X. Liu, and H. Meng. "CNN-RNN-CTC based end-to-
proposed from Leung et al., by consulting the diagnosis                  end mispronunciation detection and diagnosis," in Proc. ICASSP, pp.
results produced by the output of the attention-based decoder.           8132–8136, 2019.
Second, when the CTC-ATT model takes input raw                      [12] B.-C. Yan et al., “An End-to-End Mispronunciation Detection System
waveform signals with either CNN or SincNet (i.e., the                   for L2 English Speech Leveraging Novel Anti-Phone Modeling,” in
streamlined E2E model), it can slightly perform better than              Proc. Interspeech, pp. 3032–3036, 2020.
the CTC-ATT model takes input FBANK features (i.e., the             [13] Y. Feng, et al., “SED-MDD: Towards Sentence Dependent End-To-
                                                                         End Mispronunciation Detection and Diagnosis,” in Proc. ICASSP, pp.
cascaded E2E model) in terms of true positive rate, whereas              3492–3496, 2020.
the situation is reversed when consider the true negative rate.     [14] J. Tepperman, S. Narayanan, “Using articulatory representations to
It implies that the streamlined E2E model can learns to be               detect segmental errors in nonnative pronunciation,” IEEE trans. on
more discriminative on correct pronunciation detection.                  audio, speech, and language processing, vol. 16, 8–22, 2007.
Finally, we turn to investigating the mispronunciation              [15] L. Yang, et al., “Pronunciation Erroneous Tendency Detection with
                                                                         Language Adversarial Represent Learning,” in Proc. Interspeech,
diagnosis rate (DAR). As can be seen, using CNN to extract               3042–3046, 2020.
acoustic features from raw waveform signals can achieve the         [16] R. Duan, et al., “Cross-Lingual Transfer Learning of Non-Native
best performance. Besides that, the DAR of using SincNet to              Acoustic Modeling for Pronunciation Error Detection and Diagnosis,”
extract acoustic features from raw waveform signals also                 IEEE/ACM Trans. on Audio, Speech, and Language Processing, vol.
better than the baseline E2E model that takes input FBANK                28 pp. 391–401, 2019.
features.                                                           [17] D. Palaz, et al., “Estimating phoneme class conditional probabilities
                                                                         from raw speech signal using convolutional neural networks,” in Proc,
                                                                         Interspeech, pp. 1766–1770, 2013.
                      V. CONCLUSION
                                                                    [18] M. Ravanelli, and Y. Bengio, “Interpretable convolutional filters with
In this paper, we have designed and developed a fully end-to-            sincnet,” arXiv preprint arXiv:1811.09725, 2018.
end (E2E) neural model architecture that streamlines the            [19] S. Watanabe, et al., “Hybrid CTC/attention architecture for endto-end
MDD process by taking input waveform signals uttered by                  speech recognition,” IEEE Journal of Selected Topics in Signal
                                                                         Processing, vol. 11, no. 8, pp. 1240–1253, 2017.
L2 learners directly. Promising results have been obtained
                                                                    [20] A. Graves, et al., “Connectionist temporal classification: labelling
through a series of empirical experiments conducted on the               unsegmented sequence data with recurrent neural networks,” in Proc.
L2-ARCTIC benchmark dataset. As to future work, we will                  ICML, vol. 148, 2006.
try to investigate contrastive predictive coding (CPC) model        [21] G. Zhao et al., “L2-ARCTIC: A Non-native English Speech Corpus,”
and expect to solve the data sparsity problem through                    in Proc. Interspeech, pp. 2783–2787, 2018.
leveraging datasets of native speakers. Furthermore, we also        [22] J. S. Garofolo et al., “Darpa timit acoustic-phonetic continous speech
intended to study the suprasegmental-level phenomena for                 corpus cd-rom. nist speech disc 1-1.1,” STIN, vol. 93, 1993.
CAPT, such as intonation, accent pitch and rhythm.
