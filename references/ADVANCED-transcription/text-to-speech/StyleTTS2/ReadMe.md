# StyleTTS2

Attribution: the scripts in this repo build upon the `.py` scripts in the [GitHub StyleTTS2 Repository](https://github.com/yl4579/StyleTTS2/).

## Overview

StyleTTS2 is one of the state-of-the-art models for text-to-speech synthesis, leading among all the other open source TTS models in the TTS Arena by Hugging Face. This guide will walk you through the process of fine-tuning StyleTTS2 to potentially achieve results closer to Eleven Labs quality. 

What can you use this for?
- Voice cloning or adaptation
- Creating custom text-to-speech models
- Potentially achieving quality comparable to commercial TTS solutions
- Adapting an accent for the model

*Most of the customization can be achieved with just 30 minutes of audio.*

In this `text-to-speech` folder:
- `Trelis_StyleTTS2_Finetune_Demo.ipynb`: Ronan's version of the StyleTTS2 colab fine-tuning notebook. Inference is working. Testing of fine-tuning is next.
- `dataset_curation.ipynb`: Notebook to run in Colab for dataset generation.
- `Trelis_train_finetune.py`: script to be used for full fine-tuning, in conjunction with running `Trelis_StyleTTS2_Finetune_Demo.ipynb`
- `Trelis_train_lora_finetune.py`: script to be used for LoRA fine-tuning, in conjunction with running `Trelis_StyleTTS2_Finetune_Demo.ipynb`
- `naive-TTS`: contains scripts to make datasets for simple TTS model creation using either a GAN or Diffuser approach. This has not been fully tested, and the models likely require 100 hours of data to achieve reasonable quality.
- `archive`: deprecated fine-tuning and inference scripts.

---

## Data Preparation

The first step in the fine-tuning process is to curate your dataset. We use the `dataset_curation.ipynb` notebook for this purpose.

You can run this script locally or on Runpod [one-click template here](https://runpod.io/gsc?template=ifyqsvjlzj&ref=jmfkcdio), but using Colab is highly recommended if you are downloading YouTube transcripts/audio (because YouTube sometimes blocks downloads from elsewhere). Be aware also of licensing restrictions on content you are working with. 

### Using dataset_curation.ipynb

1. Open the `dataset_curation.ipynb` file in Google Colab.
2. Follow the instructions in the notebook to generate a dataset based on either:
   - A YouTube link
   - Your own audio files
3. The notebook will guide you through the process of:
   - Downloading audio (if using YouTube)
   - Transcribing the audio
   - Segmenting the audio into suitable chunks
   - Generating the necessary files for fine-tuning

### Output Files

After running the notebook, you should have the following files:

- Segmented WAV audio files
- `train_list.txt`: List of training data
- `val_list.txt`: List of validation data
- A dataset uploaded to your Huggingface account

---

## Fine-Tuning Process

### How to Finetune

To finetune the StyleTTS2 model on your own dataset, follow these steps:

1. You can run this notebook locally if you have a GPU or on a service like Google Colab or Runpod [one-click template here](https://runpod.io/gsc?template=ifyqsvjlzj&ref=jmfkcdio). A minimum of ~48 GB VRAM is recommended, which means using an RTX A6000 or A100.

1. Start Jupyter notebook.

1. Upload and step through the `Trelis_StyleTTS2_Finetune_Demo.ipynb` notebook.

1. Upload the `Trelis_train_finetune.py` or the `Trelis_train_lora_finetune.py` script into the StyleTTS2 directory.

1. Adjust batch size and other parameters based on your available computational resources.

> [!TIP]
> Always monitor the training and validation losses to ensure the model is improving and not overfitting.

---

## StyleTTS 2 Overview

### Model architecture:
- The decoder, `G`, produces an output wave `x_hat` by taking in outputs of:
   - T (Acoustic Text Encoder): This module encodes the input phonemes into phoneme representations. It processes the text input and generates embeddings that are used for further processing in the TTS pipeline.
   - B (Prosodic Text Encoder): This is a phoneme-level BERT (PL-BERT) model that provides a richer, more prosodically aware encoding of the input text. It helps capture the nuances of prosody and is used to enhance the naturalness of the synthesized speech.
   - S (Duration Predictor): This module predicts the duration of each phoneme in the input text. It generates the alignment between the text and the speech frames, which is crucial for timing the phoneme pronunciations correctly. It takes in the output of B. It produces durations for phenomes that are applied to the output of the accoustic text encoder T in order to feed aligned text encodings to the decoder `G`.
   - P (Prosody Predictor): This predictor forecasts the pitch and energy curves for the speech. These prosodic features are essential for making the speech sound natural and expressive. It also takes in the output of B.
   - V (Style Diffusion Denoiser): This is the style diffusion model that samples the style vector using diffusion processes. It refines the noisy style vector to produce a clean style representation that captures the characteristics of the desired speech style. It takes in the output of B and generates prosodic style `s_p` and accoustic style `s_a` vectors that serve as inputs to the duration predictor `S`, the prosody predictor (for pitch and energy) `P`, and the decoder `G` respectively.

### Losses tracked in the fine-tuning script:
1. Mel Loss (wandb: train/mel_loss)
- Paper: Lmel
- Description: Mel-spectrogram reconstruction loss.

2. Generator Loss (wandb: train/gen_loss)
- Paper: Combination of generator losses including Ladv(G; D), Lrel(G; D), and Lfm
- Description: Overall generator loss including adversarial and feature-matching losses.

3. Discriminator Loss (wandb: train/d_loss)
- Paper: Combination of discriminator losses including Ladv(D; G) and Lrel(D; G)
- Description: Overall discriminator loss.

4. Cross-Entropy Loss (wandb: train/ce_loss)
- Paper: Part of duration prediction, related to cross-entropy loss (Lce)
- Description: Cross-entropy loss used for duration prediction.

5. Duration Loss (wandb: train/dur_loss)
- Paper: Ldur
- Description: L1 loss for duration prediction.

6. Norm Loss (wandb: train/norm_loss)
- Paper: Ln
- Description: Energy reconstruction loss.

7. F0 Loss (wandb: train/F0_loss)
- Paper: Lf0
- Description: F0 (pitch) reconstruction loss.

8. Style Loss (wandb: train/sty_loss)
- Paper: Not explicitly named, related to style reconstruction in joint training objectives.
- Description: Loss related to ensuring the style encoding.

9. Diffusion Loss (wandb: train/diff_loss)
- Paper: Related to style diffusion in acoustic module pre-training, included in joint training objectives.
- Description: Loss related to style diffusion.

10. SLM Loss (wandb: train/slm_loss)
- Paper: Lslm(G; D) and Lslm(D; G)
- Description: Overall SLM adversarial loss, combining both generator and discriminator losses.

11. SLM Discriminator Loss (wandb: train/d_loss_slm)
- Paper: Lslm(D; G)
- Description: SLM discriminator loss during training.

12. SLM Generator Loss (wandb: train/gen_loss_slm)
- Paper: Lslm(G; D)
- Description: SLM generator loss during training.

13. Sequence-to-Sequence Loss (wandb: not present)
- Paper: Ls2s
- Description: Sequence-to-sequence ASR loss function.

14. Monotonic Loss (wandb: not present)
- Paper: Lmono
- Description: Loss to ensure soft attention approximates its non-differentiable monotonic version.

> [!NOTE]
> - For more detailed information on specific components or techniques, please refer to the StyleTTS2 documentation or research papers.
> - The fine-tuning process can be resource-intensive. Ensure you have adequate GPU resources available, ideally 48 GB+ of VRAM.

## Troubleshooting

If you encounter issues during the fine-tuning process, consider the following:
> [!IMPORTANT]
> - Ensure all dependencies are correctly installed
> - Check that your dataset is properly formatted and all paths in the configuration file are correct
> - Monitor GPU memory usage and adjust batch size if necessary. Note that the minimum batch size is 2.

## Issues
1. Add support for running fine-tuning on multiple GPUs. Currently, training does not progress when running with 4x A6000s.

## Todo
Rohan:
1. - [ ] Fix up any areas of `Trelis_StyleTTS2_Finetune_Demo` where I have put "Rohan" in a comment. Partially Done
1. - [ ] Review how Ronan has added code to speed up the final audio. Could this have been done in another way? (Don't spend too much time on this).

## Changelog
17Jul2024 (Rohan)
- Tidied Up the README.
- Updated the descriptions of the losses.

17Jul2024 (Ronan)
- Allow pushing of the fine-tuned model to hub.
- Fix fine-tuning notebook to allow for adversarial training.
- Moved Todo down towards changelog.
- Moved deprecated notebooks and scripts to the `archive` folder.

16Jul2024 (Rohan)
- Fixed the Merging and Inferencing Pipeline
- Test out the `Trelis_StyleTTS2_Finetune_Demo.ipynb`
- Ran both LoRA and full finetuning for 1 epoch.
- Figured out the inferencing for less number of epochs.
- Did Some more research on the training process
- Fixed SLM Logging

16Jul2024 (Ronan)
- Review [article](https://www.unite.ai/styletts-2-human-level-text-to-speech-with-large-speech-language-models/).
- Added naive-TTS with some simple GAN and Diffuser approaches to TTS.
- Finish testing LoRA. Quality is worse than full fine tuning.
- Complete fft training.
- Allow audio output speedup.
- Add markdown to the top of the file.
- clean up file naming conventions.
- add in params to allow the diffuser and SLM to be fine tuned.

15Jul2024 (Ronan)
- ensure the log dir is loaded for lora fine-tuning in `Trelis_train_lora_finetune.py`
- added the option to set `use_lora` to True or False.
- Fix dataset_curation to correctly count tokens. DONE. Turns out that neither naive phonemes nor the Bert tokenizer are used to tokenize. Rather, phenomes are passed through a script in `text_utils.py`. The dataset_curation.ipynb notebook has been fixed now to generate sequences just under 512 tokens. It is clear now why using about 75 naive tokens worked (when counting naive phenomes).

14Jul2024 (Rohan)
- Debugged the process of lora finetuning, merging and inferencing
- Debugged the dataset preparation process
- Figured out that there is a mistake while using Max Phoneme length as an indicator for maximum snippet length that can be fed to the model.


11Jul2024 (Ronan)
- What are alpha and beta doing? If you want to mimic the reference style more closely, then you use alpha and beta closer to zero. Basically, it's a weighted average between the model's original reference and the sample reference provided.
- Try to print out each model's architecture to see inputs and outputs. DONE! at the bottom. It's very complicated.
- Run a fine-tuning on Trelis data (re-generate data if needed). Done, running fine, need to test inference.
- RAM assessment: Turns out that with a batch size of 2, you need - on an A6000 - to have only 150 tokens of input, max.
- Understand context length (what are "frames"? is frame length just a constant times hop length?). Answer: Frames refer to the number of mel spectogram snapshots. The length of frames is set in `meldataset.py`, to 1200 ms.
- What sets the max number of frames (probably the models to which audio is inputted)? Roughly how long in time is this? With 1/24000 ms per frame, 400 frames is about 20000/24000 = ~1 second. 
- Where is the sampling rate defined for the model. Seems to be 24,000 Hz.
- dataset_curation.ipynb updates : added yt-dlp as a fallback if pytube doesn't work. Allow dataset creation to a max phoneme length (instead of audio duration). Set target range to 410-510 phonemes. This gave snippets of about 150 seconds in duration, which is about 150 x 24,000 / 1,200 = 3000 frames.

10Jul2024 (Ronan)
- Run through the default fine-tuning script (and see where the files get saved). Result = got fine-tuning working for one epoch. 
- See if I can add in weights and biases. Result: I got this running and we can now see all 12 losses. Training speed: Things are pretty fast for me.
- Check out Rohan's inference notebook. Made some small additions like `-qU` in places. Works well. Working on adding and testing LoRA merging. Done tentatively.
- What's the difference between fine-tuning and lora-fine-tuning scripts? Mostly just that lora adapters are added.
- Try out using different style inputs to influence the output. (this vector can also be tested to use to measure deepfakes, may or may not work). Result = changing the audio snippet does have some effect on the style vector, but it's hardly discernable.
- Printed out all loaded modules to understand what they do ( in `Trelis_StyleTTS2_Finetune_Demo.py`).
- Got inference of LJSpeech working using the StyleTTS2 fine-tuning notebook.
- Cleaned up top of README (still need to del AI generated portions)


09Jul2024 (Ronan)
- Integrated all scripts into the dataset_curation notebook.
- Add pre-buffering (on top of post), so padding is not needed.
- Dynamically created chunks of 20-30 seconds to provide a better and longer dataset.
- Improved comments and instructions.
- Fix typo in segmentation code causing incorrect gap calculations.

07Jul2024 (Rohan)
- Added Dataset Curation Script
- Added the finetuning notebook
- Added the finetuning scripts for LoRA and full finetuing

