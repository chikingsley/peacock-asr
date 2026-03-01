Title: GitHub - canopyai/Orpheus-TTS: TTS Towards Human-Sounding Speech

URL Source: http://github.com/canopyai/Orpheus-TTS

Markdown Content:
Orpheus TTS
-----------

[](http://github.com/canopyai/Orpheus-TTS#orpheus-tts)

Overview
--------

[](http://github.com/canopyai/Orpheus-TTS#overview)

Orpheus TTS is an open-source text-to-speech system built on the Llama-3b backbone. Orpheus demonstrates the emergent capabilities of using LLMs for speech synthesis. We offer comparisons of the models below to leading closed models like Eleven Labs and PlayHT in our blog post.

[Check out our blog post](https://canopylabs.ai/model-releases)

demo.mp4

Abilities
---------

[](http://github.com/canopyai/Orpheus-TTS#abilities)

*   **Human-Like Speech**: Natural intonation, emotion, and rhythm that is superior to SOTA closed source models
*   **Zero-Shot Voice Cloning**: Clone voices without prior fine-tuning
*   **Guided Emotion and Intonation**: Control speech and emotion characteristics with simple tags
*   **Low Latency**: ~200ms streaming latency for realtime applications, reducible to ~100ms with input streaming

Models
------

[](http://github.com/canopyai/Orpheus-TTS#models)

We provide three models in this release, and additionally we offer the data processing scripts and sample datasets to make it very straightforward to create your own finetune.

1.  [**Finetuned Prod**](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod) – A finetuned model for everyday TTS applications
    
2.  [**Pretrained**](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained) – Our base model trained on 100k+ hours of English speech data
    

### Inference

[](http://github.com/canopyai/Orpheus-TTS#inference)

#### Simple setup on colab

[](http://github.com/canopyai/Orpheus-TTS#simple-setup-on-colab)

1.  [Colab For Tuned Model](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing) (not streaming, see below for realtime streaming) – A finetuned model for everyday TTS applications.
2.  [Colab For Pretrained Model](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS?usp=sharing) – This notebook is set up for conditioned generation but can be extended to a range of tasks.

#### Streaming Inference Example

[](http://github.com/canopyai/Orpheus-TTS#streaming-inference-example)

1.  Clone this repo
    
    git clone https://github.com/canopyai/Orpheus-TTS.git
    
2.  Navigate and install packages
    
    cd Orpheus-TTS && pip install orpheus-speech # uses vllm under the hood for fast inference
    
    vllm pushed a slightly buggy version on March 18th so some bugs are being resolved by reverting to `pip install vllm==0.7.3` after `pip install orpheus-speech`
3.  Run the example below:
    
    from orpheus\_tts import OrpheusModel
    import wave
    import time
    
    model \= OrpheusModel(model\_name \="canopylabs/orpheus-tts-0.1-finetune-prod")
    prompt \= '''Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.'''
    
    start\_time \= time.monotonic()
    syn\_tokens \= model.generate\_speech(
       prompt\=prompt,
       voice\="tara",
       )
    
    with wave.open("output.wav", "wb") as wf:
       wf.setnchannels(1)
       wf.setsampwidth(2)
       wf.setframerate(24000)
    
       total\_frames \= 0
       chunk\_counter \= 0
       for audio\_chunk in syn\_tokens: \# output streaming
          chunk\_counter += 1
          frame\_count \= len(audio\_chunk) // (wf.getsampwidth() \* wf.getnchannels())
          total\_frames += frame\_count
          wf.writeframes(audio\_chunk)
       duration \= total\_frames / wf.getframerate()
    
       end\_time \= time.monotonic()
       print(f"It took {end\_time \- start\_time} seconds to generate {duration:.2f} seconds of audio")
    

#### Prompting

[](http://github.com/canopyai/Orpheus-TTS#prompting)

1.  The `finetune-prod` models: for the primary model, your text prompt is formatted as `{name}: I went to the ...`. The options for name in order of conversational realism (subjective benchmarks) are "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe". Our python package does this formatting for you, and the notebook also prepends the appropriate string. You can additionally add the following emotive tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`.
    
2.  The pretrained model: you can either generate speech just conditioned on text, or generate speech conditioned on one or more existing text-speech pairs in the prompt. Since this model hasn't been explicitly trained on the zero-shot voice cloning objective, the more text-speech pairs you pass in the prompt, the more reliably it will generate in the correct voice.
    

Additionally, use regular LLM generation args like `temperature`, `top_p`, etc. as you expect for a regular LLM. `repetition_penalty>=1.1`is required for stable generations. Increasing `repetition_penalty` and `temperature` makes the model speak faster.

Finetune Model
--------------

[](http://github.com/canopyai/Orpheus-TTS#finetune-model)

Here is an overview of how to finetune your model on any text and speech. This is a very simple process analogous to tuning an LLM using Trainer and Transformers.

You should start to see high quality results after ~50 examples but for best results, aim for 300 examples/speaker.

1.  Your dataset should be a huggingface dataset in [this format](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
2.  We prepare the data using this [this notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing). This pushes an intermediate dataset to your Hugging Face account which you can can feed to the training script in finetune/train.py. Preprocessing should take less than 1 minute/thousand rows.
3.  Modify the `finetune/config.yaml` file to include your dataset and training properties, and run the training script. You can additionally run any kind of huggingface compatible process like Lora to tune the model.
    
     pip install transformers datasets wandb trl flash\_attn torch
     huggingface-cli login <enter your HF token\>
     wandb login <wandb token\>
     accelerate launch train.py
    

Also Check out
--------------

[](http://github.com/canopyai/Orpheus-TTS#also-check-out)

While we can't verify these implementations are completely accurate/bug free, they have been recommended on a couple of forums, so we include them here:

1.  [A lightweight client for running Orpheus TTS locally using LM Studio API](https://github.com/isaiahbjork/orpheus-tts-local)
2.  [Gradio WebUI that runs smoothly on WSL and CUDA](https://github.com/Saganaki22/OrpheusTTS-WebUI)

Checklist
---------

[](http://github.com/canopyai/Orpheus-TTS#checklist)

*   Release 3b pretrained model and finetuned models
*   Release pretrained and finetuned models in sizes: 1b, 400m, 150m parameters
*   Fix glitch in realtime streaming package that occasionally skips frames.
*   Fix voice cloning Colab notebook implementation