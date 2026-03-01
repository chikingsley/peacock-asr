pyannote.audio speaker diarization toolkit

pyannote.audio is an open-source toolkit written in Python for speaker diarization. Based on PyTorch machine learning framework, it comes with state-of-the-art pretrained models and pipelines, that can be further finetuned to your own data for even better performance.



TL;DR

Install pyannote.audio with pip install pyannote.audio
Accept pyannote/segmentation-3.0 user conditions
Accept pyannote/speaker-diarization-3.1 user conditions
Create access token at hf.co/settings/tokens.
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("audio.wav")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...
Highlights

🤗 pretrained pipelines (and models) on 🤗 model hub
🤯 state-of-the-art performance (see Benchmark)
🐍 Python-first API
⚡ multi-GPU training with pytorch-lightning
Documentation

Changelog
Frequently asked questions
Models
Available tasks explained
Applying a pretrained model
Training, fine-tuning, and transfer learning
Pipelines
Available pipelines explained
Applying a pretrained pipeline
Adapting a pretrained pipeline to your own data
Training a pipeline
Contributing
Adding a new model
Adding a new task
Adding a new pipeline
Sharing pretrained models and pipelines
Blog
2022-12-02 > "How I reached 1st place at Ego4D 2022, 1st place at Albayzin 2022, and 6th place at VoxSRC 2022 speaker diarization challenges"
2022-10-23 > "One speaker segmentation model to rule them all"
2021-08-05 > "Streaming voice activity detection with pyannote.audio"
Videos
Introduction to speaker diarization / JSALT 2023 summer school / 90 min
Speaker segmentation model / Interspeech 2021 / 3 min
First release of pyannote.audio / ICASSP 2020 / 8 min
Community contributions (not maintained by the core team)
2024-04-05 > Offline speaker diarization (speaker-diarization-3.1) by Simon Ottenhaus
Benchmark

Out of the box, pyannote.audio speaker diarization pipeline v3.1 is expected to be much better (and faster) than v2.x. Those numbers are diarization error rates (in %):

Benchmark	v2.1	v3.1	pyannoteAI
AISHELL-4	14.1	12.2	11.9
AliMeeting (channel 1)	27.4	24.4	22.5
AMI (IHM)	18.9	18.8	16.6
AMI (SDM)	27.1	22.4	20.9
AVA-AVD	66.3	50.0	39.8
CALLHOME (part 2)	31.6	28.4	22.2
DIHARD 3 (full)	26.9	21.7	17.2
Earnings21	17.0	9.4	9.0
Ego4D (dev.)	61.5	51.2	43.8
MSDWild	32.8	25.3	19.8
RAMC	22.5	22.2	18.4
REPERE (phase2)	8.2	7.8	7.6
VoxConverse (v0.3)	11.2	11.3	9.4
Diarization error rate (in %)

Citations

If you use pyannote.audio please use the following citations:

@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
Development

The commands below will setup pre-commit hooks and packages needed for developing the pyannote.audio library.

pip install -e .[dev,testing]
pre-commit install
Test

pytest