Title: collinbarnwell/pyannote-speaker-diarization-31 · Hugging Face

URL Source: http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31

Markdown Content:
Using this open-source pipeline in production?  
Make the most of it thanks to our [consulting services](https://herve.niderb.fr/consulting.html).

[](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#%F0%9F%8E%B9-speaker-diarization-31)🎹 Speaker diarization 3.1
--------------------------------------------------------------------------------------------------------------------------------

This pipeline is the same as [`pyannote/speaker-diarization-3.0`](https://hf.co/pyannote/speaker-diarization-3.1) except it removes the [problematic](https://github.com/pyannote/pyannote-audio/issues/1537) use of `onnxruntime`.  
Both speaker segmentation and embedding now run in pure PyTorch. This should ease deployment and possibly speed up inference.  
It requires pyannote.audio version 3.1 or higher.

It ingests mono audio sampled at 16kHz and outputs speaker diarization as an [`Annotation`](http://pyannote.github.io/pyannote-core/structure.html#annotation) instance:

*   stereo or multi-channel audio files are automatically downmixed to mono by averaging the channels.
*   audio files sampled at a different rate are resampled to 16kHz automatically upon loading.

[](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#requirements)Requirements
-------------------------------------------------------------------------------------------------

1.  Install [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) `3.1` with `pip install pyannote.audio`
2.  Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
3.  Accept [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote-speaker-diarization-3.1) user conditions
4.  Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

[](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#usage)Usage
-----------------------------------------------------------------------------------

```
# instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# run the pipeline on an audio file
diarization = pipeline("audio.wav")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
```

### [](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#processing-on-gpu)Processing on GPU

`pyannote.audio` pipelines run on CPU by default. You can send them to GPU with the following lines:

```
import torch
pipeline.to(torch.device("cuda"))
```

### [](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#processing-from-memory)Processing from memory

Pre-loading audio files in memory may result in faster processing:

```
waveform, sample_rate = torchaudio.load("audio.wav")
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```

### [](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#monitoring-progress)Monitoring progress

Hooks are available to monitor the progress of the pipeline:

```
from pyannote.audio.pipelines.utils.hook import ProgressHook
with ProgressHook() as hook:
    diarization = pipeline("audio.wav", hook=hook)
```

### [](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#controlling-the-number-of-speakers)Controlling the number of speakers

In case the number of speakers is known in advance, one can use the `num_speakers` option:

```
diarization = pipeline("audio.wav", num_speakers=2)
```

One can also provide lower and/or upper bounds on the number of speakers using `min_speakers` and `max_speakers` options:

```
diarization = pipeline("audio.wav", min_speakers=2, max_speakers=5)
```

[](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#benchmark)Benchmark
-------------------------------------------------------------------------------------------

This pipeline has been benchmarked on a large collection of datasets.

Processing is fully automatic:

*   no manual voice activity detection (as is sometimes the case in the literature)
*   no manual number of speakers (though it is possible to provide it to the pipeline)
*   no fine-tuning of the internal models nor tuning of the pipeline hyper-parameters to each dataset

... with the least forgiving diarization error rate (DER) setup (named _"Full"_ in [this paper](https://doi.org/10.1016/j.csl.2021.101254)):

*   no forgiveness collar
*   evaluation of overlapped speech

[](http://huggingface.co/collinbarnwell/pyannote-speaker-diarization-31#citations)Citations
-------------------------------------------------------------------------------------------

```
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

```
@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```