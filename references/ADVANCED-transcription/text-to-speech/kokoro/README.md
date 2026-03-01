# Kokoro

## Simple Inference
Set up a venv 
```python
pip install uv
uv venv --python 3.10
uv pip install ipykernel
uv run python -m ipykernel install --user --name=.venv --display-name="Kokoro (.venv)" 
```
and install espeak-ng if on mac with `brew install espeak-ng`, go to the [espeak-ng website](https://github.com/espeak-ng/espeak-ng/releases) for windows and for Linux you can install from within the notebook.

Now Open `kokoro-simple-inference.ipynb` in jupyter and select the kernel above.

Or, skip the above and just run by downloading the files:
```bash
!mkdir -p models
!wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -O ./models/kokoro-v1.0.onnx
!wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -O ./models/voices-v1.0.bin
```
And then run:
```bash
uv run --with kokoro-onnx --with soundfile quick-kokoro.py
```

## Set up a Server
You can start a one-click template using [this Runpod affiliate link](https://console.runpod.io/deploy?template=grwfixzu60&ref=jmfkcdio). An A40 is fine.

You can then hit that server by updating `hit-kokoro.py` with the address of the pod.

## Benchmark Throughput
Test server throughput with `uv run benchmark-kokoro.py --requests 30 --concurrency 10` or use `--test-levels` to find optimal concurrency.