# ONNX Models for ZIPA

This repository contains optimized ONNX models for the ZIPA speech recognition system. The models are exported from the original PyTorch checkpoints and support FP32, FP16, and INT8 precision.

## Original Repository

[GitHub: lingjzhu/zipa](https://github.com/lingjzhu/zipa)

## Available Files

- `model.onnx`: FP32 ONNX model (or `encoder-*.onnx`, etc. for Transducer)
- `model.fp16.onnx`: FP16 Quantized model
- `model.int8.onnx`: INT8 Quantized model
- `tokens.txt`: Token vocabulary

## Usage

### Installation

```bash
pip install onnxruntime soundfile librosa lhotse
```

### Inference

Please refer to the [Inference Documentation](https://github.com/lingjzhu/zipa/tree/main/inference) in the official repository for detailed usage and scripts.

#### Quick Example (CTC)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.fp16.onnx")
# ... preprocessing ...
```
