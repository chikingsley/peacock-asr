import soundfile as sf
from kokoro_onnx import Kokoro

kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
samples, sr = kokoro.create(
    "Hello from Kokoro, this is a python script!",
    voice="bm_george",
    speed=1.0,
    # lang="en_us",
)
sf.write("audio.wav", samples, sr)