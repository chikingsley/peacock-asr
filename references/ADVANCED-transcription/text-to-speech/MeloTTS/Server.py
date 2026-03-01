from melo.api import TTS as BaseTTS
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from base64 import b64encode, b64decode
from contextlib import asynccontextmanager
import torch
import torchaudio.transforms as T
from typing import Optional
import time
from utils import write_bytesIO
import re
import melo.utils

class TTS(BaseTTS):
    def synthesize(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        for t in texts:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = melo.utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=1. / speed,
                )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
            audio_list.append(audio)
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)
        torch.cuda.empty_cache()
        return (audio, self.hps.data.sampling_rate)

    def close(self):
        torch.cuda.empty_cache()

TTS_Server = None
speaker_ids = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global TTS_Server
    global speaker_ids
    # Load the ML model
    TTS_Server = TTS('EN')
    speaker_ids = TTS_Server.hps.data.spk2id
    yield
    # Clean up the ML models and release the resources
    TTS_Server.close()

class TTSResponse(BaseModel):
    voice_id: str
    text: str
    sr: int
    sdp_ratio: Optional[float] = 0.2
    noise_scale: Optional[float] = 0.6
    noise_scale_w: Optional[float] = 0.8
    speed: Optional[float] = 1.0

app = FastAPI(lifespan=lifespan)

@app.post("/connection")
def tts_process(response: TTSResponse):
    __t = time.time()
    audio, sr = TTS_Server.synthesize(response.text, speaker_id=speaker_ids[response.voice_id], sdp_ratio=response.sdp_ratio, noise_scale=response.noise_scale, noise_scale_w=response.noise_scale_w, speed=response.speed)
    audio = torch.from_numpy(audio)
    resampler = T.Resample(sr, response.sr, dtype=audio.dtype)
    audio = resampler(audio)
    audio = audio.detach().numpy()
    files = write_bytesIO(response.sr, audio)
    return {'audio': b64encode(files.read()).decode(), 'sr': response.sr, "time": time.time() - __t}

if __name__ == "__main__":
    uvicorn.run("Server:app", host='0.0.0.0', port=8000, reload=True)