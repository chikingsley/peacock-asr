import socket
import sys
import os
from melo.api import TTS as MeloTTS
import io
import wave
import struct
import torch
import torchaudio
import re
import melo.utils

class TTS(MeloTTS):
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

def checkArg():
    if len(sys.argv) != 2:
        print("ERROR. Wrong number of arguments passed. System will exit. Next time please supply 1 argument!")
        sys.exit()
    else:
        print("1 Argument exists. We can proceed further")

def checkPort():
    if int(sys.argv[1]) <= 5000:
        print("Port number invalid. Port number should be greater than 5000. Next time enter valid port.")
        sys.exit()
    else:
        print("Port number accepted!")

def text_to_speech(text, tts_model, speaker_id=0):
    audio, sr = tts_model.synthesize(text, speaker_id=speaker_id)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor, sr, format="wav", bits_per_sample=16)
    buffer.seek(0)
    return buffer.read(), sr  # Return the audio data and sample rate

def send_audio(audio_data, client_address):
    chunk_size = 1024
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        s.sendto(chunk, client_address)
    s.sendto(b"END", client_address)

host = ""
checkArg()
try:
    port = int(sys.argv[1])
except ValueError:
    print("Error. Exiting. Please enter a valid port number.")
    sys.exit()
checkPort()

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Server socket initialized")
    s.bind((host, port))
    print("Successful binding. Waiting for Client now.")
except socket.error:
    print("Failed to create socket")
    sys.exit()

# Initialize MeloTTS
tts_model = TTS('EN')
speaker_id = 0  # You may need to adjust this based on your MeloTTS model

while True:
    try:
        data, clientAddr = s.recvfrom(4096)
    except ConnectionResetError:
        print("Error. Port numbers not matching. Exiting. Next time enter same port numbers.")
        sys.exit()
    
    text = data.decode('utf8')
    print(f"Received text: {text}")
    
    if text.lower() == "exit":
        print("Exiting server...")
        s.close()
        sys.exit()
    
    audio_data, sample_rate = text_to_speech(text, tts_model, speaker_id)
    
    # Send sample rate first
    s.sendto(struct.pack('I', sample_rate), clientAddr)
    
    send_audio(audio_data, clientAddr)
    print("Audio sent to client")

print("Program will end now.")
s.close()
tts_model.close()  # Clean up MeloTTS resources