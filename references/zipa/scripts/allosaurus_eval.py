import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
from lhotse import CutSet, load_manifest, load_manifest_lazy
import numpy as np
import panphon.distance
from tqdm import tqdm
from glob import glob
import sentencepiece as spm
#from jiwer import wer
import logging
import json
from allosaurus.app import read_recognizer
import soundfile as sf
import torchaudio
dst = panphon.distance.Distance()

    
def test_clean_cuts():
    logging.info("About to get test cuts")
    return CutSet.from_shar(
        {
            "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/test-clean/cuts.000000.jsonl.gz"],
            "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/test-clean/recording.000000.tar"]
        }
    )

    
def test_ucla_cuts():
    logging.info("About to get UCLA test cuts")
    data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/vox_angeles'
    cuts = sorted(glob(os.path.join(data_path,"**",'cuts*'),recursive=True))
    recordings =  sorted(glob(os.path.join(data_path,"**",'recording*'),recursive=True))
    
    return CutSet.from_shar(
        fields={
            "cuts": cuts,
            "recording": recordings
        }
    )


def test_other_cuts():
    logging.info("About to get test cuts")
    return CutSet.from_shar(
        {
            "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/test-other/cuts.000000.jsonl.gz"],
            "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/test-other/recording.000000.tar"]
        }
    )


def test_doreco_cuts():
    logging.info("About to get doreco test cuts")
    data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/doreco_shar'
    cuts = sorted(glob(os.path.join(data_path,"**",'cuts*'),recursive=True))
    recordings =  sorted(glob(os.path.join(data_path,"**",'recording*'),recursive=True))
    print(len(cuts))
    print(len(recordings))
    
    return CutSet.from_shar(
        fields={
            "cuts": cuts,
            "recording": recordings
        }
    )


def test_aishell_cuts():
    logging.info("About to get aishell test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/aishell_shar/test/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/aishell_shar/test/recording.000000.tar']
        }
    )


def test_mls_german_cuts():
    logging.info("About to get mls german test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_german_shar/test/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_german_shar/test/recording.000000.tar']
        }
    )


def test_mls_italian_cuts():
    logging.info("About to get mls italian test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_italian_shar/test/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_italian_shar/test/recording.000000.tar']
        }
    )


def test_mls_french_cuts():
    logging.info("About to get mls french test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_french_shar/test/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_french_shar/test/recording.000000.tar']
        }
    )


def test_mls_spanish_cuts():
    logging.info("About to get mls spanish test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_spanish_shar/test/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_spanish_shar/test/recording.000000.tar']
        }
    )


def test_mls_portuguese_cuts():
    logging.info("About to get mls portuguese test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_portuguese_shar/test/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_portuguese_shar/test/recording.000000.tar']
        }
    )


def test_mls_dutch_cuts():
    logging.info("About to get mls dutch test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_dutch_shar/test/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_dutch_shar/test/recording.000000.tar']
        }
    )



def test_buckeye_cuts():
    logging.info("About to get buckeye test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/multilingual/buckeye_shar/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/multilingual/buckeye_shar/recording.000000.tar']
        }
    )


def test_l2_arctic_cuts():
    logging.info("About to get l2_arctic test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/l2arctic_shar/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/l2arctic_shar/recording.000000.tar']
        }
    )



def test_l2_arctic_preceived_cuts():
    logging.info("About to get l2_arctic_perceived test cuts")
     
    return CutSet.from_shar(
        fields={
            "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/l2arctic_perceived_shar/cuts.000000.jsonl.gz'],
            "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/l2arctic_perceived_shar/recording.000000.tar']
        }
    )

class Audio:

    def __init__(self, samples=None, sample_rate=8000):

        # default parameters
        if samples is None:
            samples = []

        self.sample_rate = sample_rate
        self.samples = samples
        
        
# load model and processor
model = read_recognizer()
model.config.device_id = 0
model.am.to('cuda')

sp = spm.SentencePieceProcessor()
sp.load("ipa_simplified/unigram_127.model")


test_clean_cuts = test_clean_cuts()
test_other_cuts = test_other_cuts()
test_doreco_cuts = test_doreco_cuts()
test_ucla_cuts = test_ucla_cuts()
test_aishell_cuts = test_aishell_cuts()
test_mls_german_cuts = test_mls_german_cuts()
test_mls_italian_cuts = test_mls_italian_cuts()
test_mls_french_cuts = test_mls_french_cuts()
test_mls_spanish_cuts = test_mls_spanish_cuts()
test_mls_portuguese_cuts = test_mls_portuguese_cuts()
test_mls_dutch_cuts = test_mls_dutch_cuts()
test_buckeye_cuts = test_buckeye_cuts()
test_l2_arctic_cuts = test_l2_arctic_cuts()
test_l2_arctic_preceived_cuts = test_l2_arctic_preceived_cuts()


test_sets = ["test-clean", "test-other","doreco", "ucla", "aishell", "mls-french", "mls-german", "mls-italian", "mls-portuguese", "mls-spanish", "mls-dutch", "buckeye", "l2-arctic", "l2-arctic-perceived"]
test_cuts = [test_clean_cuts, test_other_cuts, test_doreco_cuts, test_ucla_cuts, test_aishell_cuts, test_mls_french_cuts, test_mls_german_cuts, test_mls_italian_cuts, test_mls_portuguese_cuts, test_mls_spanish_cuts, test_mls_dutch_cuts, test_buckeye_cuts, test_l2_arctic_cuts, test_l2_arctic_preceived_cuts]

for name, cutset in zip(test_sets, test_cuts):
    print(name)
    cutset.describe()
'''
output_path = os.path.join('/scratch/lingjzhu_root/lingjzhu1/lingjzhu/asr_exp','allosaurus')
if not os.path.exists(output_path):
    os.mkdir(output_path)
    
for name, cutset in zip(test_sets, test_cuts):
    # tokenize
    with open(os.path.join(output_path,f"{name}.jsonl"),'w') as out:
        pfers = []
        for cut in tqdm(cutset):
            audio = cut.load_audio()
            sf.write(f"{output_path}/sample.wav", np.array(audio).squeeze(), 16000)
            transcript = cut.supervisions[0].text.replace("g","ɡ")
            transcription = model.recognize(f"{output_path}/sample.wav")
            transcription = "".join(transcription).replace("͡",'').replace(" ",'')
            
            pfer = dst.feature_edit_distance(transcript,transcription)
            pfers.append(pfer)
            out.write(json.dumps({"ground truth": transcript, "predicted": transcription, "pfer":pfer},ensure_ascii=False)+"\n")
            
    with open(os.path.join(output_path,"results.jsonl"),'a') as out_result:
        out_result.write(json.dumps({'name':name, 'pfer':np.mean(pfers)})+"\n")
    print(f"PFER mean: {np.mean(pfers)}")
    

wers = []
for cut in tqdm(cutset):
    audio = cut.load_audio()
    transcript = cut.supervisions[0].text
    input_values = processor(audio, return_tensors="pt").input_values

    # retrieve logits
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits.cpu(), dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    transcription = "".join(transcription).replace(" ",'')
    print(f"ground truth: {transcript}")
    print(f"predicted: {transcription}")
    transcript = sp.encode(transcript, out_type=str)
    transcription = sp.encode(transcription, out_type=str)
    error = wer(' '.join(transcript[1:])," ".join(transcription[1:]))
    wers.append(error)
    
print(f"WER: {np.mean(wers)}")
'''