import sys
import os
zipformer_path = os.path.abspath("./zipformer_transducer")
sys.path.insert(0, zipformer_path)

import torch
import torch.nn as nn

from zipformer_transducer.train import get_model
from icefall.utils import AttributeDict
from zipformer_transducer.beam_search import (
    greedy_search,
    greedy_search_batch,
    )
from lhotse.features.kaldi.extractors import Fbank
import sentencepiece as spm


small_params = AttributeDict(
    {
        # Fixed parameters
        "feature_dim": 80,
        "subsampling_factor": 4,
        "vocab_size": 127,

        # Zipformer encoder stack parameters
        "num_encoder_layers": "2,2,3,4,3,2",
        "downsampling_factor": "1,2,4,8,4,2",
        "feedforward_dim": "512,768,1024,1536,1024,768",
        "num_heads": "4,4,4,8,4,4",
        "encoder_dim": "192,256,384,512,384,256",
        "query_head_dim": "32",
        "value_head_dim": "12",
        "pos_head_dim": "4",
        "pos_dim": 48,
        "encoder_unmasked_dim": "192, 192, 256, 256, 256, 192",
        "cnn_module_kernel": "31,31,15,15,15,31",

        # Decoder and joiner
        "decoder_dim": 512,
        "joiner_dim": 512,
        "context_size": 2,

        # Attention decoder
        "attention_decoder_dim": 512,
        "attention_decoder_num_layers": 6,
        "attention_decoder_attention_dim": 512,
        "attention_decoder_num_heads": 8,
        "attention_decoder_feedforward_dim": 2048,

        # Training details
        "causal": False,
        "chunk_size": "16,32,64,-1",
        "left_context_frames": "64,128,256,-1",

        # Loss and decoding heads
        "use_transducer": True,
        "use_ctc": False,
        "use_attention_decoder": False,
        "use_cr_ctc": False,
        "use_unsup_cr_ctc": False,
    }
)


large_params = AttributeDict(
    {
        # Fixed parameters
        "feature_dim": 80,
        "subsampling_factor": 4,
        "vocab_size": 127,

        # Zipformer encoder stack parameters
        "num_encoder_layers": "4,3,4,5,4,4",
        "downsampling_factor": "1,2,4,8,4,2",
        "feedforward_dim": "768,768,1536,2048,1536,768",
        "num_heads": "6,6,6,8,6,6",
        "encoder_dim": "512,512,768,1024,768,512",
        "query_head_dim": "64",
        "value_head_dim": "48",
        "pos_head_dim": "4",
        "pos_dim": 48,
        "encoder_unmasked_dim": "192,192,256,320,256,192",
        "cnn_module_kernel": "31,31,15,15,15,31",

        # Decoder and joiner
        "decoder_dim": 1024,
        "joiner_dim": 1024,
        "context_size": 2,

        # Attention decoder
        "attention_decoder_dim": 512,
        "attention_decoder_num_layers": 6,
        "attention_decoder_attention_dim": 512,
        "attention_decoder_num_heads": 8,
        "attention_decoder_feedforward_dim": 2048,

        # Training details
        "causal": False,
        "chunk_size": "16,32,64,-1",
        "left_context_frames": "64,128,256,-1",

        # Loss and decoding heads
        "use_transducer": True,
        "use_ctc": False,
        "use_attention_decoder": False,
        "use_cr_ctc": False,
        "use_unsup_cr_ctc": False,
    }
)

class ZIPA_Transducer(nn.Module):

    def __init__(self, params):
        super().__init__()
        
        self.bpe_model = spm.SentencePieceProcessor()
        self.bpe_model.load(params.bpe_model)
        params.blank_id = self.bpe_model.piece_to_id("<blk>")
        params.sos_id = params.eos_id = self.bpe_model.piece_to_id("<sos/eos>")
        params.vocab_size = self.bpe_model.get_piece_size()

        self.model = get_model(params)
        self.model.load_state_dict(torch.load(params.model_path),strict=True)
        self.model.to(params.device)
        self.model.eval()

        self.fbank = Fbank()
        self.device = params.device

    def predict(self, feature, feature_lens):

        encoder_out, encoder_out_lens = self.model.forward_encoder(feature.to(self.device), feature_lens.to(self.device))

        hyp_tokens = greedy_search_batch(
            model=self.model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        hyps = []
        for hyp in self.bpe_model.decode(hyp_tokens):
            hyps.append(hyp.split())
        return hyps

    def get_fbank(self, audio):
        features = self.fbank.extract_batch(
                        audio, sampling_rate=16000
                    )
        feature_lens = torch.tensor([len(feature) for feature in features])
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        return features, feature_lens


    def inference(self, audio):

        features, feature_lens = self.get_fbank(audio)
        hyps = self.predict(features, feature_lens)
        
        return hyps


def initialize_model(model_path, bpe_model):
    
    if "small" in model_path:
        params =  small_params
    elif "large" in model_path:
        params = large_params
    else:
        raise ValueError("model_name must contain 'small' or 'large'")
        
    params.bpe_model = bpe_model
    params.model_path = model_path
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ZIPA_Transducer(params)

    
if __name__ == "__main__":

    import torchaudio

    
    model_path = "zipformer_weights/zipa_large_noncausal_500000_avg10.pth"
    bpe_model_path = "ipa_simplified/unigram_127.model"

    model = initialize_model(model_path, bpe_model_path)

    # Generate a dummy audio batch (1 sample of 2 seconds of silence)
    sample_rate = 16000
    dummy_audio = [torch.zeros(int(sample_rate * 2)),torch.zeros(int(sample_rate * 2)),torch.zeros(int(sample_rate * 2))]  # 2-second silent audio

    # Run inference
    output = model.inference(dummy_audio)
    print("Predicted transcript:", output)