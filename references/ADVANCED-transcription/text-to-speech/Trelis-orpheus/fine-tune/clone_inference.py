#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orpheus Voice Cloning Inference Script
For cloning voices using the Orpheus model with a reference audio file
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import soundfile as sf
import librosa
from snac import SNAC
import torchaudio
from huggingface_hub import snapshot_download
import subprocess
import platform

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run voice cloning inference with Orpheus model")
    parser.add_argument("--model_path", type=str, default="canopylabs/orpheus-3b-0.1-pretrained", 
                      help="Path to model directory or model name from HuggingFace")
    parser.add_argument("--device", type=str, default="mps", 
                      choices=["cpu", "cuda", "mps"], 
                      help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--reference_audio", type=str, default="../X.wav", 
                      help="Path to reference audio file (.wav)")
    parser.add_argument("--reference_text", type=str, 
                      default="This is how you count to three: one, two, three",
                      help="Transcript of the reference audio")
    parser.add_argument("--prompt", type=str, 
                      default="Great. Let's continue that: four, five, six, seven, eight, nine, ten.",
                      help="Text prompt to synthesize in the cloned voice")
    parser.add_argument("--speaker", type=str, default="", 
                      help="Speaker name to prepend to both reference and prompt (if not already included)")
    parser.add_argument("--output", type=str, default="cloned_output.wav", 
                      help="Output audio file path")
    parser.add_argument("--temperature", type=float, default=0.5, 
                      help="Temperature for sampling (higher = more diverse)")
    parser.add_argument("--top_p", type=float, default=0.9, 
                      help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=990, 
                      help="Maximum number of new tokens to generate")
    parser.add_argument("--no_play", action="store_true",
                      help="Disable automatic playback of audio files")
    return parser.parse_args()

def setup_model(model_path, device, hf_token=None):
    print(f"Loading model from {model_path}...")
    
    # Set up environment for HF transfer if using a HF model
    if not os.path.isdir(model_path):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        # Download model from HF Hub
        model_path = snapshot_download(
            repo_id=model_path,
            allow_patterns=[
                "config.json",
                "*.safetensors",
                "model.safetensors.index.json",
            ],
            ignore_patterns=[
                "optimizer.pt",
                "pytorch_model.bin",
                "training_args.bin",
                "scheduler.pt",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "tokenizer.*"
            ]
        )
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        local_files_only=os.path.isdir(model_path)
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=os.path.isdir(model_path)
    )
    
    # Load SNAC model for audio encoding/decoding
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to(device)
    
    return model, tokenizer, snac_model

def tokenise_audio(waveform, snac_model):
    # Get the device from the SNAC model
    device = next(snac_model.parameters()).device
    
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32, device=device)
    waveform = waveform.unsqueeze(0)

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item()+128266)
        all_codes.append(codes[1][0][2*i].item()+128266+4096)
        all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

    return all_codes

def format_text_with_speaker(text, speaker):
    """Add speaker prefix to text if not already present and speaker is provided."""
    if not speaker:
        return text
    
    if not text.startswith(f"{speaker}:"):
        return f"{speaker}: {text}"
    
    return text

def prepare_cloning_prompt(reference_audio_path, reference_text, prompt_text, tokenizer, snac_model, device, speaker=""):
    # Add speaker prefix to texts if provided
    reference_text_with_speaker = format_text_with_speaker(reference_text, speaker)
    prompt_text_with_speaker = format_text_with_speaker(prompt_text, speaker)
    
    # Resolve the reference audio path if it's relative
    if not os.path.isabs(reference_audio_path):
        # Check if the file exists as is
        if not os.path.exists(reference_audio_path):
            # Try relative to the parent directory
            parent_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), reference_audio_path)
            if os.path.exists(parent_dir_path):
                reference_audio_path = parent_dir_path
            else:
                # Try another common location
                alt_path = os.path.join(os.getcwd(), reference_audio_path)
                if os.path.exists(alt_path):
                    reference_audio_path = alt_path
                else:
                    raise FileNotFoundError(f"Could not find reference audio file at: {reference_audio_path}, {parent_dir_path}, or {alt_path}")
                
    print(f"Processing reference audio: {reference_audio_path}")
    print(f"Reference text: {reference_text_with_speaker}")
    print(f"Prompt text: {prompt_text_with_speaker}")
    
    # Load and process the reference audio
    try:
        audio_array, sample_rate = librosa.load(reference_audio_path, sr=24000)
        print(f"Loaded audio with length: {len(audio_array)} samples, sample rate: {sample_rate}Hz")
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")
    
    # Tokenize the audio into SNAC codes
    audio_tokens = tokenise_audio(audio_array, snac_model)
    
    # Special tokens for creating the prompt
    start_tokens = torch.tensor([[ 128259]], dtype=torch.int64).to(device)  # Start of human
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64).to(device)  # End of text, End of human, Start of AI, Start of speech
    final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64).to(device)  # End of speech, End of AI
    
    # Tokenize the reference text
    prompt_tokenized = tokenizer(reference_text_with_speaker, return_tensors="pt")
    reference_input_ids = prompt_tokenized["input_ids"].to(device)
    
    # Create the full prompt with reference audio tokens
    audio_tokens_tensor = torch.tensor([audio_tokens], dtype=torch.int64).to(device)
    zero_prompt_input_ids = torch.cat([
        start_tokens, 
        reference_input_ids, 
        end_tokens, 
        audio_tokens_tensor, 
        final_tokens
    ], dim=1)
    
    # Tokenize the target prompt text
    target_input_ids = tokenizer(prompt_text_with_speaker, return_tensors="pt").input_ids.to(device)
    
    # Create the final input with both reference and target
    final_input_ids = torch.cat([
        zero_prompt_input_ids, 
        start_tokens, 
        target_input_ids, 
        end_tokens
    ], dim=1)
    
    # Create attention mask
    attention_mask = torch.ones(final_input_ids.shape, dtype=torch.int64).to(device)
    
    return final_input_ids, attention_mask

def generate_speech(model, input_ids, attention_mask, args):
    print("Generating speech...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
        )
    return generated_ids

def process_output(generated_ids, snac_model, device):
    token_to_find = 128257  # Start-of-speech token
    token_to_remove = 128258  # End-of-speech token

    # Check if the token exists in the tensor
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids

    processed_rows = []
    for row in cropped_tensor:
        # Apply the mask to each row
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        # row is a 1D tensor with its own length
        row_length = row.size(0)
        new_length = (row_length // 7) * 7  # largest multiple of 7 that fits in this row
        trimmed_row = row[:new_length]
        trimmed_row = [t.item() - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    # Generate audio samples for each code list
    samples = []
    for code_list in code_lists:
        # Convert each code list to an audio sample
        audio = redistribute_codes(code_list, snac_model, device)
        samples.append(audio)
    
    return samples

def redistribute_codes(code_list, snac_model, device):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    for i in range((len(code_list)+1)//7):
        if 7*i < len(code_list):
            layer_1.append(code_list[7*i])
            
            if 7*i+1 < len(code_list) and 7*i+4 < len(code_list):
                layer_2.append(code_list[7*i+1]-4096)
                layer_2.append(code_list[7*i+4]-(4*4096))
            
            if 7*i+2 < len(code_list) and 7*i+3 < len(code_list) and 7*i+5 < len(code_list) and 7*i+6 < len(code_list):
                layer_3.append(code_list[7*i+2]-(2*4096))
                layer_3.append(code_list[7*i+3]-(3*4096))
                layer_3.append(code_list[7*i+5]-(5*4096))
                layer_3.append(code_list[7*i+6]-(6*4096))
    
    # Convert to tensors - ensure they're on the right device
    device_tensor = next(snac_model.parameters()).device
    if str(device_tensor) != str(device):
        print(f"Warning: Device mismatch - SNAC model is on {device_tensor} but requested device is {device}")
    
    codes = [
        torch.tensor(layer_1, dtype=torch.long).unsqueeze(0).to(device_tensor),
        torch.tensor(layer_2, dtype=torch.long).unsqueeze(0).to(device_tensor),
        torch.tensor(layer_3, dtype=torch.long).unsqueeze(0).to(device_tensor)
    ]
    
    # Decode audio
    audio_hat = snac_model.decode(codes)
    return audio_hat

def save_audio(audio, output_path, sr=24000):
    print(f"Saving audio to {output_path}...")
    # Convert to numpy if it's a tensor
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().squeeze().cpu().numpy()
    sf.write(output_path, audio, sr)

def play_audio(audio_file):
    """Play an audio file using the appropriate command for the OS."""
    system = platform.system()
    
    # Full path to the audio file
    if not os.path.isabs(audio_file):
        audio_file = os.path.abspath(audio_file)
    
    print(f"Playing audio: {audio_file}")
    
    try:
        # Check if running in SSH environment
        in_ssh = 'SSH_CONNECTION' in os.environ or 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ
        
        if in_ssh:
            print(f"SSH session detected. Skipping audio playback.")
            print(f"You can manually listen to the file at: {audio_file}")
            return
            
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_file])
        elif system == "Linux":
            # Try different Linux audio players in order of preference
            linux_players = [
                ["aplay", audio_file],
                ["paplay", audio_file],
                ["mplayer", audio_file],
                ["mpv", "--no-video", audio_file],
                ["ffplay", "-nodisp", "-autoexit", audio_file]
            ]
            
            success = False
            for player_cmd in linux_players:
                try:
                    subprocess.run(player_cmd, stderr=subprocess.DEVNULL, check=False)
                    success = True
                    break
                except FileNotFoundError:
                    continue
                
            if not success:
                print("Could not find a suitable audio player on Linux.")
                print(f"You can manually listen to the file at: {audio_file}")
        elif system == "Windows":
            subprocess.run(["start", "wmplayer", audio_file], shell=True)
        else:
            print(f"Unsupported platform: {system} - can't play audio automatically")
            print(f"You can manually listen to the file at: {audio_file}")
    except Exception as e:
        print(f"Error playing audio: {e}")
        print(f"You can manually listen to the file at: {audio_file}")

def main():
    args = parse_arguments()
    
    try:
        # Check CUDA availability for better error messages
        if args.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"
        
        print(f"Using device: {args.device}")
        
        # Handle speaker parameter
        if args.speaker:
            print(f"Using speaker: {args.speaker}")
        
        # Resolve the input file path early so we can use it for playback later
        reference_audio_path = args.reference_audio
        if not os.path.isabs(reference_audio_path):
            if not os.path.exists(reference_audio_path):
                parent_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), reference_audio_path)
                if os.path.exists(parent_dir_path):
                    reference_audio_path = parent_dir_path
                else:
                    alt_path = os.path.join(os.getcwd(), reference_audio_path)
                    if os.path.exists(alt_path):
                        reference_audio_path = alt_path
        
        # Setup the model
        model, tokenizer, snac_model = setup_model(args.model_path, args.device)
        
        # Verify models are on the correct device
        model_device = next(model.parameters()).device
        snac_device = next(snac_model.parameters()).device
        print(f"Model is on device: {model_device}")
        print(f"SNAC model is on device: {snac_device}")
        
        # Prepare the cloning prompt
        input_ids, attention_mask = prepare_cloning_prompt(
            reference_audio_path, 
            args.reference_text, 
            args.prompt, 
            tokenizer, 
            snac_model,
            args.device,
            args.speaker
        )
        
        # Generate speech
        generated_ids = generate_speech(model, input_ids, attention_mask, args)
        
        # Process output and create audio
        audio_samples = process_output(generated_ids, snac_model, args.device)
        
        # Save the audio
        if audio_samples:
            # Ensure the output directory exists
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            save_audio(audio_samples[0], args.output)
            output_path = os.path.abspath(args.output)
            print(f"Successfully generated cloned speech at {output_path}")
            
            # Play the audio files if not disabled
            if not args.no_play:
                print("\n--- Playing Reference Audio ---")
                play_audio(reference_audio_path)
                
                print("\n--- Playing Cloned Audio ---")
                play_audio(output_path)
        else:
            print("Failed to generate audio samples")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print("\nDetailed error information:")
        traceback.print_exc()
        print("\n💡 Tip: If this is a device-related error, try using --device cpu instead.")
        sys.exit(1)

if __name__ == "__main__":
    main() 