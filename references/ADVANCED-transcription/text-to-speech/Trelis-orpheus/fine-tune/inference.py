#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orpheus Non-Streaming Inference Script
For testing fine-tuned Orpheus TTS models
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import soundfile as sf
from snac import SNAC

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Orpheus model")
    parser.add_argument("--model_path", type=str, default="./output", 
                      help="Path to fine-tuned model or model name from HuggingFace")
    parser.add_argument("--device", type=str, default="mps", 
                      choices=["cpu", "cuda", "mps"], 
                      help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--prompt", type=str, 
                      default="Hey there! This is a test of my fine-tuned text-to-speech model.",
                      help="Text prompt to synthesize")
    parser.add_argument("--speaker", type=str, default="", 
                      help="Speaker name to prepend to prompt (if not included in prompt)")
    parser.add_argument("--output", type=str, default="output.wav", 
                      help="Output audio file path")
    parser.add_argument("--temperature", type=float, default=0.6, 
                      help="Temperature for sampling (higher = more diverse)")
    parser.add_argument("--top_p", type=float, default=0.95, 
                      help="Top-p sampling parameter")
    return parser.parse_args()

def setup_model(model_path, device, hf_token=None):
    print(f"Loading model from {model_path}...")
    
    # Set up environment for HF transfer if using a HF model
    if not os.path.isdir(model_path):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        local_files_only=os.path.isdir(model_path)
    ).to(device)

    print(f"Model architecture: {model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=os.path.isdir(model_path)
    )
    
    # Load SNAC model for audio decoding
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to(device)
    
    return model, tokenizer, snac_model

def format_prompt(prompt, speaker):
    # Add speaker prefix if provided and not already in prompt
    if speaker and not prompt.startswith(f"{speaker}:"):
        prompt = f"{speaker}: {prompt}"
    return prompt

def encode_prompt(prompt, tokenizer, device):
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Add special tokens (Start of human, End of text, End of human)
    start_token = torch.tensor([[ 128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    
    # Create attention mask
    attention_mask = torch.ones(modified_input_ids.shape, dtype=torch.int64)
    
    return modified_input_ids.to(device), attention_mask.to(device)

def generate_speech(model, input_ids, attention_mask, args):
    print("Generating speech...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
        )
    return generated_ids

def process_output(generated_ids, snac_model):
    # Define special tokens used in the model's tokenization
    token_to_find = 128257  # Start-of-speech token
    token_to_remove = 128258  # End-of-speech token

    # Find indices where the start-of-speech token appears
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    # Extract the last occurrence of the start-of-speech token
    if len(token_indices[1]) > 0:
        # Get the index of the last start-of-speech token
        last_occurrence_idx = token_indices[1][-1].item()
        # Crop the tensor to start after this token
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        # If no start-of-speech token is found, use the entire generated tensor
        cropped_tensor = generated_ids

    # Process each row of the cropped tensor
    processed_rows = []
    for row in cropped_tensor:
        # Remove end-of-speech tokens from each row
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    # Prepare to convert tokens to audio codes
    code_lists = []
    for row in processed_rows:
        # Ensure the row length is divisible by 7
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]

        # Subtract base value from each token
        trimmed_row = [t.item() - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    # Generate audio samples for each code list
    samples = []
    for code_list in code_lists:
        # Convert each code list to an audio sample
        audio = redistribute_codes(code_list, snac_model)
        samples.append(audio)
    
    return samples

def redistribute_codes(code_list, snac_model):
    # Initialize layers for audio code reconstruction
    layer_1 = []
    layer_2 = []
    layer_3 = []

    # Reorganize codes into specific layers
    for i in range((len(code_list)+1)//7):
        if 7*i < len(code_list):
            # First layer: first code of each 7-token group
            layer_1.append(code_list[7*i])

            # Second layer: check if indices exist
            if 7*i+1 < len(code_list) and 7*i+4 < len(code_list):
                layer_2.append(code_list[7*i+1]-4096)
                layer_2.append(code_list[7*i+4]-(4*4096))
            
            # Third layer: check if indices exist
            if 7*i+2 < len(code_list) and 7*i+3 < len(code_list) and 7*i+5 < len(code_list) and 7*i+6 < len(code_list):
                layer_3.append(code_list[7*i+2]-(2*4096))
                layer_3.append(code_list[7*i+3]-(3*4096))
                layer_3.append(code_list[7*i+5]-(5*4096))
                layer_3.append(code_list[7*i+6]-(6*4096))

    # Convert layers to tensors
    # Get device from the snac_model parameters instead of accessing .device attribute
    device = next(snac_model.parameters()).device
    codes = [
        torch.tensor(layer_1, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(layer_2, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(layer_3, dtype=torch.long).unsqueeze(0).to(device)
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

def main():
    args = parse_arguments()
    
    # Setup the model
    model, tokenizer, snac_model = setup_model(args.model_path, args.device)
    
    # Format and encode the prompt
    prompt = format_prompt(args.prompt, args.speaker)
    print(f"Using prompt: {prompt}")
    input_ids, attention_mask = encode_prompt(prompt, tokenizer, args.device)
    
    # Generate speech
    generated_ids = generate_speech(model, input_ids, attention_mask, args)
    
    # Process output and create audio
    audio_samples = process_output(generated_ids, snac_model)
    
    # Save the audio
    if audio_samples:
        save_audio(audio_samples[0], args.output)
        print(f"Successfully generated speech at {args.output}")
    else:
        print("Failed to generate audio samples")

if __name__ == "__main__":
    main() 