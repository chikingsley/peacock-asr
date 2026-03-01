#!/usr/bin/env python3
"""
Script to tokenize speech datasets for Orpheus fine-tuning

Based on an original file is located at
    https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD
"""

import os
import sys
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm
from datasets import load_dataset, Dataset, Audio, Features, Value, Sequence
from huggingface_hub import snapshot_download, HfApi, create_repo
from snac import SNAC
from transformers import AutoTokenizer

# Fix for locale issues with PyTorch
# This prevents the "<lambda>() takes 0 positional arguments but 1 was given" error
import locale
if hasattr(locale, 'getpreferredencoding'):
    # Save the original function
    _original_getpreferredencoding = locale.getpreferredencoding
    
    # Replace it with a function that ignores the parameter
    def _fixed_getpreferredencoding(do_setlocale=False):
        return _original_getpreferredencoding()
    
    # Replace the function in the locale module
    locale.getpreferredencoding = _fixed_getpreferredencoding


def tokenise_audio(model, waveform, sample_rate, device):
    """Tokenize audio using SNAC model - matches the original notebook implementation"""
    # Ensure waveform is a numpy array
    if isinstance(waveform, list):
        waveform = np.array(waveform, dtype=np.float32)
    
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to(device)

    # Generate the codes from SNAC using the encode method
    with torch.inference_mode():
        codes = model.encode(waveform)

    # Process codes like in the original notebook
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


def remove_duplicate_frames(codes_list):
    """Remove duplicate frames from the codes list (identical to original notebook)"""
    if len(codes_list) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = codes_list[:7]
    removed_frames = 0

    for i in range(7, len(codes_list), 7):
        current_first = codes_list[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(codes_list[i:i+7])
        else:
            removed_frames += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Tokenize speech dataset for Orpheus fine-tuning")
    parser.add_argument("--input_dataset", type=str, required=True, 
                        help="Input dataset name on Hugging Face Hub")
    parser.add_argument("--output_dataset", type=str, required=True, 
                        help="Output dataset name to push to Hugging Face Hub")
    parser.add_argument("--device", type=str, default="cpu", 
                        choices=["cpu", "cuda", "mps"], 
                        help="Device to use for tokenization")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing")
    parser.add_argument("--private", action="store_true",
                        help="Create a private dataset repository")
    parser.add_argument("--tokenizer", type=str, default="canopylabs/orpheus-tts-0.1-pretrained",
                        help="Tokenizer to use for text encoding")
    
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    # Ensure the output repository exists (create if it doesn't)
    try:
        api = HfApi()
        print(f"Creating/ensuring dataset repository '{args.output_dataset}'...")
        create_repo(args.output_dataset, repo_type="dataset", private=args.private, exist_ok=True)
    except Exception as e:
        print(f"Note about repository creation: {e}")
    
    # Download raw dataset
    print(f"Downloading dataset: {args.input_dataset}")
    try:
        snapshot_download(
            repo_id=args.input_dataset,
            repo_type="dataset",
            revision="main",
            max_workers=64,
        )
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Continue anyway as the dataset might already be downloaded
    
    # Load the dataset
    print("Loading dataset...")
    ds = load_dataset(args.input_dataset, split="train")
    
    # Verify sample_rate exists in the first example
    if "audio" not in ds[0] or "sampling_rate" not in ds[0]["audio"]:
        print("Error: Dataset doesn't have the expected audio format with sampling_rate.")
        sys.exit(1)
        
    ds_sample_rate = ds[0]["audio"]["sampling_rate"]
    print(f"Dataset sample rate: {ds_sample_rate}")
    
    # Load the SNAC model
    print("Loading SNAC model...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    model = model.to(device)
    model.eval()
    
    # Load the tokenizer for text encoding
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Define the special tokens (same as original notebook)
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009
    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2
    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4
    start_of_ai = tokeniser_length + 5
    end_of_ai = tokeniser_length + 6
    pad_token = tokeniser_length + 7
    audio_tokens_start = tokeniser_length + 10
    
    # Process the dataset
    print("Processing dataset...")
    processed_data = {
        "input_ids": [],
        "labels": [],
        "attention_mask": []
    }
    
    num_examples = 0
    
    for i, example in enumerate(tqdm(ds)):
        try:
            source = example["source"] if "source" in example else ""
            text = example["text"]
            
            # Format text with speaker name if available (for multi-speaker models)
            if source:
                formatted_text = f"{source}: {text}"
            else:
                formatted_text = text
            
            # Extract audio array
            if isinstance(example["audio"], dict) and "array" in example["audio"]:
                audio = example["audio"]["array"]
            else:
                print(f"Warning: Unexpected audio format in example {i}. Skipping.")
                continue
            
            # Tokenize audio
            codes_list = tokenise_audio(model, audio, ds_sample_rate, device)
            
            # Remove duplicate frames (optional but recommended)
            codes_list = remove_duplicate_frames(codes_list)
            
            # Tokenize text
            text_tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
            text_tokens.append(end_of_text)
            
            # Create the complete input_ids sequence with special tokens
            input_ids = (
                [start_of_human]
                + text_tokens
                + [end_of_human]
                + [start_of_ai]
                + [start_of_speech]
                + codes_list
                + [end_of_speech]
                + [end_of_ai]
            )
            
            # Create attention mask (all 1s)
            attention_mask = [1] * len(input_ids)
            
            # Store processed data
            processed_data["input_ids"].append(input_ids)
            processed_data["labels"].append(input_ids)  # For causal LM, labels = input_ids
            processed_data["attention_mask"].append(attention_mask)
            
            num_examples += 1
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
    
    # Define features schema for better dataset visibility
    features = Features({
        'input_ids': Sequence(Value('int64')),
        'labels': Sequence(Value('int64')),
        'attention_mask': Sequence(Value('int8'))
    })
    
    # Check if we have any processed examples
    if num_examples == 0:
        print("Error: No examples were successfully processed. Check your dataset format.")
        sys.exit(1)
    
    print(f"Creating dataset with {num_examples} processed examples...")
    formatted_ds = Dataset.from_dict(
        processed_data,
        features=features
    )
    
    # Add dataset metadata for better discoverability
    formatted_ds.info.description = f"Tokenized speech dataset for Orpheus fine-tuning. Original dataset: {args.input_dataset}"
    formatted_ds.info.license = "Same as source dataset"
    formatted_ds.info.citation = f"Derived from {args.input_dataset}"
    
    # Push to Hub with visibility settings and explicit config
    print(f"Pushing formatted dataset to {args.output_dataset}...")
    formatted_ds.push_to_hub(
        args.output_dataset,
        private=args.private,
        embed_external_files=False,
        config_name="default"  # Explicitly name the config
    )
    
    # Create a simple README.md for the dataset to improve discoverability
    readme_content = f"""# {args.output_dataset.split('/')[-1]}

This dataset contains tokenized speech for Orpheus fine-tuning.

## Dataset details
- Source dataset: {args.input_dataset}
- Number of examples: {num_examples}
- Format: Each example contains input_ids, labels, and attention_mask in the format expected by Orpheus models.

## Usage
This dataset is ready for fine-tuning Orpheus TTS models.

## Creation
Created with the tokenise_speech_dataset.py script from the Trelis-Orpheus repository.
"""
    
    # Upload README directly
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=args.output_dataset,
        repo_type="dataset"
    )
    
    print(f"\nDone! Dataset pushed to {args.output_dataset}")
    print("You can now use this dataset for fine-tuning Orpheus.\n")
    print(f"To check your dataset: huggingface.co/datasets/{args.output_dataset}")
    print("To confirm dataset structure, run: check_dataset.py --dataset " + args.output_dataset)


if __name__ == "__main__":
    main() 