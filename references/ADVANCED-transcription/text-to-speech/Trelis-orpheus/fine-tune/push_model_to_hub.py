#!/usr/bin/env python3
"""
Script to push an Orpheus fine-tuned model to Hugging Face Hub
"""

import os
import argparse
import colorama
from colorama import Fore, Style
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize colorama for colored output
colorama.init()

def push_model_to_hub(model_path, repo_id, private=False):
    """Push a fine-tuned model to Hugging Face Hub"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'Push Model to Hugging Face Hub':^80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'Upload your fine-tuned Orpheus model to Hugging Face Hub':^80}{Style.RESET_ALL}")
    print("=" * 80 + "\n")
    
    # Check if directory exists
    if not os.path.exists(model_path):
        print(f"{Fore.RED}Error: Model directory {model_path} does not exist!{Style.RESET_ALL}")
        return False
    
    # Check if directory contains model files
    if not os.path.exists(os.path.join(model_path, "config.json")) or \
       not os.path.exists(os.path.join(model_path, "pytorch_model.bin")) and \
       not any(f.startswith("model.safetensors") for f in os.listdir(model_path)):
        print(f"{Fore.RED}Error: {model_path} doesn't appear to be a valid model directory!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}The directory should contain config.json and pytorch_model.bin or model.safetensors files.{Style.RESET_ALL}")
        return False
    
    try:
        
        # Load model and tokenizer to verify they're valid
        print(f"{Fore.YELLOW}Validating model files...{Style.RESET_ALL}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        
        # Set up environment for HF transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # Create repository description
        model_card = f"""---
language: en
tags:
- orpheus
- text-to-speech
- tts
- fine-tuned
---

# {repo_id.split('/')[-1]}

This is a fine-tuned Orpheus TTS model based on [canopylabs/orpheus-tts-0.1-pretrained](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained).

## Model Details
- **Fine-tuned from:** canopylabs/orpheus-tts-0.1-pretrained
- **Model type:** Autoregressive text-to-speech

## Usage Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf

# Load the model
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

# Set up your prompt
prompt = "Hello, this is a test of my fine-tuned voice."

# Generate
# ... (see the inference.py script in the Trelis-Orpheus repo for full inference code)
```

## Related Links
- [Trelis-Orpheus repo](https://github.com/canopyai/Orpheus-TTS) - Base repository and fine-tuning scripts
"""
        
        # Push the model to Hub
        print(f"{Fore.YELLOW}Pushing model to Hugging Face Hub ({repo_id})...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}This may take some time depending on model size and internet speed.{Style.RESET_ALL}")
        
        # Push using the Hugging Face API
        tokenizer.push_to_hub(repo_id, private=private)
        model.push_to_hub(repo_id, private=private)
        
        # Create and upload a README.md
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Model successfully pushed to Hugging Face Hub!{Style.RESET_ALL}")
        print(f"Model URL: {Fore.CYAN}https://huggingface.co/{repo_id}{Style.RESET_ALL}")
        
        # Next steps for users
        print(f"\n{Fore.MAGENTA}Next steps:{Style.RESET_ALL}")
        print(f"1. Visit your model page: {Fore.CYAN}https://huggingface.co/{repo_id}{Style.RESET_ALL}")
        print(f"2. Fill in any additional model card details")
        print(f"3. To use your model for inference:")
        print(f"   - Run: {Fore.CYAN}uv run python inference.py --model_path {repo_id} --prompt \"Your text here\"{Style.RESET_ALL}")
        
        return True
    
    except Exception as e:
        print(f"\n{Fore.RED}Error pushing model to Hub: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please ensure you have a Hugging Face account and are logged in with huggingface-cli login{Style.RESET_ALL}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Push a fine-tuned Orpheus model to Hugging Face Hub')
    parser.add_argument('--model_path', required=True, 
                        help='Path to the fine-tuned model directory')
    parser.add_argument('--repo_id', required=True, 
                        help='Hugging Face repository ID to push to (e.g., username/model-name)')
    parser.add_argument('--private', action='store_true',
                        help='Create a private model repository')
    
    args = parser.parse_args()
    push_model_to_hub(args.model_path, args.repo_id, args.private)

if __name__ == '__main__':
    main() 