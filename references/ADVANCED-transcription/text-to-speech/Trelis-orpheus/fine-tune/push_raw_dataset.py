import os
import argparse
import soundfile as sf
import numpy as np
from datasets import Dataset
import colorama
from colorama import Fore, Style
import glob
import json

# Initialize colorama
colorama.init()

def push_raw_dataset(dataset_dir, dataset_name, speaker_name):
    """Push an existing dataset directory to Hugging Face Hub in raw format suitable for Orpheus"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'Push Raw Voice Dataset':^80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'Upload your previously recorded voice data to Hugging Face':^80}{Style.RESET_ALL}")
    print("=" * 80 + "\n")
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        print(f"{Fore.RED}Error: Directory {dataset_dir} does not exist!{Style.RESET_ALL}")
        return
        
    # Find all wav files
    wav_files = glob.glob(os.path.join(dataset_dir, "*.wav"))
    if not wav_files:
        print(f"{Fore.RED}Error: No .wav files found in {dataset_dir}{Style.RESET_ALL}")
        return
        
    print(f"{Fore.GREEN}Found {len(wav_files)} audio files in {dataset_dir}{Style.RESET_ALL}")
    
    # Try to read metadata from multiple sources in order of preference
    metadata = {}
    dataset_info_file = os.path.join(dataset_dir, "dataset_info.json")
    metadata_file = os.path.join(dataset_dir, "metadata.txt")
    transcripts_file = os.path.join(dataset_dir, "transcripts.json")
    
    # First check for the most detailed format
    if os.path.exists(dataset_info_file):
        print(f"{Fore.GREEN}Found dataset_info.json file, loading complete metadata...{Style.RESET_ALL}")
        try:
            with open(dataset_info_file, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
                # Extract speaker name from dataset_info if available and not provided
                if not speaker_name and "speaker" in dataset_info:
                    speaker_name = dataset_info["speaker"]
                    print(f"{Fore.GREEN}Using speaker name from dataset_info: {speaker_name}{Style.RESET_ALL}")
                
                # Build metadata dictionary from recordings list
                if "recordings" in dataset_info:
                    for recording in dataset_info["recordings"]:
                        filename = os.path.basename(recording["path"])
                        metadata[filename] = recording["text"]
                    print(f"{Fore.GREEN}Loaded {len(metadata)} transcripts from dataset_info.json{Style.RESET_ALL}")
        except json.JSONDecodeError:
            print(f"{Fore.RED}Error: dataset_info.json is not valid JSON{Style.RESET_ALL}")
    
    # Then check for the standard metadata.txt format
    elif os.path.exists(metadata_file):
        print(f"{Fore.GREEN}Found metadata.txt file, loading text transcriptions...{Style.RESET_ALL}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'): # Skip comment lines
                    continue
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    text = parts[1].strip()
                    metadata[filename] = text
            print(f"{Fore.GREEN}Loaded {len(metadata)} transcripts from metadata.txt{Style.RESET_ALL}")
    
    # Finally check for transcripts.json
    elif os.path.exists(transcripts_file):
        print(f"{Fore.GREEN}Found transcripts.json file, loading text transcriptions...{Style.RESET_ALL}")
        try:
            with open(transcripts_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                print(f"{Fore.GREEN}Loaded {len(metadata)} transcripts from transcripts.json{Style.RESET_ALL}")
        except json.JSONDecodeError:
            print(f"{Fore.RED}Error: transcripts.json is not valid JSON{Style.RESET_ALL}")
    
    # No speaker name provided and not found in metadata
    if not speaker_name:
        print(f"{Fore.YELLOW}No speaker name provided. Using directory name as speaker name.{Style.RESET_ALL}")
        speaker_name = os.path.basename(os.path.normpath(dataset_dir))
    
    # Prepare dataset items
    recordings = []
    print(f"\n{Fore.YELLOW}Preparing dataset items in Orpheus format...{Style.RESET_ALL}")
    
    for i, wav_path in enumerate(wav_files):
        filename = os.path.basename(wav_path)
        
        # Try to extract text from metadata or filename
        if filename in metadata:
            text = metadata[filename]
        else:
            # Check if filename contains timestamp followed by underscore and text
            parts = filename.split('_', 1)
            if len(parts) > 1 and parts[0].isdigit():
                # Try to extract text from filename
                text = parts[1].replace('.wav', '')
            else:
                text = "No transcript available"
                print(f"{Fore.YELLOW}No transcript found for {filename}, using default text{Style.RESET_ALL}")
        
        # Load audio file
        try:
            audio_data, sample_rate = sf.read(wav_path)
            
            # Create dataset item in the exact format expected by Orpheus tokenisation notebook
            recordings.append({
                'text': text,  # Plain text without speaker prefix
                'source': speaker_name,  # Speaker name in the source field
                'audio': {
                    'path': wav_path,
                    'array': audio_data,
                    'sampling_rate': sample_rate
                }
            })
            
            # Show progress
            progress = (i + 1) / len(wav_files) * 100
            print(f"\r{Fore.CYAN}Progress: {progress:.1f}% ({i+1}/{len(wav_files)}){Style.RESET_ALL}", end="")
            
        except Exception as e:
            print(f"\n{Fore.RED}Error loading {filename}: {e}{Style.RESET_ALL}")
    
    print("\n")
    
    if not recordings:
        print(f"{Fore.RED}No valid recordings found. Dataset creation failed.{Style.RESET_ALL}")
        return
    
    # Create dataset
    dataset = Dataset.from_list(recordings)
    print(f"{Fore.GREEN}Created raw dataset with {len(dataset)} items{Style.RESET_ALL}")
    
    # Push to Hub
    try:
        print(f"\n{Fore.YELLOW}Pushing raw dataset to Hugging Face Hub...{Style.RESET_ALL}")
        
        # Push dataset to Hugging Face Hub
        dataset.push_to_hub(dataset_name)
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Raw dataset successfully pushed to Hugging Face:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{dataset_name}{Style.RESET_ALL}")
        
        print(f"\n{Fore.MAGENTA}Next steps:{Style.RESET_ALL}")
        print(f"1. Open {Fore.CYAN}tokenise_speech_dataset.ipynb{Style.RESET_ALL}")
        print(f"2. Set {Fore.CYAN}my_original_dataset_name = \"{dataset_name}\"{Style.RESET_ALL}")
        print(f"3. Set {Fore.CYAN}name_to_push_dataset_to{Style.RESET_ALL} to your desired formatted dataset name")
        print(f"4. Run the notebook to tokenize your dataset")
        
    except Exception as e:
        print(f"\n{Fore.RED}Error pushing dataset to Hub: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please ensure you have a Hugging Face account and are logged in.{Style.RESET_ALL}")

def create_transcript_template(dataset_dir):
    """Create a template transcript file for manual editing"""
    metadata_path = os.path.join(dataset_dir, "metadata.txt")
    
    if os.path.exists(metadata_path):
        overwrite = input(f"{Fore.YELLOW}metadata.txt already exists. Overwrite? (y/n): {Style.RESET_ALL}").lower()
        if overwrite != 'y':
            print(f"{Fore.GREEN}Keeping existing metadata file.{Style.RESET_ALL}")
            return
    
    wav_files = glob.glob(os.path.join(dataset_dir, "*.wav"))
    if not wav_files:
        print(f"{Fore.RED}No .wav files found in {dataset_dir}{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}Creating transcript template for {len(wav_files)} files...{Style.RESET_ALL}")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("# Format: filename|transcript\n")
        f.write("# This file is used to match audio files with their transcripts\n")
        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            f.write(f"{filename}|Enter transcript here\n")
    
    print(f"{Fore.GREEN}Created transcript template at {metadata_path}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Please edit this file to add transcripts before pushing your dataset.{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description='Push raw voice dataset to Hugging Face Hub')
    parser.add_argument('--dir', required=True, help='Directory containing the voice recordings (.wav files)')
    parser.add_argument('--dataset_name', required=True, help='Hugging Face dataset name to push to')
    parser.add_argument('--speaker', help='Speaker name (used for multi-speaker training). If not provided, will try to extract from metadata.')
    parser.add_argument('--create_transcript', action='store_true', help='Create a metadata.txt template file')
    
    args = parser.parse_args()
    
    if args.create_transcript:
        create_transcript_template(args.dir)
    else:
        push_raw_dataset(args.dir, args.dataset_name, args.speaker)

if __name__ == '__main__':
    main() 