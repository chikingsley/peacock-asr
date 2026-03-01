"""
Voice Dataset Recorder for Orpheus TTS Fine-Tuning

This script provides an interactive way to record a voice dataset for fine-tuning
the Orpheus Text-to-Speech model. It guides the user through recording a series of
phrases, allowing them to review and re-record if needed.

Features:
- Records voice samples in the required format for Orpheus TTS
- Organizes phrases by category (basic, technical, conversational, etc.)
- Allows selection of input recording device
- Provides audio playback for quality control
- Saves metadata in multiple formats
- Pushes dataset to Hugging Face Hub
- Supports "test mode" for quick verification

Usage:
    python record_voice_dataset.py --speaker "YourName" --dataset_name "your-username/dataset-name" [--test] [--device "Device Name"]

Requirements:
    - sounddevice, soundfile, numpy, datasets, huggingface_hub, colorama
"""

import os
import time
import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi, login
import threading
import colorama
from colorama import Fore, Style, Back
import random
import json

# Initialize colorama
colorama.init()

class VoiceDatasetRecorder:
    def __init__(self, speaker_name, output_dir='voice_dataset', sample_rate=24000, test_mode=False, device=None):
        self.speaker_name = speaker_name
        self.output_dir = os.path.join(output_dir, speaker_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.sample_rate = sample_rate
        self.test_mode = test_mode
        self.device = device  # Store the selected device
        
        # Define phrase categories
        self.phrase_categories = {
            "Basic Sentences": [
                "The quick brown fox jumps over the lazy dog.",
                "I would like a cup of coffee with milk and sugar.",
                "Can you please tell me how to get to the nearest bus station?",
                "The weather today is absolutely beautiful.",
                "I'm planning to go on vacation next month."
            ],
            "Technical": [
                "Machine learning is transforming how we understand complex systems.",
                "The latest advancements in quantum computing are quite fascinating.",
                "Artificial intelligence can help solve some of humanity's greatest challenges.",
                "Cloud computing has revolutionized the way businesses operate.",
                "Blockchain technology offers a new paradigm for secure transactions."
            ],
            "Conversational": [
                "I'm not sure about that, what do you think?",
                "That's an interesting perspective, I hadn't considered it that way.",
                "Could you elaborate a bit more on what you mean?",
                "I see your point, but I respectfully disagree.",
                "Let me think about that for a moment before I respond."
            ],
            "Emotional": [
                "I'm so excited about the upcoming concert this weekend!",
                "That news made me feel quite sad and disappointed.",
                "I can't believe how frustrating this situation has become.",
                "The surprise party they organized for me was absolutely heartwarming.",
                "I'm feeling quite anxious about the presentation tomorrow."
            ],
            "Questions": [
                "What are your thoughts on renewable energy sources?",
                "How do you think technology will change in the next decade?",
                "Have you ever visited a foreign country?",
                "Do you prefer reading books or watching movies?",
                "What would you consider your greatest accomplishment?"
            ],
            "Long Sentences": [
                "Despite the challenges we've faced over the past few years, including economic downturns and global health crises, I remain optimistic about our capacity for innovation and resilience in the face of adversity.",
                "The intricate relationship between technological advancement and societal well-being necessitates thoughtful consideration of ethical implications, particularly as artificial intelligence continues to reshape how we live, work, and interact with one another.",
                "When considering the multifaceted nature of climate change, we must acknowledge that effective solutions will require unprecedented cooperation between governments, industries, and individuals across the globe, transcending political and economic differences.",
                "The book I recently finished reading not only presented a compelling narrative about historical events that shaped our modern world, but also offered profound insights into human psychology and the complex motivations that drive people to make difficult choices under pressure.",
                "As we navigate through an increasingly interconnected world where information travels instantaneously across borders, we must develop more sophisticated frameworks for distinguishing between reliable knowledge and misleading content that can polarize communities and undermine democratic institutions."
            ]
        }
        
        # Select phrases based on mode
        self.phrases = []
        if test_mode:
            # In test mode, take one phrase from each category (up to 4)
            categories = list(self.phrase_categories.keys())[:4]
            for category in categories:
                self.phrases.append(random.choice(self.phrase_categories[category]))
        else:
            # In normal mode, take multiple phrases from each category
            for category, category_phrases in self.phrase_categories.items():
                # Take 3 random phrases from each category or all if less than 3
                selected = random.sample(category_phrases, min(3, len(category_phrases)))
                self.phrases.extend(selected)

    def select_device(self):
        """List available audio devices and allow user to select one"""
        devices = sd.query_devices()
        input_devices = [device for device in devices if device['max_input_channels'] > 0]
        
        print(f"\n{Fore.YELLOW}Available Input Devices:{Style.RESET_ALL}")
        for i, device in enumerate(input_devices):
            print(f"{Fore.CYAN}[{i+1}]{Style.RESET_ALL} {device['name']} ({device['max_input_channels']} channels)")
        
        # Show default device with asterisk
        default_device = sd.query_devices(kind='input')
        default_index = next((i for i, device in enumerate(input_devices) 
                              if device['name'] == default_device['name']), None)
        
        if default_index is not None:
            print(f"\n{Fore.GREEN}Default device: {Fore.CYAN}[{default_index+1}]{Style.RESET_ALL} {default_device['name']}{Style.RESET_ALL}")
        
        # Ask user to select a device
        while True:
            choice = input(f"\n{Fore.YELLOW}Select an input device (1-{len(input_devices)}) or press Enter for default: {Style.RESET_ALL}")
            
            if choice == '':
                # Use default device
                self.device = None
                print(f"{Fore.GREEN}Using default input device: {default_device['name']}{Style.RESET_ALL}")
                break
            
            try:
                index = int(choice) - 1
                if 0 <= index < len(input_devices):
                    self.device = input_devices[index]['name']
                    print(f"{Fore.GREEN}Selected device: {self.device}{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.RED}Invalid selection. Please enter a number between 1 and {len(input_devices)}.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Invalid input. Please enter a number or press Enter.{Style.RESET_ALL}")
    
    def display_banner(self):
        """Display a nice banner for the application"""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{Style.BRIGHT}{'Voice Dataset Recorder':^80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'Record your voice to fine-tune Orpheus TTS':^80}{Style.RESET_ALL}")
        print("=" * 80 + "\n")
        
        print(f"{Fore.YELLOW}Speaker: {Fore.WHITE}{self.speaker_name}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Mode: {Fore.WHITE}{'Test' if self.test_mode else 'Full'}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Total phrases: {Fore.WHITE}{len(self.phrases)}{Style.RESET_ALL}")
        
        # Display selected device
        if self.device:
            print(f"{Fore.YELLOW}Recording device: {Fore.WHITE}{self.device}{Style.RESET_ALL}\n")
        else:
            default_device = sd.query_devices(kind='input')
            print(f"{Fore.YELLOW}Recording device: {Fore.WHITE}{default_device['name']} (default){Style.RESET_ALL}\n")
        
        print(f"{Fore.MAGENTA}Instructions:{Style.RESET_ALL}")
        print("1. Read each phrase aloud after pressing Enter")
        print("2. Press Enter again to stop recording")
        print("3. Listen to your recording and decide whether to keep it")
        print("4. Repeat for all phrases\n")
        
        input(f"{Fore.GREEN}Press Enter to begin...{Style.RESET_ALL}")

    def countdown_timer(self, seconds=3):
        # """Display a countdown timer"""
        # for i in range(seconds, 0, -1):
        #     print(f"\r{Fore.YELLOW}Starting in {i}...{Style.RESET_ALL}", end="")
        #     time.sleep(1)
        print(f"\r{Fore.GREEN}Recording now! {Style.RESET_ALL}Press Enter to stop...{' ' * 20}")

    def record_audio(self, phrase, category=""):
        """Record audio for a phrase"""
        # Clear space and show phrase with category
        print("\n" + "-" * 80)
        if category:
            print(f"{Fore.BLUE}Category: {Style.BRIGHT}{category}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}Please read:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}\"{phrase}\"{Style.RESET_ALL}\n")
        
        input(f"{Fore.YELLOW}Press Enter to start recording...{Style.RESET_ALL}")
        
        # Start countdown
        self.countdown_timer()
        
        # Start recording
        recording_data = []
        recording_finished = threading.Event()
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Error: {status}")
            recording_data.append(indata.copy())
        
        # Start the recording stream with the selected device
        stream = sd.InputStream(callback=callback, 
                               channels=1, 
                               samplerate=self.sample_rate,
                               device=self.device)
        with stream:
            input()  # Wait for the user to press Enter to stop recording
        
        # Combine all recorded chunks
        recording = np.concatenate(recording_data, axis=0) if recording_data else np.array([])
        
        # Save recording
        filename = f"{int(time.time())}_{self.speaker_name}.wav"
        filepath = os.path.join(self.output_dir, filename)
        sf.write(filepath, recording, self.sample_rate)
        
        print(f"\n{Fore.GREEN}Recording saved: {Style.BRIGHT}{filename}{Style.RESET_ALL}")
        
        # Playback recording
        print(f"\n{Fore.BLUE}Playing back your recording...{Style.RESET_ALL}")
        sd.play(recording, self.sample_rate)
        sd.wait()
        
        # Ask user if they want to keep it
        while True:
            choice = input(f"\n{Fore.YELLOW}Keep this recording? (y/n): {Style.RESET_ALL}").lower()
            if choice == 'y':
                print(f"{Fore.GREEN}Recording kept!{Style.RESET_ALL}")
                return filepath, recording, phrase, category
            elif choice == 'n':
                print(f"{Fore.RED}Recording discarded. Deleting file...{Style.RESET_ALL}")
                # Delete the discarded recording file
                try:
                    os.remove(filepath)
                    print(f"{Fore.GREEN}Deleted: {filename}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error deleting file: {e}{Style.RESET_ALL}")
                # Try recording again
                return self.record_audio(phrase, category)
            else:
                print(f"{Fore.RED}Invalid choice. Please enter 'y' or 'n'.{Style.RESET_ALL}")

    def save_metadata_locally(self, metadata_list):
        """Save metadata locally in both JSON and text formats"""
        # Create metadata.txt (pipe-delimited format)
        metadata_txt_path = os.path.join(self.output_dir, "metadata.txt")
        with open(metadata_txt_path, 'w', encoding='utf-8') as f:
            f.write("# Format: filename|transcript|category\n")
            for item in metadata_list:
                filename = os.path.basename(item['path'])
                f.write(f"{filename}|{item['text']}|{item['category']}\n")
        
        # Create transcripts.json (JSON format)
        transcripts_json_path = os.path.join(self.output_dir, "transcripts.json")
        transcripts = {}
        for item in metadata_list:
            filename = os.path.basename(item['path'])
            transcripts[filename] = item['text']
        
        with open(transcripts_json_path, 'w', encoding='utf-8') as f:
            json.dump(transcripts, f, indent=2)
        
        # Get device information for metadata
        if self.device:
            device_info = self.device
        else:
            default_device = sd.query_devices(kind='input')
            device_info = f"{default_device['name']} (default)"
        
        # Create dataset_info.json (detailed format with all metadata)
        dataset_info_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            dataset_info = {
                "speaker": self.speaker_name,
                "sample_rate": self.sample_rate,
                "recording_device": device_info,
                "recordings": metadata_list,
                "recording_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n{Fore.GREEN}Metadata saved locally:{Style.RESET_ALL}")
        print(f"  - {Fore.CYAN}metadata.txt{Style.RESET_ALL} (for text-based tools)")
        print(f"  - {Fore.CYAN}transcripts.json{Style.RESET_ALL} (for JSON-based tools)")
        print(f"  - {Fore.CYAN}dataset_info.json{Style.RESET_ALL} (complete dataset info)")
        print(f"\n{Fore.YELLOW}You can use these files with push_raw_dataset.py if needed later{Style.RESET_ALL}")

    def create_dataset(self):
        """Create a dataset from recorded phrases"""
        # Ask user to select a recording device before starting
        self.select_device()
        
        self.display_banner()
        
        recordings = []
        metadata_list = []
        total_phrases = len(self.phrases)
        
        # Get current categories for each phrase
        phrase_to_category = {}
        for category, phrases in self.phrase_categories.items():
            for phrase in phrases:
                phrase_to_category[phrase] = category
        
        for i, phrase in enumerate(self.phrases):
            category = phrase_to_category.get(phrase, "")
            print(f"\n{Fore.CYAN}Recording {i+1}/{total_phrases}{Style.RESET_ALL}")
            
            filepath, audio_data, text, category = self.record_audio(phrase, category)
            
            # Store metadata for local saving
            metadata_list.append({
                'path': filepath,
                'text': text,
                'category': category,
            })
            
            # Create the dataset item in the exact format expected by Orpheus tokenisation notebook
            recordings.append({
                'text': text,  # Plain text without speaker prefix
                'source': self.speaker_name,  # Speaker name for the source field
                'audio': {
                    'path': filepath,
                    'array': audio_data,
                    'sampling_rate': self.sample_rate
                }
            })
            
            # Display progress
            progress = (i + 1) / total_phrases * 100
            print(f"\n{Fore.CYAN}Progress: {Style.BRIGHT}{progress:.1f}%{Style.RESET_ALL}")
            
            # After each recording, except the last one, ask if user wants to continue
            if i < total_phrases - 1:
                print(f"\n{Fore.YELLOW}Press Enter to continue to next phrase or type 'quit' to stop...{Style.RESET_ALL}")
                choice = input()
                if choice.lower() == 'quit':
                    print(f"\n{Fore.RED}Recording session ended early.{Style.RESET_ALL}")
                    break
        
        # Save metadata locally
        self.save_metadata_locally(metadata_list)
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}All recordings completed!{Style.RESET_ALL}")
        return Dataset.from_list(recordings)

    def push_to_hub(self, dataset, dataset_name):
        """Push raw dataset to Hugging Face Hub"""
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
            print(f"{Fore.YELLOW}If the push failed, use push_raw_dataset.py to try again.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}The metadata has been saved locally in the {self.output_dir} directory.{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description='Record voice samples for TTS dataset')
    parser.add_argument('--speaker', required=True, help='Name of the speaker')
    parser.add_argument('--dataset_name', required=True, help='Hugging Face dataset name to push to')
    parser.add_argument('--test', action='store_true', help='Run in test mode with fewer phrases')
    parser.add_argument('--device', help='Specify input device name for recording (if not provided, will prompt user to select or use default)')
    
    args = parser.parse_args()
    
    recorder = VoiceDatasetRecorder(args.speaker, test_mode=args.test, device=args.device)
    dataset = recorder.create_dataset()
    recorder.push_to_hub(dataset, args.dataset_name)

if __name__ == '__main__':
    main() 