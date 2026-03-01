#!/usr/bin/env python3
import argparse
import csv
import os
import random
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from colorama import Fore, Style, init

# Initialize colorama
init()

# Define some conversational training phrases
TRAINING_PHRASES = [
    # Greetings and common expressions
    "Hello, how can I help you today?",
    "Good morning! I hope you're having a great day.",
    "I'd be happy to assist you with that request.",
    "Let me think about that for a moment.",
    "That's an interesting question. Let me explain.",
    
    # Questions
    "Would you like me to continue with more details?",
    "Do you need any clarification on what I just explained?",
    "How does that solution sound to you?",
    "What specific aspect would you like to know more about?",
    "Is there anything else you'd like me to help with today?",
    
    # Informational statements
    "The process involves several important steps that we should follow.",
    "Based on the information you provided, I can suggest a few options.",
    "There are multiple approaches we could take to solve this problem.",
    "I've analyzed the data and found some interesting patterns.",
    "This technique has been proven effective in similar situations.",
    
    # Responses
    "I understand your concern, and I think we can address it effectively.",
    "That's a great point. Let me build on that idea.",
    "I agree with your assessment of the situation.",
    "You're absolutely right about that observation.",
    "I can see why you might feel that way about this issue.",
    
    # Longer, more complex sentences
    "The combination of these factors creates a unique opportunity that we should carefully consider before making any decisions.",
    "While there are several potential solutions to this problem, I believe we should focus on the one that offers the best long-term benefits.",
    "If we analyze the historical data alongside current trends, we can make more accurate predictions about future outcomes.",
    "When implementing this approach, it's important to maintain balance between efficiency and thoroughness to achieve optimal results.",
    "Although this may seem challenging at first, breaking it down into smaller steps will make the process much more manageable.",
    
    # Transitions and conclusions
    "Now that we've covered the basics, let's move on to the more advanced concepts.",
    "To summarize what we've discussed so far, there are three key points to remember.",
    "In conclusion, I believe this approach offers the best solution to your problem.",
    "Based on all the information we've reviewed, my recommendation would be to proceed with option two.",
    "Thank you for your patience while I worked through this analysis."
]

def parse_args():
    parser = argparse.ArgumentParser(description='Voice dataset collection tool')
    parser.add_argument('--output_dir', type=str, default='fine-tune/dataset',
                        help='Output directory for the dataset')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples to record')
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='Sample rate for audio recording')
    parser.add_argument('--speaker_id', type=int, default=0,
                        help='Speaker ID for the dataset')
    parser.add_argument('--phrases_file', type=str, default=None,
                        help='File containing custom phrases to read (one per line)')
    return parser.parse_args()


def list_audio_devices():
    """List available audio input devices"""
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print(f"{Fore.CYAN}Available audio input devices:{Style.RESET_ALL}")
    input_devices = []
    
    for i in range(numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            input_devices.append((i, device_info.get('name')))
            print(f"  [{i}] {device_info.get('name')}")
    
    p.terminate()
    return input_devices


def select_audio_device(input_devices):
    """Prompt user to select an audio input device"""
    if not input_devices:
        print(f"{Fore.RED}No input devices found!{Style.RESET_ALL}")
        return None
    
    while True:
        try:
            device_idx = input(f"{Fore.YELLOW}Select input device number: {Style.RESET_ALL}")
            device_idx = int(device_idx)
            if device_idx < 0 or device_idx >= len(input_devices):
                print(f"{Fore.RED}Invalid device number. Please try again.{Style.RESET_ALL}")
                continue
            return device_idx
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")


def record_audio(device_idx, duration=None, sample_rate=24000):
    """Record audio from the selected device"""
    import threading
    import queue

    p = pyaudio.PyAudio()
    
    # Set up a queue to communicate between threads
    audio_queue = queue.Queue()
    stop_recording = threading.Event()
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_idx,
        frames_per_buffer=1024
    )
    
    print(f"{Fore.GREEN}Recording... {Style.RESET_ALL}(Press Enter to stop)", end='', flush=True)
    
    frames = []
    max_duration = 60  # Maximum recording duration in seconds as a safeguard
    
    # Function to run in a separate thread to collect frames
    def collect_audio():
        start_time = time.time()
        try:
            while not stop_recording.is_set() and (time.time() - start_time) < max_duration:
                data = stream.read(1024, exception_on_overflow=False)
                audio_queue.put(data)
                
                # Convert to numpy array for visualization
                current_frame = np.frombuffer(data, dtype=np.int16)
                frame_volume = np.abs(current_frame).mean()
                
                # Print audio level
                level = min(40, int(frame_volume / 100))
                print('\r' + f"{Fore.GREEN}Recording... {Style.RESET_ALL}(Press Enter to stop) " + '█' * level + ' ' * (40 - level), end='', flush=True)
        except Exception as e:
            print(f"\n{Fore.RED}Error during recording: {e}{Style.RESET_ALL}")
        finally:
            audio_queue.put(None)  # Signal end of recording
    
    # Start the recording thread
    record_thread = threading.Thread(target=collect_audio)
    record_thread.daemon = True
    record_thread.start()
    
    # Wait for user to press Enter to stop recording
    input()
    stop_recording.set()
    
    # Collect all frames from the queue
    while True:
        data = audio_queue.get()
        if data is None:
            break
        frames.append(data)
    
    print('\r' + ' ' * 80, end='', flush=True)
    print(f"\r{Fore.GREEN}Recording complete!{Style.RESET_ALL}")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert frames to audio data
    if not frames:
        print(f"{Fore.RED}No audio was recorded. Please try again.{Style.RESET_ALL}")
        return torch.zeros(1)
        
    audio_data = b''.join(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    return torch.tensor(audio_array)


def play_audio(audio_tensor, sample_rate=24000):
    """Play back the recorded audio"""
    try:
        sd.play(audio_tensor.numpy(), sample_rate)
        sd.wait()
    except Exception as e:
        print(f"{Fore.RED}Error playing audio: {e}{Style.RESET_ALL}")


def save_dataset(output_dir, recordings, phrases, speaker_id):
    """Save the dataset to disk"""
    wavs_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)
    
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    
    with open(metadata_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        writer.writerow(['filename', 'text', 'speaker_id'])
        
        for i, (audio, text) in enumerate(zip(recordings, phrases)):
            filename = f'sample_{i+1:03d}.wav'
            filepath = os.path.join(wavs_dir, filename)
            
            # Save the audio file
            sf.write(filepath, audio.numpy(), 24000)
            
            # Write metadata
            writer.writerow([filename, text, speaker_id])
    
    print(f"{Fore.GREEN}Dataset saved to {output_dir}{Style.RESET_ALL}")
    print(f"  - {len(recordings)} audio samples")
    print(f"  - Metadata file: {metadata_path}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load custom phrases if provided
    phrases = TRAINING_PHRASES
    if args.phrases_file and os.path.exists(args.phrases_file):
        with open(args.phrases_file, 'r') as f:
            custom_phrases = [line.strip() for line in f if line.strip()]
            if custom_phrases:
                phrases = custom_phrases
    
    # Shuffle and select phrases
    random.shuffle(phrases)
    selected_phrases = phrases[:args.num_samples]
    
    print(f"{Fore.CYAN}" + "=" * 80)
    print(f"Voice Dataset Collection Tool")
    print("=" * 80)
    print(f"This tool will guide you through recording {args.num_samples} voice samples for fine-tuning.")
    print(f"Each sample should be a clear reading of the displayed text.")
    print(f"Speak naturally in a quiet environment for best results.")
    print(f"RECORDING INSTRUCTIONS:")
    print(f"  1. Press Enter to START recording when prompted")
    print(f"  2. Read the entire phrase clearly")
    print(f"  3. Press Enter to STOP recording when you've finished")
    print(f"{Fore.CYAN}" + "=" * 80 + f"{Style.RESET_ALL}")
    
    # List and select input device
    input_devices = list_audio_devices()
    device_idx = select_audio_device(input_devices)
    if device_idx is None:
        return
    
    # Test recording
    print(f"{Fore.YELLOW}Let's do a quick test recording. Say something...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Press Enter to start recording, then press Enter again to stop when you're done.{Style.RESET_ALL}")
    input(f"{Fore.YELLOW}Press Enter to begin test recording...{Style.RESET_ALL}")
    test_audio = record_audio(device_idx, sample_rate=args.sample_rate)
    
    print(f"{Fore.YELLOW}Playing back test recording...{Style.RESET_ALL}")
    play_audio(test_audio, args.sample_rate)
    
    proceed = input(f"{Fore.YELLOW}Does the recording sound clear? (y/n): {Style.RESET_ALL}").lower()
    if proceed != 'y':
        print(f"{Fore.RED}Recording test failed. Please try again with a different device or settings.{Style.RESET_ALL}")
        return
    
    # Start collecting samples
    recordings = []
    recorded_phrases = []
    
    for i, phrase in enumerate(selected_phrases):
        print(f"\n{Fore.CYAN}Sample {i+1}/{args.num_samples}:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{phrase}{Style.RESET_ALL}")
        
        input(f"{Fore.YELLOW}Press Enter to start recording (and press Enter again to stop when done)...{Style.RESET_ALL}")
        audio = record_audio(device_idx, sample_rate=args.sample_rate)
        
        # Skip if recording is empty
        if audio.numel() <= 1:
            print(f"{Fore.YELLOW}Empty recording detected. Let's try again.{Style.RESET_ALL}")
            continue
        
        print(f"{Fore.YELLOW}Playback:{Style.RESET_ALL}")
        play_audio(audio, args.sample_rate)
        
        keep = input(f"{Fore.YELLOW}Keep this recording? (y/n): {Style.RESET_ALL}").lower()
        if keep == 'y':
            recordings.append(audio)
            recorded_phrases.append(phrase)
        else:
            print(f"{Fore.YELLOW}Discarded. Let's try again.{Style.RESET_ALL}")
            i -= 1
        
        if len(recordings) >= args.num_samples:
            break
    
    # Save the dataset
    save_dataset(args.output_dir, recordings, recorded_phrases, args.speaker_id)
    
    print(f"\n{Fore.GREEN}Dataset collection complete!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}You can now use this dataset to fine-tune the CSM model with:{Style.RESET_ALL}")
    print(f"  uv run python fine-tune/train.py --dataset_path {args.output_dir}")


if __name__ == "__main__":
    main() 