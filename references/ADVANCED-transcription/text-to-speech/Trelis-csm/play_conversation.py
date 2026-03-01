#!/usr/bin/env python
"""
Play Conversation Script

This script plays all the audio files from a conversation created by context-aware-tts.py
in sequence, simulating a complete conversation. It handles both the case where a human
recording was used and when a dummy response was generated.
"""

import os
import sys
import time
import argparse
import platform
import subprocess
from pathlib import Path
import torchaudio
import pygame

def play_audio_file(file_path):
    """
    Play an audio file using the appropriate method for the current platform.
    Returns the duration of the audio file in seconds.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist. Skipping.")
        return 0
        
    print(f"Playing: {os.path.basename(file_path)}")
    
    # Get audio duration using torchaudio instead of wave
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        duration = waveform.shape[1] / sample_rate  # Calculate duration from waveform size
    except Exception as e:
        print(f"Warning: Could not determine audio duration: {e}")
        duration = 0  # Set a default duration
        
    # Use platform-specific audio player
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["afplay", file_path])
    else:
        # For other platforms, use pygame
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            # Wait for audio to finish
            if duration > 0:
                pygame.time.wait(int(duration * 1000))
            else:
                # If duration couldn't be determined, wait until playback ends
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"Error playing audio with pygame: {e}")
            # Fallback to system player if available
            try:
                if platform.system() == "Windows":
                    os.system(f'start {file_path}')
                elif platform.system() == "Linux":
                    os.system(f'xdg-open {file_path}')
            except:
                print(f"Could not play audio file: {file_path}")
    
    return duration

def display_transcript(turn_number, speaker, text):
    """Display the transcript for the current turn"""
    print("\n" + "="*50)
    print(f"Turn {turn_number} ({speaker}):")
    print("-"*50)
    print(text)
    print("="*50)
    
def main():
    parser = argparse.ArgumentParser(description="Play a complete conversation from context-aware-tts.py")
    parser.add_argument("--dir", type=str, default="data/conversation", 
                        help="Directory containing the conversation audio files")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Pause duration between turns in seconds")
    parser.add_argument("--with_transcript", action="store_true",
                        help="Display transcript for each turn")
    args = parser.parse_args()
    
    # Initialize pygame for non-macOS platforms
    if platform.system() != "Darwin":
        pygame.init()
    
    # Check if the directory exists
    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} does not exist.")
        print("Make sure you've run context-aware-tts.py first to generate the conversation.")
        sys.exit(1)
    
    print(f"Playing conversation from: {args.dir}")
    
    # Define the transcripts for each turn
    transcripts = {
        1: {
            "speaker": "Assistant", 
            "text": "Hello! I'm an AI assistant. How are you doing today?"
        },
        2: {
            "speaker": "Human", 
            "text": "I'm doing well, thanks for asking! What can you tell me about voice synthesis?"
        },
        3: {
            "speaker": "Assistant", 
            "text": "Voice synthesis has made incredible progress in recent years. Modern models like CSM can clone voices with very little data and maintain the speaker's style and emotion. It's fascinating how we can now have these more natural conversations with AI systems!"
        },
        4: {
            "speaker": "Human (Clone)", 
            "text": "That's really interesting! I've been curious about how these systems actually work. Do they need a lot of training data, and how natural can they really sound compared to real human speech?"
        },
        5: {
            "speaker": "Assistant", 
            "text": "Great question! Traditional voice synthesis required hours of recordings from a single speaker, but modern AI models like CSM can produce remarkably natural speech with just a short sample. The key is that they're trained on diverse speakers and can generalize voice characteristics. While they're incredibly good now, human speech still has subtle nuances that AI is getting better at capturing. The gap is narrowing every year though!"
        }
    }
    
    # Determine which Turn 2 file exists (human or dummy)
    turn2_human_path = os.path.join(args.dir, "turn2_human.wav")
    turn2_dummy_path = os.path.join(args.dir, "turn2_human_dummy.wav")
    
    # Define the file sequence
    sequence = [
        {"file": os.path.join(args.dir, "turn1_assistant.wav"), "turn": 1},
        {"file": turn2_human_path if os.path.exists(turn2_human_path) else turn2_dummy_path, "turn": 2},
        {"file": os.path.join(args.dir, "turn3_assistant.wav"), "turn": 3},
        {"file": os.path.join(args.dir, "turn4_human_clone.wav"), "turn": 4},
        {"file": os.path.join(args.dir, "turn5_assistant_final.wav"), "turn": 5},
    ]
    
    # Print intro message
    print("\n" + "="*50)
    print(" CONVERSATION PLAYBACK".center(50))
    print("="*50)
    print(f"Playing a 5-turn conversation from: {args.dir}")
    print(f"Pause between turns: {args.pause} seconds")
    print("Press Ctrl+C at any time to stop playback")
    print("="*50 + "\n")
    
    # Play each file in the sequence
    try:
        for i, item in enumerate(sequence):
            turn_number = item["turn"]
            file_path = item["file"]
            
            # Display transcript if requested
            if args.with_transcript and turn_number in transcripts:
                display_transcript(
                    turn_number, 
                    transcripts[turn_number]["speaker"],
                    transcripts[turn_number]["text"]
                )
            
            # Play the audio file
            play_audio_file(file_path)
            
            # Pause between turns (except after the last one)
            if i < len(sequence) - 1:
                time.sleep(args.pause)
                
        print("\nConversation playback complete!")
        
    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
        # Clean up pygame if needed
        if platform.system() != "Darwin":
            pygame.mixer.quit()
            pygame.quit()
    
    print("\nThank you for listening!")

if __name__ == "__main__":
    main() 