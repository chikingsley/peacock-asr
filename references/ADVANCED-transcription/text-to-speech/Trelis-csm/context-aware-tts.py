from generator import Segment, load_csm_1b
import torchaudio
import torch
import os
import argparse
import time
import wave
import pyaudio
import numpy as np
from datetime import datetime

def record_audio(output_file, sample_rate=16000, channels=1, chunk=1024, format=pyaudio.paInt16):
    """Record audio from microphone and save to file"""
    p = pyaudio.PyAudio()
    
    print("\n=== RECORDING INSTRUCTIONS ===")
    print("Press ENTER to start recording")
    print("Press Ctrl+C to stop recording")
    print("================================\n")
    
    input("Press ENTER when ready to start recording...")
    
    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    print("\n🔴 Recording... (Press Ctrl+C to stop)")
    print("Speak clearly into your microphone")
    
    frames = []
    
    try:
        while True:
            data = stream.read(chunk)
            frames.append(data)
            # Print a dot every second to show it's recording
            if len(frames) % (sample_rate // chunk) == 0:
                print(".", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n✅ Recording stopped")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded audio to WAV file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Audio saved to {output_file}")
    
    # Convert wave to tensor for CSM
    audio_tensor, _ = torchaudio.load(output_file)
    return audio_tensor.squeeze(0)

def main():
    parser = argparse.ArgumentParser(description="Multi-turn conversation with CSM")
    parser.add_argument("--output_dir", type=str, default="data/conversation", 
                        help="Directory to store conversation outputs")
    parser.add_argument("--human_recording", type=str, default=None,
                        help="Path to human recording (if already recorded)")
    parser.add_argument("--no_recording", action="store_true",
                        help="Skip recording and use a dummy response")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the generator
    print("Loading CSM model...")
    generator = load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded successfully!")
    
    # Keep track of all segments for context
    conversation_history = []
    
    # Define speaker IDs - keep these consistent!
    ASSISTANT_SPEAKER_ID = 0  # CSM's default voice
    HUMAN_SPEAKER_ID = 1      # Your cloned voice
    
    # Turn 1: CSM's first response
    turn1_text = "Hello! I'm an AI assistant. How are you doing today?"
    
    print(f"\n[CSM TURN 1 (Assistant)]: {turn1_text}")
    print("Generating audio for turn 1...")
    
    turn1_audio = generator.generate(
        text=turn1_text,
        speaker=ASSISTANT_SPEAKER_ID,
        context=[],  # No context for first turn
        max_audio_length_ms=10_000,
    )
    
    turn1_path = os.path.join(args.output_dir, "turn1_assistant.wav")
    torchaudio.save(turn1_path, turn1_audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {turn1_path}")
    
    # Create segment for Turn 1 and add to history
    turn1_segment = Segment(
        text=turn1_text,
        speaker=ASSISTANT_SPEAKER_ID,
        audio=turn1_audio
    )
    conversation_history.append(turn1_segment)
    
    # Play the audio (optional)
    try:
        print("\nPlaying CSM's response...")
        os.system(f"afplay {turn1_path}")  # macOS
    except:
        print("Could not play audio. Please play it manually.")
    
    # Turn 2: Human's response
    turn2_text = "I'm doing well, thanks for asking! What can you tell me about voice synthesis?"
    
    if args.human_recording:
        # Use provided recording
        human_file = args.human_recording
        print(f"\nUsing provided recording: {human_file}")
        
        if not os.path.exists(human_file):
            print(f"ERROR: File {human_file} not found. Using dummy response instead.")
            args.no_recording = True
    
    elif args.no_recording:
        human_file = None
        print("\nSkipping recording as requested. Using dummy response.")
    
    else:
        # Prompt what to say and record
        print("\n-----------------------------------------")
        print("Now it's your turn to respond!")
        print("\n📝 Please read the following text when recording starts:")
        print(f"\n\"{turn2_text}\"\n")
        
        # Record audio
        human_file = os.path.join(args.output_dir, "turn2_human.wav")
        try:
            human_audio = record_audio(human_file, sample_rate=generator.sample_rate)
        except Exception as e:
            print(f"Error recording audio: {e}")
            print("Using dummy response instead.")
            human_file = None
            args.no_recording = True
    
    if args.no_recording or not human_file:
        # Generate a dummy human response using CSM's default voice
        print("Generating dummy human response...")
        turn2_audio = generator.generate(
            text=turn2_text,
            speaker=HUMAN_SPEAKER_ID,  # Use human speaker ID but without specific voice characteristics
            context=[],  # No context for generic voice
            max_audio_length_ms=10_000,
        )
        turn2_path = os.path.join(args.output_dir, "turn2_human_dummy.wav")
        torchaudio.save(turn2_path, turn2_audio.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"Dummy audio saved to {turn2_path}")
        
        # Create Turn 2 segment with dummy audio
        turn2_segment = Segment(
            text=turn2_text,
            speaker=HUMAN_SPEAKER_ID,
            audio=turn2_audio
        )
    else:
        # Load the human recording
        print(f"Loading human recording from {human_file}...")
        human_audio, sample_rate = torchaudio.load(human_file)
        
        # Resample if needed
        if sample_rate != generator.sample_rate:
            human_audio = torchaudio.functional.resample(
                human_audio.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
            )
        else:
            human_audio = human_audio.squeeze(0)
        
        # Create turn 2 segment with the human recording
        turn2_segment = Segment(
            text=turn2_text,
            speaker=HUMAN_SPEAKER_ID,
            audio=human_audio
        )
        
        print(f"Using transcript: \"{turn2_text}\"")
    
    # Add Turn 2 to conversation history
    conversation_history.append(turn2_segment)
    
    # Turn 3: CSM's response with context (using assistant voice)
    turn3_text = "Voice synthesis has made incredible progress in recent years. Modern models like CSM can clone voices with very little data and maintain the speaker's style and emotion. It's fascinating how we can now have these more natural conversations with AI systems!"
    
    print(f"\n[CSM TURN 3 (Assistant)]: {turn3_text}")
    print("Generating audio for turn 3 using assistant voice...")
    
    turn3_audio = generator.generate(
        text=turn3_text,
        speaker=ASSISTANT_SPEAKER_ID,  # Using the assistant speaker ID
        context=conversation_history,  # Using all previous conversation
        max_audio_length_ms=20_000,
    )
    
    turn3_path = os.path.join(args.output_dir, "turn3_assistant.wav")
    torchaudio.save(turn3_path, turn3_audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {turn3_path}")
    
    # Create segment for Turn 3 and add to history
    turn3_segment = Segment(
        text=turn3_text,
        speaker=ASSISTANT_SPEAKER_ID,
        audio=turn3_audio
    )
    conversation_history.append(turn3_segment)
    
    # Play the audio (optional)
    try:
        print("\nPlaying CSM's response...")
        os.system(f"afplay {turn3_path}")  # macOS
    except:
        print("Could not play audio. Please play it manually.")
    
    # Turn 4: Generate a response from CSM that simulates the human's next turn
    print("\n-----------------------------------------")
    print("Adding Turn 4: CSM generating your next response (voice cloning)...")
    
    turn4_text = "That's really interesting! I've been curious about how these systems actually work. Do they need a lot of training data, and how natural can they really sound compared to real human speech?"
    
    print(f"\n[CSM TURN 4 (Human voice clone)]: {turn4_text}")
    print("Generating audio that simulates your voice...")
    
    # Generate Turn 4 audio using the previous context to maintain voice characteristics
    turn4_audio = generator.generate(
        text=turn4_text,
        speaker=HUMAN_SPEAKER_ID,  # Using the human speaker ID
        context=conversation_history,  # Using all previous conversation for better voice cloning
        max_audio_length_ms=20_000,
    )
    
    turn4_path = os.path.join(args.output_dir, "turn4_human_clone.wav")
    torchaudio.save(turn4_path, turn4_audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {turn4_path}")
    
    # Create segment for Turn 4 and add to history
    turn4_segment = Segment(
        text=turn4_text,
        speaker=HUMAN_SPEAKER_ID,  # This is a cloned human voice
        audio=turn4_audio
    )
    conversation_history.append(turn4_segment)
    
    # Play the audio (optional)
    try:
        print("\nPlaying CSM's simulated human response...")
        os.system(f"afplay {turn4_path}")  # macOS
    except:
        print("Could not play audio. Please play it manually.")
    
    # Turn 5: Final CSM response to complete the conversation
    turn5_text = "Great question! Traditional voice synthesis required hours of recordings from a single speaker, but modern AI models like CSM can produce remarkably natural speech with just a short sample. The key is that they're trained on diverse speakers and can generalize voice characteristics. While they're incredibly good now, human speech still has subtle nuances that AI is getting better at capturing. The gap is narrowing every year though!"
    
    print(f"\n[CSM TURN 5 (Assistant)]: {turn5_text}")
    print("Generating final assistant response...")
    
    # Generate Turn 5 audio with all previous context
    turn5_audio = generator.generate(
        text=turn5_text,
        speaker=ASSISTANT_SPEAKER_ID,  # Back to assistant speaker ID
        context=conversation_history,  # Full conversation context
        max_audio_length_ms=30_000,
    )
    
    turn5_path = os.path.join(args.output_dir, "turn5_assistant_final.wav")
    torchaudio.save(turn5_path, turn5_audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {turn5_path}")
    
    # Create segment for Turn 5 and add to history (if continuing the conversation)
    turn5_segment = Segment(
        text=turn5_text,
        speaker=ASSISTANT_SPEAKER_ID,
        audio=turn5_audio
    )
    conversation_history.append(turn5_segment)
    
    # Play the audio (optional)
    try:
        print("\nPlaying CSM's final response...")
        os.system(f"afplay {turn5_path}")  # macOS
    except:
        print("Could not play audio. Please play it manually.")
    
    print("\nConversation complete! The audio files are saved in:", args.output_dir)
    print("- Turn 1 (Assistant): turn1_assistant.wav")
    if args.no_recording:
        print("- Turn 2 (Human/Dummy): turn2_human_dummy.wav")
    else:
        print("- Turn 2 (Human): turn2_human.wav")
    print("- Turn 3 (Assistant): turn3_assistant.wav")
    print("- Turn 4 (Human/Clone): turn4_human_clone.wav")
    print("- Turn 5 (Assistant): turn5_assistant_final.wav")
    
    print("\nSpeaker IDs used:")
    print(f"- Assistant: {ASSISTANT_SPEAKER_ID}")
    print(f"- Human: {HUMAN_SPEAKER_ID}")

if __name__ == "__main__":
    main() 