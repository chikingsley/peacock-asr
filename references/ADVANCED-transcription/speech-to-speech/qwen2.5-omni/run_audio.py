import os
import soundfile as sf
import time
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# Set environment variable for faster downloads
os.environ["HF_TRANSFER"] = "1"

def main():
    # Load model and processor
    model = Qwen2_5OmniModel.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype="auto",
        device_map="auto",
        enable_audio_output=True,
        attn_implementation="flash_attention_2", # not working unless in fp32 for now.
    )
    
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    # Example conversation with audio input
    conversation = [
        {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "input.wav"},  # Replace with your audio file path
            ],
        },
    ]

    # Process inputs
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Measure generation time
    start_time = time.time()
    
    # Generate response
    text_ids, audio = model.generate(**inputs, return_audio=True) # female voice default
    # text_ids, audio = model.generate(**inputs, return_audio=True, spk="Ethan") # male voice
    
    # Calculate and print generation time
    generation_time = time.time() - start_time
    print(f"\nGeneration time: {generation_time:.2f} seconds")

    # Decode text response
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Text Response:", text)

    # Save audio response
    sf.write(
        "audio_output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
    print("Audio response saved to audio_output.wav")

if __name__ == "__main__":
    main() 