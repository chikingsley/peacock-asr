#!/usr/bin/env python3
# /// script
# dependencies = [
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "mistral-common[audio]>=1.8.1",
#   "torch>=2.0.0",
#   "librosa>=0.11.0",
#   "huggingface-hub>=0.19.0",
# ]
# ///
"""
Voxtral Demo: Audio Instruct vs Transcription
===========================================

This script demonstrates the key differences between Voxtral's two main approaches:
1. Audio Instruct: Conversational AI with multi-modal inputs (audio + text)
2. Transcription: Direct speech-to-text conversion

We'll show the tokenized inputs and expected outputs to illustrate how each approach works.
"""

from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

print("=" * 80)
print("VOXTRAL DEMO: Audio Instruct vs Transcription")
print("=" * 80)
print(f"Device: {device}")
print(f"Model: {repo_id}")
print()

# Initialize processor (we'll actually use it to show real templates)
print("# Initialize processor")
print("processor = AutoProcessor.from_pretrained(repo_id)")
print()
try:
    processor = AutoProcessor.from_pretrained(repo_id)
    print("✅ Processor loaded successfully")
except Exception as e:
    print(f"❌ Could not load processor: {e}")
    print("Will show conceptual templates instead")
    processor = None
print()

print("=" * 80)
print("APPROACH 1: AUDIO INSTRUCT (Conversational AI)")
print("=" * 80)
print("Purpose: Multi-modal conversation with reasoning about audio content")
print("Use case: Chat about audio, answer questions, analyze multiple audio files")
print()

# Audio Instruct Example
print("## Sample Conversation Structure:")
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3",
            },
            {
                "type": "audio", 
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
            },
            {"type": "text", "text": "What sport and what nursery rhyme are referenced?"},
        ],
    }
]

print("conversation = [")
print("    {")
print("        'role': 'user',")
print("        'content': [")
print("            {'type': 'audio', 'path': 'mary_had_lamb.mp3'},")
print("            {'type': 'audio', 'path': 'winning_call.mp3'},")
print("            {'type': 'text', 'text': 'What sport and what nursery rhyme are referenced?'},")
print("        ],")
print("    }")
print("]")
print()

print("## Template Application:")
print("inputs = processor.apply_chat_template(conversation)")
print()

if processor:
    try:
        # Apply the actual chat template
        inputs = processor.apply_chat_template(conversation)
        print("## ACTUAL Chat Template Tokens:")
        print("Input IDs shape:", inputs.input_ids.shape if hasattr(inputs, 'input_ids') else "N/A")
        if hasattr(inputs, 'input_ids'):
            tokens = inputs.input_ids[0].tolist()
            total_tokens = len(tokens)
            
            # Show first 20 tokens
            print("First 20 tokens:")
            first_tokens = tokens[:20]
            first_decoded = [processor.tokenizer.decode([token]) for token in first_tokens]
            for i, (token_id, token_text) in enumerate(zip(first_tokens, first_decoded)):
                print(f"  [{i:2d}] {token_id:5d} -> '{token_text}'")
            
            # Find where audio tokens (ID 24) end and show what comes next
            audio_end_idx = None
            for i, token_id in enumerate(tokens):
                if token_id == 24:  # [AUDIO] token
                    continue
                else:
                    # Found first non-audio token after audio sequence
                    if i > 0 and tokens[i-1] == 24:
                        audio_end_idx = i
                        break
            
            if audio_end_idx is not None:
                print(f"  ... [audio tokens] ...")
                print("4 tokens before and 6 tokens after audio ends:")
                pre_start = max(0, audio_end_idx - 4)
                context_tokens = tokens[pre_start:audio_end_idx+6]
                context_decoded = [processor.tokenizer.decode([token]) for token in context_tokens]
                for i, (token_id, token_text) in enumerate(zip(context_tokens, context_decoded)):
                    actual_index = pre_start + i
                    marker = " <-- AUDIO ENDS" if actual_index == audio_end_idx else ""
                    print(f"  [{actual_index:2d}] {token_id:5d} -> '{token_text}'{marker}")
            
            # Find [/INST] token (ID 4) and show context around it
            inst_end_idx = None
            for i, token_id in enumerate(tokens):
                if token_id == 4:  # [/INST] token
                    inst_end_idx = i
                    break
                    
            if inst_end_idx is not None:
                print("4 tokens before and 2 tokens after [/INST]:")
                pre_start = max(0, inst_end_idx - 4)
                context_tokens = tokens[pre_start:inst_end_idx+3]
                context_decoded = [processor.tokenizer.decode([token]) for token in context_tokens]
                for i, (token_id, token_text) in enumerate(zip(context_tokens, context_decoded)):
                    actual_index = pre_start + i
                    marker = " <-- [/INST]" if actual_index == inst_end_idx else ""
                    print(f"  [{actual_index:2d}] {token_id:5d} -> '{token_text}'{marker}")
            
            # Only show "last tokens" if they're different from what we just showed
            if (audio_end_idx is None and inst_end_idx is None) or total_tokens > max(audio_end_idx or 0, inst_end_idx or 0) + 16:
                print(f"  ... [{total_tokens - 30:,} tokens omitted] ...")
                print("Last 10 tokens:")
                last_tokens = tokens[-10:]
                last_decoded = [processor.tokenizer.decode([token]) for token in last_tokens]
                for i, (token_id, token_text) in enumerate(zip(last_tokens, last_decoded)):
                    actual_index = total_tokens - 10 + i
                    print(f"  [{actual_index:2d}] {token_id:5d} -> '{token_text}'")
        print()
    except Exception as e:
        print(f"❌ Could not apply chat template: {e}")
        print("## Conceptual Input Structure:")
        print("< audio_start > [audio_tokens_mary_had_lamb] < audio_end >")
        print("< audio_start > [audio_tokens_winning_call] < audio_end >")
        print("< text > What sport and what nursery rhyme are referenced? < /text >")
        print("< assistant >")
        print()
else:
    print("## Conceptual Input Structure:")
    print("< audio_start > [audio_tokens_mary_had_lamb] < audio_end >")
    print("< audio_start > [audio_tokens_winning_call] < audio_end >")
    print("< text > What sport and what nursery rhyme are referenced? < /text >")
    print("< assistant >")
    print()



print("=" * 80)
print("APPROACH 2: TRANSCRIPTION (Speech-to-Text)")
print("=" * 80)
print("Purpose: Direct conversion of speech to text")
print("Use case: Transcribe meetings, lectures, audio files")
print()

print("## Transcription Request Structure (Processor Method):")
print("inputs = processor.apply_transcrition_request(  # Note: typo in actual method name!")
print("    language='en',")
print("    audio='https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3',")
print("    model_id=repo_id")
print(")")
print()

print("## Alternative: Direct mistral_common Approach:")
print("from mistral_common.protocol.transcription.request import TranscriptionRequest")
print("from mistral_common.protocol.instruct.messages import RawAudio")
print("from mistral_common.audio import Audio")
print("from huggingface_hub import hf_hub_download")
print()
print("obama_file = hf_hub_download('patrickvonplaten/audio_samples', 'obama.mp3', repo_type='dataset')")
print("audio = Audio.from_file(obama_file, strict=False)")
print("audio = RawAudio.from_audio(audio)")
print("req = TranscriptionRequest(model=model, audio=audio, language='en', temperature=0.0)")
print()

if processor:
    # Try to find the correct transcription method
    transcription_methods = [
        'apply_transcrition_request',  # Note: actual method has typo!
        'apply_transcription_template', 
        'apply_transcription_request',
        'apply_audio_template',
        'process_transcription'
    ]
    
    transcription_worked = False
    for method_name in transcription_methods:
        if hasattr(processor, method_name):
            try:
                print(f"## Trying transcription method: {method_name}")
                method = getattr(processor, method_name)
                if method_name == 'apply_transcription_template':
                    inputs = method(
                        audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
                        language="en"
                    )
                elif method_name == 'apply_transcrition_request':
                    inputs = method(
                        language="en",
                        audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
                        model_id=repo_id
                    )
                else:
                    inputs = method(
                        language="en",
                        audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
                        model_id=repo_id
                    )
                print("## ACTUAL Transcription Template Tokens:")
                print("Input IDs shape:", inputs.input_ids.shape if hasattr(inputs, 'input_ids') else "N/A")
                if hasattr(inputs, 'input_ids'):
                    tokens = inputs.input_ids[0].tolist()
                    total_tokens = len(tokens)
                    
                    # Show first 20 tokens
                    print("First 20 tokens:")
                    first_tokens = tokens[:20]
                    first_decoded = [processor.tokenizer.decode([token]) for token in first_tokens]
                    for i, (token_id, token_text) in enumerate(zip(first_tokens, first_decoded)):
                        print(f"  [{i:2d}] {token_id:5d} -> '{token_text}'")
                    
                    # Find where audio tokens (ID 24) end and show what comes next
                    audio_end_idx = None
                    for i, token_id in enumerate(tokens):
                        if token_id == 24:  # [AUDIO] token
                            continue
                        else:
                            # Found first non-audio token after audio sequence
                            if i > 0 and tokens[i-1] == 24:
                                audio_end_idx = i
                                break
                    
                    if audio_end_idx is not None:
                        print(f"  ... [audio tokens] ...")
                        print("4 tokens before and 6 tokens after audio ends:")
                        pre_start = max(0, audio_end_idx - 4)
                        context_tokens = tokens[pre_start:audio_end_idx+6]
                        context_decoded = [processor.tokenizer.decode([token]) for token in context_tokens]
                        for i, (token_id, token_text) in enumerate(zip(context_tokens, context_decoded)):
                            actual_index = pre_start + i
                            marker = " <-- AUDIO ENDS" if actual_index == audio_end_idx else ""
                            print(f"  [{actual_index:2d}] {token_id:5d} -> '{token_text}'{marker}")
                    
                    # Find [/INST] token (ID 4) and show context around it
                    inst_end_idx = None
                    for i, token_id in enumerate(tokens):
                        if token_id == 4:  # [/INST] token
                            inst_end_idx = i
                            break
                            
                    # if inst_end_idx is not None:
                    #     print("4 tokens before and 2 tokens after [/INST]:")
                    #     pre_start = max(0, inst_end_idx - 4)
                    #     context_tokens = tokens[pre_start:inst_end_idx+3]
                    #     context_decoded = [processor.tokenizer.decode([token]) for token in context_tokens]
                    #     for i, (token_id, token_text) in enumerate(zip(context_tokens, context_decoded)):
                    #         actual_index = pre_start + i
                    #         marker = " <-- [/INST]" if actual_index == inst_end_idx else ""
                    #         print(f"  [{actual_index:2d}] {token_id:5d} -> '{token_text}'{marker}")
                    
                    # Only show "last tokens" if they're different from what we just showed
                    if (audio_end_idx is None and inst_end_idx is None) or total_tokens > max(audio_end_idx or 0, inst_end_idx or 0) + 16:
                        print(f"  ... [{total_tokens - 30:,} tokens omitted] ...")
                        print("Last 10 tokens:")
                        last_tokens = tokens[-10:]
                        last_decoded = [processor.tokenizer.decode([token]) for token in last_tokens]
                        for i, (token_id, token_text) in enumerate(zip(last_tokens, last_decoded)):
                            actual_index = total_tokens - 10 + i
                            print(f"  [{actual_index:2d}] {token_id:5d} -> '{token_text}'")
                transcription_worked = True
                break
            except Exception as e:
                print(f"❌ Method {method_name} failed: {e}")
                continue
    
    if not transcription_worked:
        print("❌ Could not find working transcription method")
        print("Available methods:", [m for m in dir(processor) if not m.startswith('_')])
        print("## Conceptual Transcription Structure:")
        print("< transcribe > < lang_en > < audio_start > [audio_tokens_obama_speech] < audio_end >")
    print()
else:
    print("## Conceptual Transcription Structure:")
    print("< transcribe > < lang_en > < audio_start > [audio_tokens_obama_speech] < audio_end >")
    print()





print("=" * 80)
print("TEMPLATE ANALYSIS COMPLETE")
print("=" * 80) 