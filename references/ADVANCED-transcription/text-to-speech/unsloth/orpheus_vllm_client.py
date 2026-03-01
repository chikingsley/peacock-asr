#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "snac",
#     "soundfile",
#     "torch",
#     "transformers",
#     "openai",
#     "protobuf",
# ]
# ///
from __future__ import annotations
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time

import argparse, re, sys
import openai

# --- constants --------------------------------------------------------------
MODEL_NAME          = "unsloth/orpheus-3b-0.1-ft"
# MODEL_NAME          = "Trelis/orpheus-3b-0.1-ft-lora-ft_20250617_101017-merged"
AUDIO_SR            = 24_000                 # 24 kHz

# Special token IDs
START_TOKEN_ID = 128259  # Start of human
END_TEXT_TOKEN_ID = 128009  # End of text
END_HUMAN_TOKEN_ID = 128260  # End of human

# ---------------------------------------------------------------------------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft", use_fast=False)

def template_prompt(text: str, voice: str = "tara") -> str:
    """
    Tokenize the prompt, add special tokens, decode back to string.
    Returns the decoded string to send to the endpoint.
    """
    prompt_text = f"{voice}: {text}"
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors=None)
    # Prepend and append special tokens
    full_ids = [START_TOKEN_ID] + input_ids + [END_TEXT_TOKEN_ID, END_HUMAN_TOKEN_ID]
    prompt_str = tokenizer.decode(full_ids, skip_special_tokens=False)
    print(f"Prompt token IDs: {full_ids}")
    # print(f"Decoded prompt string: {prompt_str}")
    return prompt_str

def call_completions_endpoint(
    endpoint: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Call the OpenAI-compatible completions endpoint
    """
    # Configure the client to use the custom endpoint
    client = openai.OpenAI(base_url=endpoint, api_key="dummy_key")
    
    try:
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=repetition_penalty - 1.0,  # OpenAI API uses different scale
            stream=False,
            stop=["<custom_token_6>"]  # Stop at EOS token
        )
        
        print(response)
        return response.choices[0].text
            
    except Exception as e:
        print(f"Error calling completions endpoint: {e}")
        return ""

def main() -> None:
    p = argparse.ArgumentParser(description="Run Orpheus-3B via vLLM and decode to WAV.")
    p.add_argument("--endpoint", required=False, help="Base URL, e.g. https://xxx-8000.proxy.runpod.net", default="https://ak7njvfh86b3du-8000.proxy.runpod.net")
    p.add_argument("--prompt", required=True, help="Plain-text prompt to synthesise")
    p.add_argument("--voice", default="tara", help="Voice to use (default: tara)")
    p.add_argument("--out", default=None, help="Output .wav filename (24 kHz mono)")
    p.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    args = p.parse_args()

    # Set default output filename to voice name if not specified
    if args.out is None:
        args.out = f"{args.voice}.wav"

    # 1) Format the prompt with the chosen voice
    formatted_prompt = template_prompt(args.prompt, args.voice)
    print(f"Formatted prompt: {formatted_prompt}")
    
    # 2) Call the completions endpoint
    print("→ contacting endpoint…", file=sys.stderr)
    endpoint_url = args.endpoint.rstrip("/")
    if not endpoint_url.endswith("/v1"):
        endpoint_url = f"{endpoint_url}/v1"
        
    llm_start_time = time.time()
    response = call_completions_endpoint(
        endpoint=endpoint_url,
        prompt=formatted_prompt,
        max_new_tokens=args.max_tokens,
    )
    llm_end_time = time.time()
    print(f"\nLLM call took: {llm_end_time - llm_start_time:.2f}s")
    
    if isinstance(response, str):
        response_text = response
    elif hasattr(response, 'choices') and response.choices:
        response_text = response.choices[0].text
    else:
        response_text = ""
    
    if response_text:
        postproc_start_time = time.time()
        custom_tokens = re.findall(r'<custom_token_\d+>', response_text)
        custom_token_ids = [tokenizer.encode(token, add_special_tokens=False, return_tensors=None)[0] for token in custom_tokens]
        print("Custom Token IDs:")
        print(custom_token_ids)

        # --- Canonical Orpheus/SNAC post-processing block ---
        import torch
        import soundfile as sf
        from snac import SNAC

        device= "cpu"
        # # Check for MPS availability and set the device
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        #     print("\nUsing MPS device for SNAC decoding.")
        # else:
        #     device = torch.device("cpu")
        #     print("\nUsing CPU for SNAC decoding.")
        
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

        generated_ids = torch.tensor([custom_token_ids])

        token_to_find = 128257
        token_to_remove = 128258

        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_ids

        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t.item() - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        def redistribute_codes(code_list):
            layer_1, layer_2, layer_3 = [], [], []
            for i in range(len(code_list)//7):
                layer_1.append(code_list[7*i])
                layer_2.append(code_list[7*i+1]-4096)
                layer_3.append(code_list[7*i+2]-(2*4096))
                layer_3.append(code_list[7*i+3]-(3*4096))
                layer_2.append(code_list[7*i+4]-(4*4096))
                layer_3.append(code_list[7*i+5]-(5*4096))
                layer_3.append(code_list[7*i+6]-(6*4096))
            
            codes = [torch.tensor(layer_1).unsqueeze(0).to(device),
                     torch.tensor(layer_2).unsqueeze(0).to(device),
                     torch.tensor(layer_3).unsqueeze(0).to(device)]
            
            with torch.no_grad():
                audio_hat = snac_model.decode(codes)
            return audio_hat

        if code_lists:
            samples = redistribute_codes(code_lists[0])
            sf.write(args.out, samples.detach().squeeze().to("cpu").numpy(), 24000)
            print(f"\nAudio successfully written to {args.out}")
        else:
            print("\nCould not generate audio, no codes to process.")

        postproc_end_time = time.time()
        print(f"Post-processing and decoding took: {postproc_end_time - postproc_start_time:.2f}s")
    
    sys.exit()

if __name__ == "__main__":
    main()