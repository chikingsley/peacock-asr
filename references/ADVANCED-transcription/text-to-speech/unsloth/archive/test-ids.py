# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers",
# ]
# ///
import re
from transformers import AutoTokenizer

MODEL_NAME = "canopylabs/orpheus-3b-0.1-ft"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Example API output string (replace with your actual output)
api_text = """
<custom_token_26441><custom_token_2><custom_token_6><custom_token_3>"""

# 1. Extract all <custom_token_...> tokens
custom_tokens = re.findall(r"<custom_token_\d+>", api_text)

# 2. Encode each token to get its ID (should yield one ID per token)
token_ids = [tokenizer.encode(tok, add_special_tokens=False)[0] for tok in custom_tokens]

print("Extracted custom tokens:", custom_tokens)
print("Extracted token IDs:", token_ids)
