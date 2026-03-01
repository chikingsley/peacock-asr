#!/usr/bin/env python3
"""
Test the corrected token comparison logic.
Run this after loading model, processor, and running a forward pass.
"""

def test_corrected_token_offset(batch_output, forward_result, processor):
    """Test the corrected next-token prediction comparison."""

    print("=== TESTING CORRECTED TOKEN COMPARISON ===")

    # Get predictions and inputs
    predicted_tokens = forward_result.logits.argmax(dim=-1)[0]  # [seq_len]
    input_tokens = batch_output['input_ids'][0, :, 0]  # [seq_len] - text dimension
    labels = batch_output['labels'][0]  # [seq_len]

    pad_token_id = 3
    ignore_index = -100

    print(f"Sequence length: {len(predicted_tokens)}")
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Labels shape: {labels.shape}")

    # Method 1: Compare prediction[i] with input[i+1] (next token prediction)
    print("\\n=== METHOD 1: Prediction[i] vs Input[i+1] ===")
    correct_next = 0
    total_next = 0

    for i in range(len(predicted_tokens) - 1):
        if input_tokens[i] == pad_token_id or input_tokens[i+1] == pad_token_id:
            continue

        pred_token = predicted_tokens[i].item()
        target_token = input_tokens[i+1].item()
        current_token = input_tokens[i].item()

        if pred_token == target_token:
            correct_next += 1
        total_next += 1

        if total_next <= 10:  # Show first 10
            pred_text = processor.tokenizer.decode([pred_token])
            target_text = processor.tokenizer.decode([target_token])
            current_text = processor.tokenizer.decode([current_token])
            match = "✓" if pred_token == target_token else "✗"
            print(f"  {i:3d}: {match} '{current_text:8s}' → pred:'{pred_text:8s}' target:'{target_text:8s}'")

    next_accuracy = correct_next / total_next if total_next > 0 else 0
    print(f"Next-token accuracy: {correct_next}/{total_next} = {next_accuracy:.1%}")

    # Method 2: Compare with labels (if available)
    print("\\n=== METHOD 2: Prediction vs Labels ===")
    correct_labels = 0
    total_labels = 0

    for i in range(len(predicted_tokens)):
        if labels[i] == ignore_index:
            continue

        pred_token = predicted_tokens[i].item()
        label_token = labels[i].item()

        if pred_token == label_token:
            correct_labels += 1
        total_labels += 1

        if total_labels <= 10:  # Show first 10
            pred_text = processor.tokenizer.decode([pred_token])
            label_text = processor.tokenizer.decode([label_token])
            match = "✓" if pred_token == label_token else "✗"
            print(f"  {i:3d}: {match} pred:'{pred_text:8s}' label:'{label_text:8s}'")

    label_accuracy = correct_labels / total_labels if total_labels > 0 else 0
    print(f"Label accuracy: {correct_labels}/{total_labels} = {label_accuracy:.1%}")

    print("\\n=== ANALYSIS ===")
    if next_accuracy > 0.8:
        print("✅ Next-token prediction looks good! The model is learning correctly.")
    elif label_accuracy > 0.8:
        print("✅ Label prediction looks good! The data collator is setting up labels correctly.")
    else:
        print("❌ Both methods show low accuracy. There may be other alignment issues.")

    return {
        'next_token_accuracy': next_accuracy,
        'label_accuracy': label_accuracy,
        'total_predictions': total_next,
        'total_labels': total_labels
    }

# Usage in notebook:
# results = test_corrected_token_offset(batch_output, forward_result, processor)