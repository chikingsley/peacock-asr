from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model
import numpy as np
import yaml
import os
import math
import torch

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = f"{config['run_name']}-lora"  # Add lora suffix to run name
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = float(config["learning_rate"])
scheduler_type = config.get("scheduler_type", "constant")  # Default to constant if not specified

print(f"Using learning rate scheduler: {scheduler_type}")

# LoRA parameters
lora_rank = 32
lora_alpha = 64
lora_dropout = 0.0

# Determine log directory for tensorboard
log_dir = os.path.join(f"./{base_repo_id}", "logs", run_name)
os.makedirs(log_dir, exist_ok=True)
print(f"Tensorboard logs will be saved to: {log_dir}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

# Define LoRA configuration
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    lora_dropout=lora_dropout,
    modules_to_save=["lm_head", "embed_tokens"],
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True
)

print(f"Preparing model for LoRA training (rank={lora_rank}, alpha={lora_alpha})...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load the dataset
full_ds = load_dataset(dsn, split="train") 
print(f"Dataset {config['TTS_dataset']} loaded: {full_ds}")

# Create an evaluation split (10% of data, min 1 row, max 25 rows)
total_samples = len(full_ds)
eval_size = max(1, min(25, math.ceil(total_samples * 0.1)))
train_size = total_samples - eval_size

print(f"Splitting dataset: {train_size} training samples, {eval_size} evaluation samples")

# Split the dataset into training and evaluation sets
split_dataset = full_ds.train_test_split(test_size=eval_size, seed=42)
train_ds = split_dataset["train"]
eval_ds = split_dataset["test"]

print(f"Training dataset: {train_ds}")
print(f"Evaluation dataset: {eval_ds}")

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_dir=log_dir,  # Directory for storing logs
    logging_strategy="steps",
    logging_steps=1,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=0.25,
    save_strategy="steps",
    save_steps=save_steps,
    bf16=True,
    output_dir=f"./{base_repo_id}/lora",
    report_to="tensorboard",
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type=scheduler_type,  # Use the scheduler type from config
    run_name=run_name,  # Name for the run in tensorboard
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

print("Starting LoRA training...")
trainer.train()

# Create output directory
output_dir = f"./output/{model_name.split('/')[-1]}-lora-ft"
os.makedirs(output_dir, exist_ok=True)

print("Merging LoRA weights with base model...")
# Get the base model from the PEFT model
merged_model = model.merge_and_unload()

# Save the merged model
print(f"Saving merged model to {output_dir}")
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
print(f"To view tensorboard logs, run: tensorboard --logdir={log_dir}") 