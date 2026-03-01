from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import os
import math

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
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

# Determine log directory for tensorboard
log_dir = os.path.join(f"./{base_repo_id}", "logs", run_name)
os.makedirs(log_dir, exist_ok=True)
print(f"Tensorboard logs will be saved to: {log_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

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
    output_dir=f"./{base_repo_id}",
    report_to="tensorboard",
    remove_unused_columns=True, 
    learning_rate=learning_rate,
    lr_scheduler_type=scheduler_type,  # Use the scheduler type from config
    run_name=run_name,    # Name for the run in tensorboard
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

trainer.train()

import os
os.makedirs("./output", exist_ok=True)

# Save the model and tokenizer
model.save_pretrained(f"./output/{model_name.split('/')[-1]}-ft")
tokenizer.save_pretrained(f"./output/{model_name.split('/')[-1]}-ft")

print(f"Model and tokenizer saved to ./output/{model_name.split('/')[-1]}-ft")
print(f"To view tensorboard logs, run: tensorboard --logdir={log_dir}")
