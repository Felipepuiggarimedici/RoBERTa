import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.notebook import tqdm
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
import transformers.trainer_utils as trainer_utils

# Fix tqdm in notebooks
trainer_utils.tqdm = tqdm

# Ensure working directory is script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Parse arguments
parser = argparse.ArgumentParser(description="Run one-fold MLM training with logging and saving.")
parser.add_argument("-nHeads", type=int, required=True)
parser.add_argument("-nLayers", type=int, required=True)
parser.add_argument("-mlmProb", type=float, required=True)
parser.add_argument("-batch_size", type=int, required=True)
parser.add_argument("-hidden_size", type=int, required=True)
parser.add_argument("-d_ff", type=int, required=True)
args = parser.parse_args()

# Assign variables
num_heads = args.nHeads
num_layers = args.nLayers
mlmProb = args.mlmProb
batch_size = args.batch_size
hidden_size = args.hidden_size
d_ff = args.d_ff

hp_tag = f"h{num_heads}.l{num_layers}.p{int(mlmProb*100):02d}.b{batch_size}.hs{hidden_size}.dff{d_ff}"
base_output = os.path.join("tmp_output", hp_tag)
fold_dir = os.path.join(base_output, "fold_5")
model_save_dir = os.path.join("models", hp_tag, "fold_5")
summary_path = "hyperparamData/loss_5fold_summary.csv"

# Reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("tokenizer")

# Load dataset
full_df = pd.read_csv("data/fullData/train_HLA_A0201.csv")
df = full_df[["peptide", "Length"]].copy()

# === Instead of StratifiedKFold, assign folds deterministically by index ===
df = df.reset_index(drop=True)
df["fold"] = df.index % 5  # Assign folds 0..4 in round robin

train_df = df[df["fold"] != 4].reset_index(drop=True)
val_df = df[df["fold"] == 4].reset_index(drop=True)

train_ds = Dataset.from_pandas(train_df[["peptide"]])
val_ds = Dataset.from_pandas(val_df[["peptide"]])

# Tokenize
def tokenize_fn(examples):
    return tokenizer(
        examples["peptide"],
        padding="max_length",
        truncation=True,
        max_length=150,
        return_special_tokens_mask=True,
    )

tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["peptide"])
tokenized_eval = val_ds.map(tokenize_fn, batched=True, remove_columns=["peptide"])

# Callbacks
class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if logs is None:
            return
        if "loss" in logs:
            self.records.append({"step": step, "type": "train", "loss": logs["loss"]})
        if "eval_loss" in logs:
            self.records.append({"step": step, "type": "eval", "loss": logs["eval_loss"]})

early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
loss_logger = LossLoggerCallback()

# Model configuration
config = RobertaConfig(
    vocab_size=len(tokenizer),
    hidden_size=hidden_size,
    intermediate_size=d_ff,
    max_position_embeddings=152,
    num_hidden_layers=num_layers,
    num_attention_heads=num_heads,
)

model = RobertaForMaskedLM(config)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=mlmProb)

# Training args
training_args = TrainingArguments(
    output_dir=fold_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    max_steps=150000,
    logging_steps=400,
    eval_strategy="steps",
    eval_steps=400,
    save_strategy="steps",
    save_steps=400,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    push_to_hub=False,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=collator,
    callbacks=[early_stopping, loss_logger],
)

trainer.train()

# Evaluate and save metrics
final_eval = trainer.evaluate()
best_loss = final_eval["eval_loss"]
last_train = [r for r in loss_logger.records if r["type"] == "train"][-1]

summary = [ {
    "fold": 5,
    "n_heads": num_heads,
    "mlm_prob": mlmProb,
    "n_layers": num_layers,
    "batch_size": batch_size,
    "hidden_size": hidden_size,
    "d_ff": d_ff,
    "last_train_step": last_train["step"],
    "last_train_loss": last_train["loss"],
    "best_eval_loss": best_loss,
} ]

# Save summary to CSV
os.makedirs("hyperparamData", exist_ok=True)
summary_df = pd.DataFrame(summary)
if os.path.exists(summary_path):
    pd.concat([pd.read_csv(summary_path), summary_df], ignore_index=True) \
      .to_csv(summary_path, index=False)
    print("Appended to existing loss_5fold_summary.csv")
else:
    summary_df.to_csv(summary_path, index=False)
    print("Created new loss_5fold_summary.csv")

# Save best model to permanent folder
def save_best_model(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    files_to_copy = ["pytorch_model.bin", "config.json", "training_args.bin"]
    for fname in files_to_copy:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest_dir, fname))
            print(f"Copied {fname} to {dest_dir}")
        else:
            print(f"Warning: {fname} not found in {src_dir}")

save_best_model(fold_dir, model_save_dir)

# Cleanup
print(f"\nAttempting to delete temporary output: {base_output}")
if os.path.exists(base_output):
    try:
        shutil.rmtree(base_output)
        print(f"Successfully deleted: {base_output}")
    except OSError as e:
        print(f"Error deleting {base_output}: {e}")
else:
    print(f"{base_output} does not exist. No cleanup needed.")