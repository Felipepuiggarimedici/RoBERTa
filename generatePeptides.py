"""
Peptide generation with Gibbs sampling over a masked-LM (RoBERTa).
Appends generated peptides by length into generatedPeptides/HLAA0201/generated_peptides_by_length.csv
Reference:
Wang, J., Shen, T., Xie, T., & Zhao, Y. (2019). BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model.
"""

import os
import math
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForMaskedLM
#very important for running in HPC
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ------------------------------
# Gibbs sampling generation
# ------------------------------
def generate_peptides(model, tokenizer, length, num_peptides, gibbs_steps, device, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    model.eval()
    CLS_ID, SEP_ID, MASK_ID = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids)
    allowed_ids = [i for i in range(tokenizer.vocab_size) if i not in special_ids]
    special_idx_tensor = torch.tensor(list(special_ids), dtype=torch.long, device=device)

    seq_len = length + 2  # CLS + peptide + SEP
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    generated = []

    for _ in tqdm(range(num_peptides), desc=f"Length {length}"):
        peptide_tokens = [random.choice(allowed_ids) for _ in range(length)]
        input_ids = torch.tensor([[CLS_ID] + peptide_tokens + [SEP_ID]], dtype=torch.long, device=device)

        for _ in range(gibbs_steps):
            pos = random.randint(1, length)
            input_ids[0, pos] = MASK_ID

            with torch.no_grad():
                logits = model(input_ids, attention_mask=attention_mask).logits[0, pos]

            logits.index_fill_(0, special_idx_tensor, float("-1e9"))

            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() == 0:
                new_id = random.choice(allowed_ids)
            else:
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[allowed_ids] = True
                probs = probs * mask.float()
                probs /= probs.sum()
                new_id = torch.multinomial(probs, 1).item()

            input_ids[0, pos] = new_id

        pep = tokenizer.decode(input_ids[0, 1:-1], skip_special_tokens=True).replace(" ", "")
        generated.append(pep)

    return generated


def allocate_by_length(props, total):
    alloc = {L: int(math.floor(p * total)) for L, p in props.items()}
    rem = total - sum(alloc.values())
    if rem > 0:
        frac = {L: (p * total) - alloc[L] for L, p in props.items()}
        for L in sorted(frac, key=frac.get, reverse=True)[:rem]:
            alloc[L] += 1
    return alloc


# ------------------------------
# Parameters
# ------------------------------
train_file = "data/fullData/train_HLA_A0201.csv"
tok_path = "tokenizer"
model_path = "modelsPeptideOnly/HLA_HLA-A_02-01"
output_dir = "generatedPeptides/HLAA0201"
output_file = os.path.join(output_dir, "generated_peptides_by_length.csv")

total_peptides = 1000
gibbs_steps = 50
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Load model & tokenizer
# ------------------------------
print(f"Loading tokenizer from {tok_path} and model from {model_path}")
tokenizer = RobertaTokenizer.from_pretrained(tok_path, local_files_only=True)
model = RobertaForMaskedLM.from_pretrained(model_path, local_files_only=True).to(device)

# ------------------------------
# Read training data to get length proportions
# ------------------------------
df = pd.read_csv(train_file)
if "Length" not in df.columns:
    raise ValueError("Train CSV must have 'Length' column")

length_props = (df["Length"].value_counts() / len(df)).sort_index()
allocations = allocate_by_length(length_props, total_peptides)

# ------------------------------
# Check how many peptides already exist
# ------------------------------
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    already_generated = len(existing_df)
else:
    already_generated = 0

print(f"Peptides already generated: {already_generated}, generating {total_peptides} more...")

# ------------------------------
# Generate
# ------------------------------
all_peptides = []
for L in sorted(allocations):
    count_to_generate = allocations[L]
    if count_to_generate > 0:
        peps = generate_peptides(
            model,
            tokenizer,
            L,
            count_to_generate,
            gibbs_steps,
            device,
            seed=seed + already_generated + L  # dynamic seed
        )
        all_peptides.extend(peps)

# ------------------------------
# Save results (append)
# ------------------------------
new_df = pd.DataFrame({"peptide": all_peptides, "length": [len(p) for p in all_peptides]})
if os.path.exists(output_file):
    new_df.to_csv(output_file, mode="a", header=False, index=False)
else:
    new_df.to_csv(output_file, index=False)

print(f"Saved {len(all_peptides)} peptides to {output_file}")