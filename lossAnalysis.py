import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch

# --- Consistent font sizes ---
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# --- Seeds (optional) ---
seed = 6
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

colours = ['#9BC995', "#083D77", '#9A031E', '#C4B7CB', '#FC7753']

# --- Paths (adjust if needed) ---
folder = r"modelsPeptideOnly\HLA_HLA-A_02-01"
file_name = "loss_history_HLA-A_02-01.csv"
csv_path = os.path.join(folder, file_name)

# --- Load and coerce to numeric ---
df = pd.read_csv(csv_path)
df['step'] = pd.to_numeric(df['step'], errors='coerce')
df['train_loss'] = pd.to_numeric(df.get('train_loss'), errors='coerce')
df['eval_loss'] = pd.to_numeric(df.get('eval_loss'), errors='coerce')

# drop any rows without a valid step
df = df.dropna(subset=['step']).copy()
if df.empty:
    raise ValueError(f"No valid 'step' values found in {csv_path}.")

# convert integer-like steps to int
if np.all(np.equal(np.mod(df['step'], 1), 0)):
    df['step'] = df['step'].astype(int)

# --- Recorded (non-NaN) points only ---
rec_train = df.dropna(subset=['train_loss']).sort_values('step')
rec_eval  = df.dropna(subset=['eval_loss']).sort_values('step')

# Basic sanity checks
if rec_train.empty and rec_eval.empty:
    raise ValueError("No numeric train_loss or eval_loss values found. Nothing to plot.")

# Early stopping window logic (same as your code)
patience_steps = 20 * 100
last_step = int(df['step'].iloc[-1])
early_stop_start = max(int(df['step'].iloc[0]), last_step - patience_steps)

# find actual recorded min eval loss (from recorded eval rows)
if not rec_eval.empty:
    min_idx = rec_eval['eval_loss'].idxmin()
    min_step = int(rec_eval.loc[min_idx, 'step'])
    min_loss = float(rec_eval.loc[min_idx, 'eval_loss'])
else:
    min_step, min_loss = None, None

# --- Plot: only recorded points joined by lines ---
FIGSIZE = (10, 8)   
fig, ax = plt.subplots(figsize=FIGSIZE)

if not rec_train.empty:
    ax.plot(rec_train['step'].values,
            rec_train['train_loss'].values,
            marker='o', linestyle='-', linewidth=2, markersize=4,
            color=colours[1], label='Training Loss')

if not rec_eval.empty:
    ax.plot(rec_eval['step'].values,
            rec_eval['eval_loss'].values,
            marker='o', linestyle='-', linewidth=2, markersize=4,
            color=colours[2], label='Validation Loss')

# Shade patience window and vertical stop line
ax.axvspan(early_stop_start, last_step, color=colours[4], alpha=0.2,
           label=f'Patience Window (Last {patience_steps} steps)')
ax.axvline(x=last_step, color=colours[4], linestyle='--', linewidth=1.5,
           label='Training Stopped')

# Annotate min eval loss (from recorded points)
if min_step is not None:
    ax.scatter(min_step, min_loss, color=colours[0], s=120, zorder=5,
               label=f'Min Eval Loss (Step {min_step}, Loss: {min_loss:.4f})')

ax.set_title("Cross Entropy Loss per Step")
ax.set_xlabel("Step")
ax.set_ylabel("Cross Entropy Loss")
ax.legend(loc='upper right')

# Save and show
os.makedirs(folder, exist_ok=True)
plot_path = os.path.join(folder, "loss_curve_recorded_only.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.show()

print(f"Loss curve saved to: {plot_path}")
