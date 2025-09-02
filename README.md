# RoBERTa – A Generative Language Model for Peptide Design

This repository contains the code for the generative component of an MSc thesis, focusing on using a **Transformer-based language model**, **RoBERTa**, to design and generate novel peptides. The goal is to create peptides with high predicted binding affinity to a specific **Human Leukocyte Antigen (HLA)** allele, specifically **HLA-A*02:01**. This project demonstrates the model's ability to learn the underlying rules of peptide–HLA binding beyond simple classification.

**Code for RoBERTa is available on GitHub:** [https://github.com/Felipepuiggarimedici/RoBERTa](https://github.com/Felipepuiggarimedici/RoBERTa)

---

## Overview

The RoBERTa model was adapted from a bidirectional language learning architecture to function as a generative tool. Instead of predicting a binding probability, the model was trained to predict masked amino acids within a peptide sequence based on the surrounding context. By leveraging this capability, the model can generate new peptides that are likely to bind.

---

## Methodology

The core methodology involved a two-step process:

1. **Model Training**  
   RoBERTa was pre-trained from scratch on a comprehensive dataset of known binding peptides to learn sequence patterns and motifs characteristic of binders.

2. **Peptide Generation**  
   Using **Gibbs sampling**, the model iteratively predicted and replaced masked amino acids in an initial sequence. This was repeated to generate a large library of **100,000 artificial peptides** for **HLA-A*02:01**.

---

## RoBERTa Model Architecture

The model is specifically designed to generate peptides for a fixed HLA, with separate models trained for each allele. This allows the model to capture the statistical patterns of binders for its target allele through **self-supervised Masked Language Modeling (MLM)**.

### Masked Language Modeling (MLM) and Loss Function

The model is trained to predict a proportion of masked tokens within a peptide sequence by minimizing cross-entropy loss:

```text
L(a_masked) = - (1/N) * sum_{i=1}^N sum_{j in X_i} log p_hat(a_{i,j} | a_i_masked)
```

Here, `a` represents tokenized peptide binders, `X_i` is the set of masked token positions for sample `i`, and `p_hat` is the pseudo-likelihood learned by the model.

> Note: formulas are presented in plain text blocks so they render correctly on GitHub.

### Multi-Head Attention

The model uses **multi-head attention** to focus on different parts of the peptide sequence simultaneously. The attention score is calculated as:

```text
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
```

### Positional Embedding

Since attention alone does not encode token order, **positional embeddings** are added. The special token `[SEP]` marks the end of a sequence.

The model also uses **GELU activations** in feed-forward layers and a dropout rate of 0.1.

### Pseudo-Likelihood (PLL) for Evaluation

To evaluate the model on unseen sequences, we compute the **pseudo-likelihood (PLL)**:

```text
PLL(a) = sum_{i=1}^{L} log p_hat(a_i | a_1 ... a_{i-1}, a_{i+1} ... a_L)
```

This metric measures how well the model predicts each amino acid given its context; however, PLL does not always map directly to biological function.

---

## Analysis and Results

The generated peptides were analyzed using the `quality.py` script. **The statistical analysis of 1-, 2-, and 3-site frequencies and 2- and 3-site covariances is from `quality.py`, developed by Yinfei Yang, and is explored in detail in `metrics.ipynb`.**

### Key Notebooks

* **`metrics.ipynb`** — Analyzes amino acid frequencies (1-, 2-, 3-site), empirical correlations (2-, 3-site), and motif visualizations.  
* **`lossAnalysis.ipynb`** — Uses `lossAnalysis.py` to analyze training and validation loss convergence and to decide early stopping.  
* **`hyperparameterAnalysis.ipynb`** — Analyzes hyperparameter sweep results to select optimal settings.  
* **`createROCs.ipynb`** — Generates ROC and precision–recall curves to evaluate binder vs. non-binder separation using PLL scores.  
* **`levenshteinDist.ipynb`** — Computes Levenshtein (edit) distances between generated peptides and training data to assess novelty.  
* **`netmhcbenchmark.ipynb`** — Validates generated peptides using **NetMHCpan-4.1** predictions for the HLA-A*02:01 allele.

All memory-intensive experiments were conducted on a **Linux HPC** (High Performance Computing) environment.

---

## Codebase

* `antibertaTrainHLAA0201.py` — Trains the final RoBERTa model for HLA-A*02:01 using the selected hyperparameters.  
* `generatePeptides.py` — Core generative script implementing Gibbs sampling to create peptide libraries.  
* `runSingleFold.py` — Quick single-fold validation script for fast checks.  
* `run5Fold.py` — Full 5-fold cross-validation for robust evaluation.  
* `hyperparamSearch.py` — Master script for launching HPC grid search jobs across hyperparameter combinations.  
* `quality.py` — Statistical analysis utilities (1/2/3-site frequencies and 2/3-site covariances); developed by Yinfei Yang.  
* `lossAnalysis.py` — Utilities for analyzing training/validation loss curves.

---

## Repository Structure

```
.
├─ antibertaTrainHLAA0201.py
├─ generatePeptides.py
├─ runSingleFold.py
├─ run5Fold.py
├─ hyperparamSearch.py
├─ quality.py
├─ lossAnalysis.py
├─ hyperparamData/
├─ results/
├─ netmhcbenchmark/
├─ modelsPeptideOnly/
│  └─ HLA_HLA-A_02-01/
│     └─ model.safetensors
└─ generatedPeptides/
   └─ HLAA0201/
      └─ generatedPeptides.csv
```

* `results/`: saved experiment outputs and plots.  
* `netmhcbenchmark/`: NetMHCpan validation results.  
* `modelsPeptideOnly/HLA_HLA-A_02-01/`: final trained model in `safetensors` format.  
* `generatedPeptides/HLAA0201/`: generated peptide library (`generatedPeptides.csv`).  
* `hyperparamData/`: CSVs and logs from hyperparameter sweeps.

---

## Tools & Dependencies

* Python (>=3.8 recommended)  
* PyTorch  
* Hugging Face Transformers  
* NumPy  
* scikit-learn  
* Matplotlib  
* Linux HPC environment (for memory-intensive model training and generation)

(Include exact package versions in a `requirements.txt` or `environment.yml` for reproducibility.)

---

## Reproducibility Notes

* Large model files use Git LFS — ensure `git lfs install` is run before cloning or pulling large `safetensors` files.  
* Training and generative jobs were run on an HPC cluster due to memory and GPU requirements. For local experiments, reduce batch sizes and sequence lengths accordingly.

---

## References

1. Imperial College Research Computing Service. (2022). doi:10.14469/hpc/2232.  
2. Liu, Y., Ott, M., Goyal, N., Du, J., Li, M., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.  
3. Jurtz, V., Paul, S., Andreatta, M., Marcatili, P., Peters, B., & Nielsen, M. (2017). NetMHCpan-4.1: improved peptide–MHC class I interaction predictions. bioRxiv / later versions.  
4. Vita, R., Overton, J. A., Greenbaum, J. A., Ponomarenko, J., Clark, J. D., et al. (2019). The Immune Epitope Database (IEDB). Nucleic Acids Research.  
5. Wang, J., Shen, T., Xie, T., & Zhao, Y. (2019). BERT as a Markov Random Field Language Model.  
6. Bojchevski, A., et al. (2021). AntiBERTa: A generative model for antibody variable regions. arXiv:2104.09945.

---

## Contact

For questions about the code or data, open an issue on the GitHub repository or contact the author via the repository profile.

---

**End of README**
