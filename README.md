# RoBERTa - A Generative Language Model for Peptide Design

This repository contains the code for the generative component of an MSc thesis, focusing on using a **Transformer-based language model**, **RoBERTa**, to design and generate novel peptides. The goal is to create peptides with high predicted binding affinity to a specific **Human Leukocyte Antigen (HLA)** allele, specifically **HLA-A\*02:01**. This project demonstrates the model's ability to learn the underlying rules of peptide-HLA binding beyond simple classification.

***

## Overview

The RoBERTa model was adapted from a bidirectional language learning architecture to function as a generative tool. Instead of predicting a binding probability, the model was trained to predict masked amino acids within a peptide sequence based on the surrounding context. By leveraging this capability, the model can generate new peptides that are likely to bind.

***

## Methodology

The core methodology for this work involved a two-step process:

1.  **Model Training**: The RoBERTa model was pre-trained from scratch on a comprehensive dataset of known binding peptides to learn the specific sequence patterns and motifs characteristic of binders.
2.  **Peptide Generation**: Using a technique called **Gibbs Sampling**, the model iteratively predicted and replaced masked amino acids in an initial sequence. This process was repeated to generate a large library of **100,000 artificial peptides** for the **HLA-A\*02:01** allele.

***

## RoBERTa Model Architecture

Our RoBERTa model is specifically designed to generate peptides for a fixed HLA, with a separate model trained for each allele. This design allows the model to concentrate on capturing the statistical patterns of binders for its target allele through **self-supervised Masked Language Modeling (MLM)**. This modular approach can be applied to other HLA alleles by providing the corresponding peptide data.

### Masked Language Modeling (MLM) and the Loss Function

The model is trained to predict a proportion of masked tokens within a peptide sequence by minimizing the cross-entropy loss of the masked training data. The loss function is defined as:

$$L(a^{\text{masked}})= - \frac{1}{N} \sum_{i=1}^{N} \sum_{j \in X_i} \log \hat{p}(a_{i,j} \mid a_i^{\text{masked}})$$

Here, $a$ represents the tokenized peptide binders, $X_i$ is the set of masked token positions for the $i$-th sample, and $\hat{p}$ is the pseudo-likelihood learned by the model.

### General Architecture

The RoBERTa architecture is a **bidirectional deep learning language model** that incorporates context from both preceding and succeeding tokens. This bidirectionality is crucial because each amino acid in a peptide exhibits complex interdependencies with others in the sequence.

The model's power comes from two key components:

* **Multi-Head Attention**: This mechanism allows the model to simultaneously focus on different parts of the peptide sequence to understand how different amino acids relate to each other. The attention score is calculated by:

<p>
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
</p>
* **Positional Embedding**: Since the attention mechanism doesn't inherently understand the order of amino acids, positional embeddings are used to inject this information. Unlike the original Transformer model, the positional embedding of BERT-based models is built from learnable parameters. The special token **[SEP]** marks the end of the sequence.

The model also uses **GELU activation functions** in the feed-forward layers and a dropout rate of 0.1.

### Pseudo-Likelihood (PLL) for Evaluation

In addition to the loss function, we use the **pseudo-likelihood (PLL)** to evaluate the model's performance on unseen sequences. The PLL is defined as the sum of log-probabilities of each amino acid, conditioned on the rest of the sequence, effectively measuring how well the model can predict each token given its context.

$$\text{PLL}(a) := \sum_{i=1}^{L} \log \hat{p}(a_i \mid a_1 \dots a_{i-1} a_{i+1} \dots a_L)$$

While a valuable metric for evaluating the model's fluency, PLL has known limitations and does not always correlate directly with biological function.

***

## Analysis and Results

The generated peptides were rigorously analyzed to validate the model's effectiveness. The statistical analysis is handled by the `quality.py` script and is explored in detail across several notebooks:

### `metrics.ipynb`

This notebook explores:

* **Statistical Tests**: Compares amino acid frequencies (1-, 2-, and 3-site) and empirical correlations (2- and 3-site) of generated peptides to known binders.
* **Motif Analysis**: Visualizes the binding motifs discovered in the generated sequences to ensure they align with established biological principles.

### Other Analysis Notebooks

* **`lossAnalysis.ipynb`**: Uses the `lossAnalysis.py` script to analyze the convergence of the model's training and validation loss over time, crucial for determining the optimal number of training epochs and implementing early stopping.
* **`hyperparameterAnalysis.ipynb`**: Analyzes the results from the hyperparameter search to identify the optimal combination of settings for the model.
* **`createROCs.ipynb`**: Generates **ROC (Receiver Operating Characteristic)** and **Precision-Recall curves** to evaluate the model's ability to distinguish between binders and non-binders based on their PLL scores.
* **`levenshteinDist.ipynb`**: Measures the **Levenshtein distance** (edit distance) between the generated peptides and the training data to confirm that the model is producing truly **novel sequences** rather than simply memorizing the input.
* **`netmhcbenchmark.ipynb`**: Further validates the results by assessing the generated peptides with an independent computational model, **NetMHCpan-4.1**, to confirm their predicted binding affinity for the HLA-A\*02:01 allele.

All memory-intensive generative experiments were conducted on a **Linux HPC (High Performance Computing)** environment.

***

## Codebase

The codebase for this project is designed for efficient and systematic experimentation, particularly for hyperparameter tuning on an HPC cluster.

* `antibertaTrainHLAA0201.py`: Used to train the final model for the HLA-A\*02:01 allele using the best hyperparameters. It tokenizes data, sets up the model, and trains it using the Hugging Face Trainer API.
* `generatePeptides.py`: The core generative component that uses **Gibbs sampling** to generate a large library of novel peptides from the trained model.
* `runSingleFold.py`: Used for initial tests and single-fold validation to quickly assess the performance of a given set of hyperparameters.
* `run5Fold.py`: Handles comprehensive **5-fold cross-validation** for a specific set of hyperparameters, ensuring the model's performance is robust.
* `hyperparamSearch.py`: The master script for HPC job submission that automates a comprehensive **grid search** over various hyperparameters by submitting parallel jobs.

***

## Repository Structure and Key Files

The project's file structure is organized to facilitate systematic experimentation and data management:

* **`results/`**: A general folder for various saved results.
* **`netmhcbenchmark/`**: Contains the results from the **NetMHCpan-4.1** external validation.
* **`modelsPeptideOnly/HLA_HLA-A_02-01/`**: The directory where the final trained model is saved, using the **`safetensors`** format.
* **`hyperparamData/`**: A CSV file with loss information from the hyperparameter sweeps.
* **`generatedPeptides/HLAA0201/`**: The location of the final, important `generatedPeptides.csv` file, which contains the library of generated peptides.

***

## Tools

The key technologies used in this project include:

* **Python**
* **PyTorch**
* **NumPy**
* **Matplotlib**
* **scikit-learn**
* **Hugging Face Transformers**
* **Linux**

***

## References

1.  Imperial College Research Computing Service. (2022). Imperial College Research Computing Service. doi: 10.14469/hpc/2232. URL: https://doi.org/10.14469/hpc/2232
2.  Liu, Y., Ott, M., Goyal, N., Du, J., Li, M., Palmer, A., ... & Stoyanov, K. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692.
3.  G. L. A. O. A. P. L. A. P. S. S. A. P. A. N. D. E. M. E. A. P. N. D. I. D. (2020). NetMHCpan-4.1: The MHC-pan tool with improved performance and allele coverage. bioRxiv.
4.  P. H. L. A. T. B. R. S. H. K. P. H. S. B. N. H. G. H. W. (2020). The Immune Epitope Database and Analysis Resource (IEDB) API. Nucleic Acids Research.
5.  Wang, J., Shen, T., Xie, T., & Zhao, Y. (2019). BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model.
6.  Bojchevski, A., N. D. L. V. G. D. S. (2021). AntiBERTa: A generative model for antibody variable regions. arXiv preprint arXiv:2104.09945.