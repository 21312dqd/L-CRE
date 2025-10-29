LCRE - Gene Expression Classifier
A deep learning-based tool for predicting plant gene expression levels from promoter and terminator sequences.
📋 Overview
LCRE (Low/High gene expression Classification using Regulatory Elements) is a deep learning framework that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to predict gene expression levels by analyzing promoter and terminator sequences. The current implementation provides a classifier for tomato (Solanum lycopersicum) genome.
✨ Key Features

Sequence Encoding: One-hot encoding of DNA sequences
Hybrid Deep Learning Architecture: Combined CNN and LSTM neural network
Chromosome-level Validation: Leave-one-chromosome-out cross-validation strategy
Class Balancing: Automatic balancing of high/low expression genes in training data
Performance Tracking: Model checkpointing and performance metrics logging for each epoch

🔧 Requirements
Python Version

Python 3.7+

Core Dependencies
tensorflow >= 2.x
pandas
numpy
pyranges
pyfaidx
scikit-learn
Installation
bashpip install tensorflow pandas numpy pyranges pyfaidx scikit-learn
📁 Data Requirements
The project requires the following data file structure:
project_root/
├── tpm_counts/
│   └── solanum_counts.csv          # TPM expression data
├── gene_models/
│   └── Solanum_lycopersicum.SL3.0.52.gtf  # Gene annotation file
├── genomes/
│   └── Solanum_lycopersicum.SL3.0.dna.toplevel.fa  # Reference genome
├── validation_genes.pickle          # Validation gene ID set
└── saved_models/                    # Model save directory (auto-created)
Data Format Specifications

TPM Data (solanum_counts.csv):

Must contain logMaxTPM column
Row indices should be gene IDs


GTF File: Standard genome annotation format, must include:

Gene coordinate information
gene_biotype field (only protein_coding genes are used)


FASTA File: Standard genome sequence format
Validation Gene Pickle File: Dictionary containing gene IDs for validation

🚀 Usage
Basic Usage
bashpython lcre_classifier.py
