# LCRE - Gene Expression Classifier

A deep learning-based tool for predicting plant gene expression levels from promoter and terminator sequences.

## 📋 Overview

LCRE (Low/High gene expression Classification using Regulatory Elements) is a deep learning framework that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to predict gene expression levels by analyzing promoter and terminator sequences. The current implementation provides a classifier for tomato (*Solanum lycopersicum*) genome.

## ✨ Key Features

- **Sequence Encoding**: One-hot encoding of DNA sequences
- **Hybrid Deep Learning Architecture**: Combined CNN and LSTM neural network
- **Chromosome-level Validation**: Leave-one-chromosome-out cross-validation strategy
- **Class Balancing**: Automatic balancing of high/low expression genes in training data
- **Performance Tracking**: Model checkpointing and performance metrics logging for each epoch

## 🔧 Requirements

### Python Version
- Python 3.7+

### Core Dependencies
```
tensorflow >= 2.x
pandas
numpy
pyranges
pyfaidx
scikit-learn
```

### Installation
```bash
pip install tensorflow pandas numpy pyranges pyfaidx scikit-learn
```

## 📁 Data Requirements

The project requires the following data file structure:

```
project_root/
├── tpm_counts/
│   └── solanum_counts.csv          # TPM expression data
├── gene_models/
│   └── Solanum_lycopersicum.SL3.0.52.gtf  # Gene annotation file
├── genomes/
│   └── Solanum_lycopersicum.SL3.0.dna.toplevel.fa  # Reference genome
├── validation_genes.pickle          # Validation gene ID set
└── saved_models/                    # Model save directory (auto-created)
```

### Data Format Specifications

1. **TPM Data** (`solanum_counts.csv`):
   - Must contain `logMaxTPM` column
   - Row indices should be gene IDs

2. **GTF File**: Standard genome annotation format, must include:
   - Gene coordinate information
   - `gene_biotype` field (only protein_coding genes are used)

3. **FASTA File**: Standard genome sequence format

4. **Validation Gene Pickle File**: Dictionary containing gene IDs for validation

## 🚀 Usage

### Basic Usage

```bash
python lcre_classifier.py
```

### Workflow

1. **Data Loading and Preprocessing**
   - Load TPM data and classify by 25th and 75th percentiles
   - 0: Low expression genes (≤ 25th percentile)
   - 1: High expression genes (≥ 75th percentile)
   - 2: Medium expression genes (excluded from training)

2. **Sequence Extraction**
   - Promoter region: 1000bp upstream + 500bp downstream of TSS
   - Terminator region: 500bp upstream + 1000bp downstream of TES
   - 20bp zero-padding separates promoter and terminator sequences

3. **Model Training**
   - Independent validation for each chromosome
   - Training data: Remaining 11 chromosomes
   - Validation data: Current chromosome
   - 35 training epochs with batch size of 64

## 🏗️ Model Architecture

```
Input Layer: (sequence_length, 4) - One-hot encoded DNA sequence
↓
Conv1D(64, kernel=8) + ReLU
Conv1D(64, kernel=8) + ReLU
MaxPooling1D(8)
Dropout(0.25)
LayerNormalization
↓
LSTM(64, return_sequences=True)
Dropout(0.25)
↓
Conv1D(128, kernel=8) + ReLU
Conv1D(128, kernel=8) + ReLU
MaxPooling1D(8)
Dropout(0.25)
↓
Conv1D(64, kernel=8) + ReLU
Conv1D(64, kernel=8) + ReLU
MaxPooling1D(8)
Dropout(0.25)
↓
Flatten
Dense(128) + ReLU
Dropout(0.25)
Dense(64) + ReLU
Dense(1) + Sigmoid
↓
Output: Gene expression probability (0-1)
```

### Model Characteristics
- **Loss Function**: Binary crossentropy
- **Optimizer**: Adam (learning rate: 0.0001)
- **Regularization**: Dropout layers (0.25) + Layer Normalization
- **Sequence Length**: 3020bp (1500 + 20 + 1500)

## 📊 Output Results

### Saved Model Files
```
saved_models/
└── solanum_model_{chromosome}/
    ├── best_model.h5              # Best validation performance model
    ├── performance_log.csv        # Training process log
    └── epoch_models/
        ├── model_epoch_0.h5
        ├── model_epoch_1.h5
        └── ...
```

### Results File
- `../results/sol_root_result.csv`: Performance metrics for all chromosomes
  - accuracy: Validation accuracy
  - auROC: Area under ROC curve on validation set
  - organism: Species identifier
  - training_size: Number of training samples

## 📈 Performance Evaluation

The model is evaluated using the following metrics:
- **Accuracy**: Proportion of correctly classified samples
- **auROC**: Area under the ROC curve, evaluating overall classifier performance

## 🔧 Configuration Parameters

Parameters modifiable in the `CONFIG` dictionary:

```python
CONFIG = {
    'MAPPED_READS': 'solanum_counts.csv',     # TPM data file
    'GENE_MODEL': 'Solanum_lycopersicum.SL3.0.52.gtf',  # GTF file
    'GENOME': 'Solanum_lycopersicum.SL3.0.dna.toplevel.fa',  # FASTA file
    'PICKLE_KEY': 'sol',                       # Validation gene pickle key
    'NUM_CHROMOSOMES': 12                      # Number of chromosomes
}
```

Other adjustable parameters:
- `upstream`: Upstream extension in base pairs (default: 1000)
- `downstream`: Downstream extension in base pairs (default: 500)
- `batch_size`: Batch size (default: 64)
- `epochs`: Number of training epochs (default: 35)
- `learning_rate`: Learning rate (default: 0.0001)

## 🧬 Core Classes and Functions

### `encode_sequence(sequence)`
Converts DNA sequences to one-hot encoded arrays.

### `SequenceLoader`
Loads and processes genomic sequences from FASTA and GTF files.

### `ConvolutionalNetwork`
Builds, trains, and evaluates the convolutional neural network model.

### `prepare_validation_sequences()`
Prepares validation sequences with labels and gene IDs.

### `train_tomato_classifier()`
Main training function that orchestrates the entire training pipeline.

## 💡 Usage Recommendations

1. **GPU Acceleration**: Code automatically configures GPU memory growth; GPU training is recommended
2. **Data Balancing**: Automatically balances low/high expression gene counts for unbiased training
3. **Sequence Masking**: Masks are applied at TSS and TES positions to prevent position bias learning
4. **Checkpointing**: Models are saved at each epoch for potential training recovery

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or use shorter sequence lengths
2. **Data Files Not Found**: Check file paths and directory structure
3. **GPU Unavailable**: Code automatically falls back to CPU training

## 📚 Documentation

### Function Parameters

**SequenceLoader.__init__()**
- `fasta_path`: Path to reference genome FASTA file
- `gtf_path`: Path to gene models GTF file
- `validation_chromosome`: Chromosome used for validation
- `pickled_ids`: Path to pickled validation gene IDs
- `pickle_key`: Key for accessing pickled validation genes
- `upstream`: Nucleotides to extend upstream (default: 1000)
- `downstream`: Nucleotides to extend downstream (default: 500)

**ConvolutionalNetwork.__init__()**
- `train_sequences`: Training sequence data
- `validation_sequences`: Validation sequence data
- `train_ids`: Training gene IDs
- `validation_ids`: Validation gene IDs
- `chromosome`: Chromosome number for validation
- `tpm_data`: TPM expression data
- `organism`: Organism identifier

## 🔬 Scientific Background

### Regulatory Elements
The model leverages two key regulatory regions:
- **Promoters**: Control transcription initiation
- **Terminators**: Control transcription termination

### Expression Classification
Genes are classified into three categories based on maximum TPM values:
- **Low expression**: ≤ 25th percentile
- **High expression**: ≥ 75th percentile
- **Medium expression**: Between 25th and 75th percentile (excluded from binary classification)

## 📝 Citation

If you use this tool in your research, please cite the relevant publications.

## 📄 License

Please add an appropriate license according to your project requirements.

## 👥 Contributing

Issues and improvement suggestions are welcome!

## 📧 Contact

For questions, please contact via [Your Contact Information].

---

**Note**: This tool is specifically designed for plant genome analysis, particularly tomato genomes. To apply it to other species, configuration parameters and data paths need to be adjusted accordingly.

## 🔄 Extension to Other Species

To adapt LCRE for other plant species:

1. Update the `CONFIG` dictionary with species-specific files
2. Adjust `NUM_CHROMOSOMES` to match your species
3. Ensure GTF file contains `gene_biotype` annotations
4. Prepare validation gene sets using the same pickle format
5. Verify sequence coordinate systems match (0-based vs 1-based)

## 📊 Example Results

Typical performance metrics for tomato genome:
- Validation Accuracy: ~70-80%
- auROC: ~0.75-0.85

Performance may vary depending on:
- Quality of expression data
- Chromosome-specific features
- Gene density and annotation quality
