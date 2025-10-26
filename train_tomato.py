import os
import pandas as pd
import numpy as np
import pickle
import pyranges as pr
from pyfaidx import Fasta
from tensorflow.keras import Sequential, optimizers, backend, models
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Dropout, Flatten, LayerNormalization, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle
import tensorflow as tf

# Configuration for GPU memory growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Tomato-specific constants
CONFIG = {
    'MAPPED_READS': 'solanum_counts.csv',
    'GENE_MODEL': 'Solanum_lycopersicum.SL3.0.52.gtf',
    'GENOME': 'Solanum_lycopersicum.SL3.0.dna.toplevel.fa',
    'PICKLE_KEY': 'sol',
    'NUM_CHROMOSOMES': 12
}


def encode_sequence(sequence):
    """Convert DNA sequence to one-hot encoded numpy array."""
    nucleotide_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'unk': [0, 0, 0, 0]
    }
    encoded = np.zeros((len(sequence), 4))
    for idx, nucleotide in enumerate(sequence):
        encoded[idx, :] = nucleotide_map.get(nucleotide, nucleotide_map['unk'])
    return encoded


class SequenceLoader:
    """Loads and processes genomic sequences from FASTA and GTF files."""

    def __init__(self, fasta_path, gtf_path, validation_chromosome, pickled_ids, pickle_key,
                 upstream=1000, downstream=500, for_prediction=False):
        """
        Initialize SequenceLoader with paths and parameters.

        Args:
            fasta_path: Path to reference genome FASTA file
            gtf_path: Path to gene models GTF file
            validation_chromosome: Chromosome used for validation
            pickled_ids: Path to pickled validation gene IDs
            pickle_key: Key for accessing pickled validation genes
            upstream: Nucleotides to extend upstream
            downstream: Nucleotides to extend downstream
            for_prediction: Flag to indicate prediction mode
        """
        self.fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True, read_ahead=10000)
        self.upstream = upstream
        self.downstream = downstream
        self.validation_chromosome = str(validation_chromosome)
        self.pickled_ids = pickled_ids
        self.pickle_key = pickle_key

        gene_data = pr.read_gtf(gtf_path, as_df=True)
        gene_data = gene_data[gene_data['Feature'] == 'gene']
        gene_data = gene_data[gene_data['gene_biotype'] == 'protein_coding']
        gene_data = gene_data[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
        self.gtf = gene_data if not for_prediction else gene_data[gene_data['Chromosome'] == self.validation_chromosome]

    def extract_sequences(self):
        """Extract and encode training and validation sequences."""
        train_sequences, train_gene_ids = [], []
        validation_sequences, validation_gene_ids = [], []

        with open(self.pickled_ids, 'rb') as handle:
            validation_genes = pickle.load(handle)
        print(f"Validation gene keys: {validation_genes.keys()}")

        for chromosome, start, end, strand, gene_id in self.gtf.values:
            if strand == '+':
                promoter_start, promoter_end = start - self.upstream, start + self.downstream
                terminator_start, terminator_end = end - self.downstream, end + self.upstream
                if promoter_start > 0 and terminator_start > 0:
                    encoded_seq = np.concatenate([
                        encode_sequence(self.fasta[chromosome][promoter_start:promoter_end]),
                        np.zeros((20, 4)),
                        encode_sequence(self.fasta[chromosome][terminator_start:terminator_end])
                    ])
                    if encoded_seq.shape[0] == 2 * (self.upstream + self.downstream) + 20:
                        if chromosome == self.validation_chromosome and gene_id in validation_genes[self.pickle_key]:
                            validation_sequences.append(encoded_seq)
                            validation_gene_ids.append(gene_id)
                        else:
                            train_sequences.append(encoded_seq)
                            train_gene_ids.append(gene_id)
            else:
                promoter_start, promoter_end = end - self.downstream, end + self.upstream
                terminator_start, terminator_end = start - self.upstream, start + self.downstream
                if promoter_start > 0 and terminator_start > 0:
                    encoded_seq = np.concatenate([
                        encode_sequence(self.fasta[chromosome][promoter_start:promoter_end])[::-1, ::-1],
                        np.zeros((20, 4)),
                        encode_sequence(self.fasta[chromosome][terminator_start:terminator_end])[::-1, ::-1]
                    ])
                    if encoded_seq.shape[0] == 2 * (self.upstream + self.downstream) + 20:
                        if chromosome == self.validation_chromosome and gene_id in validation_genes[self.pickle_key]:
                            validation_sequences.append(encoded_seq)
                            validation_gene_ids.append(gene_id)
                        else:
                            train_sequences.append(encoded_seq)
                            train_gene_ids.append(gene_id)

        return train_sequences, validation_sequences, train_gene_ids, validation_gene_ids


class ConvolutionalNetwork:
    """Convolutional neural network for genomic sequence analysis."""

    def __init__(self, train_sequences, validation_sequences, train_ids, validation_ids,
                 chromosome, tpm_data, organism, outer_flank=1000, inner_flank=500,
                 size_effect=False, tissue=""):
        self.train_sequences = train_sequences
        self.validation_sequences = validation_sequences
        self.train_ids = train_ids
        self.validation_ids = validation_ids
        self.chromosome = chromosome
        self.tpm_data = tpm_data
        self.organism = organism
        self.outer_flank = outer_flank
        self.inner_flank = inner_flank
        self.size_effect = size_effect
        self.tissue = tissue

    def build_model(self, x_train, x_val, y_train, y_val):
        """Build and train the convolutional neural network."""
        backend.clear_session()
        model = Sequential([
            Conv1D(filters=64, kernel_size=8, activation='relu', padding='same',
                   input_shape=(x_train.shape[1], x_train.shape[2])),
            Conv1D(filters=64, kernel_size=8, activation='relu', padding='same'),
            MaxPool1D(pool_size=8, padding='same'),
            Dropout(0.25),
            LayerNormalization(),
            LSTM(units=64, return_sequences=True),
            Dropout(0.25),
            Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
            Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
            MaxPool1D(pool_size=8, padding='same'),
            Dropout(0.25),
            Conv1D(filters=64, kernel_size=8, activation='relu', padding='same'),
            Conv1D(filters=64, kernel_size=8, activation='relu', padding='same'),
            MaxPool1D(pool_size=8, padding='same'),
            Dropout(0.25),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(0.25),
            Dense(units=64, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])

        model_dir = (f'saved_models/size_effect/{self.organism}_{self.chromosome}_{self.outer_flank}_{self.inner_flank}'
                     if self.size_effect else
                     f'saved_models/{self.organism}_model_{self.chromosome}{self.tissue}')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'epoch_models'), exist_ok=True)

        performance_log = []

        class PerformanceCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                model_path = os.path.join(model_dir, 'epoch_models', f'model_epoch_{epoch}.h5')
                self.model.save(model_path)
                performance_log.append({
                    'epoch': epoch,
                    'val_loss': logs.get('val_loss'),
                    'val_accuracy': logs.get('val_accuracy')
                })

        callbacks = [
            PerformanceCallback(),
            ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), save_best_only=True, verbose=1)
        ]

        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        model.fit(x_train, y_train, batch_size=64, epochs=35,
                  validation_data=(x_val, y_val), callbacks=callbacks)

        pd.DataFrame(performance_log).to_csv(os.path.join(model_dir, 'performance_log.csv'), index=False)

        best_model = models.load_model(os.path.join(model_dir, 'best_model.h5'))
        predictions = best_model.predict(x_val)
        val_auroc = roc_auc_score(y_val, predictions)
        binary_predictions = predictions > 0.5
        val_accuracy = accuracy_score(y_val, binary_predictions)

        print(f'\nBest Model Performance for Chromosome {self.chromosome}')
        print(f'Accuracy: {val_accuracy:.4f}, auROC: {val_auroc:.4f}\n')

        return ([val_accuracy, val_auroc, self.organism, self.outer_flank, x_train.shape[0]]
                if self.size_effect else
                [val_accuracy, val_auroc, self.organism, x_train.shape[0]])

    def train_model(self):
        """Prepare data and train the model."""
        train_labels, train_seqs = [], []
        validation_labels, validation_seqs = [], []

        for gene_id, sequence in zip(self.train_ids, self.train_sequences):
            train_labels.append(self.tpm_data.loc[gene_id, 'true_target'])
            train_seqs.append(sequence)
        for gene_id, sequence in zip(self.validation_ids, self.validation_sequences):
            validation_labels.append(self.tpm_data.loc[gene_id, 'true_target'])
            validation_seqs.append(sequence)

        train_labels, validation_labels = np.array(train_labels), np.array(validation_labels)
        train_seqs, validation_seqs = np.array(train_seqs), np.array(validation_seqs)

        low_train_indices = np.where(train_labels == 0)[0]
        high_train_indices = np.where(train_labels == 1)[0]
        min_class_size = min(len(low_train_indices), len(high_train_indices))
        selected_low_train = np.random.choice(low_train_indices, min_class_size, replace=False)
        selected_high_train = np.random.choice(high_train_indices, min_class_size, replace=False)

        x_train = np.concatenate([
            train_seqs[selected_low_train],
            train_seqs[selected_high_train]
        ])
        y_train = np.concatenate([
            train_labels[selected_low_train],
            train_labels[selected_high_train]
        ])
        x_train, y_train = shuffle(x_train, y_train, random_state=42)

        low_val_indices = np.where(validation_labels == 0)[0]
        high_val_indices = np.where(validation_labels == 1)[0]
        x_val = np.concatenate([validation_seqs[low_val_indices], validation_seqs[high_val_indices]])
        y_val = np.concatenate([validation_labels[low_val_indices], validation_labels[high_val_indices]])

        print(f"Training shape: {x_train.shape}, Validation shape: {x_val.shape}")
        print(f"Validation size: {x_val.shape[0]}")

        x_train[:, self.outer_flank:self.outer_flank + 3, :] = 0
        x_train[:, self.outer_flank + (self.inner_flank * 2) + 17:self.outer_flank + (self.inner_flank * 2) + 20, :] = 0
        x_val[:, self.outer_flank:self.outer_flank + 3, :] = 0
        x_val[:, self.outer_flank + (self.inner_flank * 2) + 17:self.outer_flank + (self.inner_flank * 2) + 20, :] = 0

        return self.build_model(x_train, x_val, y_train, y_val)


def prepare_validation_sequences(fasta_file, gtf_file, tpm_file, validation_chromosomes,
                                 pickle_key=False, upstream=1000, downstream=500):
    """Prepare validation sequences with labels and gene IDs."""
    fasta = Fasta(f'genomes/{fasta_file}', as_raw=True, sequence_always_upper=True, read_ahead=10000)
    gene_models = pr.read_gtf(f'gene_models/{gtf_file}', as_df=True)
    gene_models = gene_models[gene_models['Feature'] == 'gene']
    gene_models = gene_models[gene_models['gene_biotype'] == 'protein_coding']
    gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]

    if isinstance(validation_chromosomes, list):
        gene_models = gene_models[gene_models['Chromosome'].isin(validation_chromosomes)]
    else:
        gene_models = gene_models[gene_models['Chromosome'].isin([validation_chromosomes])]

    if pickle_key:
        with open('validation_genes.pickle', 'rb') as handle:
            validation_genes = pickle.load(handle)
        gene_models = gene_models[gene_models['gene_id'].isin(validation_genes[pickle_key])]

    tpm_data = pd.read_csv(f'tpm_counts/{tpm_file}', index_col=0)
    true_targets = []
    for log_count in tpm_data['logMaxTPM'].values:
        if log_count <= np.percentile(tpm_data['logMaxTPM'], 25):
            true_targets.append(0)
        elif log_count >= np.percentile(tpm_data['logMaxTPM'], 75):
            true_targets.append(1)
        else:
            true_targets.append(2)
    tpm_data['true_target'] = true_targets

    encoded_sequences, labels, gene_ids = [], [], []
    for chromosome, start, end, strand, gene_id in gene_models.values:
        if strand == '+':
            promoter_start, promoter_end = start - upstream, start + downstream
            terminator_start, terminator_end = end - downstream, end + upstream
            if promoter_start > 0 and terminator_start > 0:
                encoded_seq = np.concatenate([
                    encode_sequence(fasta[chromosome][promoter_start:promoter_end]),
                    np.zeros((20, 4)),
                    encode_sequence(fasta[chromosome][terminator_start:terminator_end])
                ])
                if encoded_seq.shape[0] == 2 * (upstream + downstream) + 20:
                    encoded_sequences.append(encoded_seq)
                    labels.append(tpm_data.loc[gene_id, 'true_target'])
                    gene_ids.append(gene_id)
        else:
            promoter_start, promoter_end = end - downstream, end + upstream
            terminator_start, terminator_end = start - upstream, start + downstream
            if promoter_start > 0 and terminator_start > 0:
                encoded_seq = np.concatenate([
                    encode_sequence(fasta[chromosome][promoter_start:promoter_end])[::-1, ::-1],
                    np.zeros((20, 4)),
                    encode_sequence(fasta[chromosome][terminator_start:terminator_end])[::-1, ::-1]
                ])
                if encoded_seq.shape[0] == 2 * (upstream + downstream) + 20:
                    encoded_sequences.append(encoded_seq)
                    labels.append(tpm_data.loc[gene_id, 'true_target'])
                    gene_ids.append(gene_id)

    labels = np.array(labels)
    encoded_sequences = np.array(encoded_sequences)
    gene_ids = np.array(gene_ids)

    low_val_indices = np.where(labels == 0)[0]
    high_val_indices = np.where(labels == 1)[0]
    x_val = np.concatenate([encoded_sequences[low_val_indices], encoded_sequences[high_val_indices]])
    y_val = np.concatenate([labels[low_val_indices], labels[high_val_indices]])
    selected_gene_ids = np.concatenate([gene_ids[low_val_indices], gene_ids[high_val_indices]])

    x_val[:, upstream:upstream + 3, :] = 0
    x_val[:, upstream + (downstream * 2) + 17:upstream + (downstream * 2) + 20, :] = 0

    return x_val, y_val, selected_gene_ids


def train_tomato_classifier():
    """Train a convolutional neural network for tomato gene expression classification."""
    # Initialize directories
    os.makedirs('../results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    # Check for existing results
    results_path = "../results/sol_root_result.csv"
    if os.path.exists(results_path):
        print("Results already exist. Skipping training.")
        return

    # Load and label TPM data
    tpm_data = pd.read_csv(f'tpm_counts/{CONFIG["MAPPED_READS"]}', index_col=0)
    true_targets = []
    tpm_percentiles = np.percentile(tpm_data['logMaxTPM'], [25, 75])
    for log_count in tpm_data['logMaxTPM'].values:
        if log_count <= tpm_percentiles[0]:
            true_targets.append(0)
        elif log_count >= tpm_percentiles[1]:
            true_targets.append(1)
        else:
            true_targets.append(2)
    tpm_data['true_target'] = true_targets
    print("TPM counts preview:")
    print(tpm_data.head())

    training_results = []

    # Process each chromosome
    for chromosome in range(1, CONFIG['NUM_CHROMOSOMES'] + 1):
        print(f"\nProcessing chromosome {chromosome}")

        # Load sequences
        sequence_loader = SequenceLoader(
            fasta_path=f"genomes/{CONFIG['GENOME']}",
            gtf_path=f"gene_models/{CONFIG['GENE_MODEL']}",
            validation_chromosome=chromosome,
            pickled_ids='validation_genes.pickle',
            pickle_key=CONFIG['PICKLE_KEY']
        )
        train_seqs, val_seqs, train_ids, val_ids = sequence_loader.extract_sequences()

        print("\n" + "-" * 80)
        print(f"Training model for chromosome {chromosome} with promoter-terminator sequences")
        print("-" * 80)

        # Initialize and train model
        conv_network = ConvolutionalNetwork(
            train_sequences=train_seqs,
            validation_sequences=val_seqs,
            train_ids=train_ids,
            validation_ids=val_ids,
            chromosome=chromosome,
            tpm_data=tpm_data,
            organism='solanum'
        )
        result = conv_network.train_model()
        training_results.append(result)

    # Save training results
    results_df = pd.DataFrame(
        training_results,
        columns=['accuracy', 'auROC', 'organism', 'training_size']
    )
    results_df.to_csv(results_path, index=False)
    print(f"Training completed. Results saved to {results_path}")


if __name__ == "__main__":
    train_tomato_classifier()