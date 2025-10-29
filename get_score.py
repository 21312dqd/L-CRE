import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from importlib import reload
import get_score
import multiprocessing
import os
import multiprocessing
# Ensure multiprocessing start method is set at the beginning of the main program
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
import os
import pandas as pd
import numpy as np
import pickle
import pyranges as pr
from pyfaidx import Fasta
from tensorflow.keras import Sequential, optimizers, backend, models
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle
import tensorflow as tf

def onehot(seq):
    code = {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'unk': [0, 0, 0, 0]}
    encoded = np.zeros(shape=(len(seq), 4))
    for i, nt in enumerate(seq):
        if nt in ['A', 'C', 'G', 'T']:
            encoded[i, :] = code[nt]
        else:
            encoded[i, :] = code['unk']
    return encoded

from sklearn.cluster import KMeans
from pyfaidx import Fasta
import pyranges as pr

def prepare_valid_seqs(fasta, gtf, tpms, val_chrom, pkey=False, upstream=1000, downstream=500):
    fasta = Fasta(f'genomes/{fasta}', as_raw=True, sequence_always_upper=True, read_ahead=10000)
    gene_models = pr.read_gtf(f'gene_models/{gtf}', as_df=True)
    gene_models = gene_models[gene_models['Feature'] == 'gene']
    gene_models = gene_models[gene_models['gene_biotype'] == 'protein_coding']
    gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    if isinstance(val_chrom, list):
        gene_models = gene_models[gene_models['Chromosome'].isin(val_chrom)]
    else:
        gene_models = gene_models[gene_models['Chromosome'].isin([val_chrom])]
    if pkey:
        # Pickled validation IDs
        with open('validation_genes.pickle', 'rb') as handle:
            validation_genes = pickle.load(handle)
        gene_models = gene_models[gene_models['gene_id'].isin(validation_genes[pkey])]

    # Transcripts per Million
    tpm_counts = pd.read_csv(f'tpm_counts/{tpms}', index_col=0)
    true_targets = []

    for log_count in tpm_counts['logMaxTPM'].values:
        if log_count <= np.percentile(tpm_counts['logMaxTPM'], 25):
            true_targets.append(0)
        elif log_count >= np.percentile(tpm_counts['logMaxTPM'], 75):
            true_targets.append(1)
        else:
            true_targets.append(2)
    tpm_counts['true_target'] = true_targets

    encoded_val_seqs, labels, gene_ids = [], [], []
    for chrom, start, end, strand, gene_id in gene_models.values:
        if strand == '+':
            prom_start, prom_end = start - upstream, start + downstream
            term_start, term_end = end - downstream, end + upstream
            if prom_start > 0 and term_start > 0:
                encoded_seq = np.concatenate([onehot(fasta[chrom][prom_start:prom_end]),
                                              np.zeros(shape=(20, 4)),
                                              onehot(fasta[chrom][term_start:term_end])])
                if encoded_seq.shape[0] == 2 * (upstream + downstream) + 20:
                    encoded_val_seqs.append(encoded_seq)
                    labels.append(tpm_counts.loc[gene_id, 'true_target'])
                    gene_ids.append(gene_id)

        else:
            prom_start, prom_end = end - downstream, end + upstream
            term_start, term_end = start - upstream, start + downstream
            if prom_start > 0 and term_start > 0:
                encoded_seq = np.concatenate([onehot(fasta[chrom][prom_start:prom_end])[::-1, ::-1],
                                              np.zeros(shape=(20, 4)),
                                              onehot(fasta[chrom][term_start:term_end])[::-1, ::-1]])

                if encoded_seq.shape[0] == 2 * (upstream + downstream) + 20:
                    encoded_val_seqs.append(encoded_seq)
                    labels.append(tpm_counts.loc[gene_id, 'true_target'])
                    gene_ids.append(gene_id)

    # Selecting validation sequences with label 1 and 0
    labels, encoded_val_seqs, gene_ids = np.array(labels), np.array(encoded_val_seqs), np.array(gene_ids)
    low_val, high_val = np.where(labels == 0)[0], np.where(labels == 1)[0]
    x_val = np.concatenate([
        np.take(encoded_val_seqs, low_val, axis=0),
        np.take(encoded_val_seqs, high_val, axis=0)
    ])
    y_val = np.concatenate([
        np.take(labels, low_val, axis=0),
        np.take(labels, high_val, axis=0)
    ])
    # Add debug info before indexing
    print(f"Shape of x_val before indexing: {x_val.shape}")
    print(f"Data type of x_val: {x_val.dtype}")
    print(f"Number of dimensions in x_val: {x_val.ndim}")
    gene_ids = np.concatenate([np.take(gene_ids, low_val, axis=0), np.take(gene_ids, high_val, axis=0)])
    x_val[:, upstream:upstream+3, :] = 0
    x_val[:, upstream+(downstream*2)+17:upstream+(downstream*2)+20, :] = 0
    return x_val, y_val, gene_ids

def plot_importance_analysis(all_importance_scores, mean_importance,
                             nucleotide_mean_importance, upstream, downstream):
    """Plot importance analysis figures"""
    plt.figure(figsize=(20, 16))

    # 1. Average importance score
    plt.subplot(4, 1, 1)
    plt.plot(mean_importance)
    plt.axvline(x=upstream, color='r', linestyle='--', label='TSS')
    plt.axvline(x=len(mean_importance) - upstream, color='g', linestyle='--', label='TTS')
    plt.title('Average Sequence Importance Score')
    plt.xlabel('Position')
    plt.ylabel('Importance Score')
    plt.legend()
    plt.grid(True)

    # 2. Importance score heatmap
    plt.subplot(4, 1, 2)
    sns.heatmap(all_importance_scores[:min(100, len(all_importance_scores))],
                cmap='coolwarm',
                center=0,
                cbar_kws={'label': 'Importance Score'})
    plt.axvline(x=upstream, color='white', linestyle='--')
    plt.axvline(x=len(mean_importance) - upstream, color='white', linestyle='--')
    plt.title('Importance Scores Heatmap (First 100 Sequences)')
    plt.xlabel('Position')
    plt.ylabel('Sequence Index')

    # 3. Cumulative importance
    plt.subplot(4, 1, 3)
    cumulative_importance = np.cumsum(np.abs(mean_importance))
    cumulative_importance = cumulative_importance / cumulative_importance[-1]
    plt.plot(cumulative_importance)
    plt.axvline(x=upstream, color='r', linestyle='--', label='TSS')
    plt.axvline(x=len(mean_importance) - upstream, color='g', linestyle='--', label='TTS')
    plt.title('Cumulative Absolute Importance')
    plt.xlabel('Position')
    plt.ylabel('Cumulative Importance')
    plt.legend()
    plt.grid(True)

    # 4. Nucleotide importance
    plt.subplot(4, 1, 4)
    colors = {'A': 'red', 'T': 'green', 'C': 'blue', 'G': 'purple'}
    for nucleotide, importance in nucleotide_mean_importance.items():
        plt.plot(importance, label=nucleotide, color=colors[nucleotide])
    plt.axvline(x=upstream, color='black', linestyle='--', label='TSS/TTS')
    plt.axvline(x=len(mean_importance) - upstream, color='black', linestyle='--')
    plt.title('Nucleotide Level Importance')
    plt.xlabel('Position')
    plt.ylabel('Importance Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def analyze_important_regions(mean_importance, upstream, downstream):
    """Analyze important sequence regions"""
    # Use Z-score to identify significant regions
    z_scores = (mean_importance - np.mean(mean_importance)) / np.std(mean_importance)
    significant_positions = np.where(z_scores > 2)[0]  # Select positions with Z-score > 2

    if len(significant_positions) == 0:
        print("No significantly important regions found")
        return None

    # Group consecutive important positions into regions
    regions = []
    current_region = [significant_positions[0]]

    for pos in significant_positions[1:]:
        if pos - current_region[-1] <= 10:  # If gap is less than 10bp, consider same region
            current_region.append(pos)
        else:
            if len(current_region) >= 5:  # Only keep regions >= 5bp
                regions.append(current_region)
            current_region = [pos]

    if len(current_region) >= 5:
        regions.append(current_region)

    # Organize region information
    region_info = []
    for i, region in enumerate(regions):
        region_start = region[0]
        region_end = region[-1]
        mean_score = np.mean(mean_importance[region_start:region_end + 1])

        # Determine region location (relative to TSS/TTS)
        if region_start < upstream:
            location = f"{upstream - region_start}bp upstream of TSS"
        elif region_start > len(mean_importance) - upstream:
            location = f"{region_start - (len(mean_importance) - upstream)}bp downstream of TTS"
        else:
            if region_start < len(mean_importance) / 2:
                location = f"{region_start - upstream}bp downstream of TSS"
            else:
                location = f"{len(mean_importance) - upstream - region_start}bp upstream of TTS"

        region_info.append({
            'Region': f'Region {i + 1}',
            'Start': region_start,
            'End': region_end,
            'Length': len(region),
            'Mean Score': mean_score,
            'Location': location
        })

    regions_df = pd.DataFrame(region_info)
    print("\nSummary of important sequence regions:")
    print(regions_df.to_string(index=False))

    return regions_df

def analyze_model_predictions(model_path, fasta_path, gtf_path, tpm_path, val_chrom,
                              upstream=10000, downstream=500, batch_size=16, num_steps=50,
                              pkey='sol', save_modisco_data=True):
    """
    Analyze sequence and nucleotide importance from model predictions and prepare data for TF-MoDISco
    """
    print("Loading data...")
    model = tf.keras.models.load_model(model_path)

    print("Preparing validation sequences...")
    x_val, y_val, gene_ids = prepare_valid_seqs(
        fasta=fasta_path,
        gtf=gtf_path,
        tpms=tpm_path,
        val_chrom=val_chrom,
        pkey=pkey,
        upstream=upstream,
        downstream=downstream
    )
    print(f"x_val shape: {x_val.shape}")

    print(f"Total of {len(x_val)} validation sequences loaded")
    print(f"Sequence shape: {x_val.shape}")
    print(f"Label distribution:\n{pd.Series(y_val).value_counts()}")

    print("\nComputing sequence importance scores...")
    all_importance_scores = []
    all_nucleotide_importance = {
        'A': np.zeros((len(x_val), x_val.shape[1])),
        'T': np.zeros((len(x_val), x_val.shape[1])),
        'C': np.zeros((len(x_val), x_val.shape[1])),
        'G': np.zeros((len(x_val), x_val.shape[1]))
    }

    contrib_scores = np.zeros_like(x_val, dtype=float)
    hypothetical_scores = np.zeros_like(x_val, dtype=float)

    for i in tqdm(range(0, len(x_val), batch_size)):
        batch_sequences = x_val[i:i + batch_size]
        batch_size_actual = len(batch_sequences)

        batch_sequences = tf.convert_to_tensor(batch_sequences, dtype=tf.float32)
        baseline = tf.zeros_like(batch_sequences)
        alphas = tf.linspace(0.0, 1.0, num_steps)

        batch_scores = []
        for alpha in alphas:
            interpolated = baseline + alpha * (batch_sequences - baseline)
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                predictions = model(interpolated)
            gradients = tape.gradient(predictions, interpolated)
            batch_scores.append(gradients)

        avg_gradients = tf.reduce_mean(batch_scores, axis=0)
        importance = (batch_sequences - baseline) * avg_gradients
        sequence_importance = tf.reduce_sum(importance, axis=-1)

        all_importance_scores.extend(sequence_importance.numpy())
        contrib_scores[i:i + batch_size_actual] = importance.numpy()

        # Compute hypothetical contribution scores (can be adjusted as needed)
        hypothetical_scores[i:i + batch_size_actual] = np.abs(importance.numpy())

        for nucleotide in ['A', 'T', 'C', 'G']:
            nucleotide_position = {'A': 0, 'T': 1, 'C': 2, 'G': 3}[nucleotide]
            nucleotide_mask = tf.cast(
                tf.equal(tf.argmax(batch_sequences, axis=-1), nucleotide_position),
                tf.float32
            )
            nucleotide_importance = tf.reduce_sum(importance * tf.expand_dims(nucleotide_mask, -1), axis=-1)
            all_nucleotide_importance[nucleotide][i:i + batch_size_actual] = nucleotide_importance.numpy()

    all_importance_scores = np.array(all_importance_scores)
    mean_importance = np.mean(all_importance_scores, axis=0)

    nucleotide_mean_importance = {
        nucleotide: np.mean(scores, axis=0)
        for nucleotide, scores in all_nucleotide_importance.items()
    }

    # Visualize results
    plot_importance_analysis(
        all_importance_scores,
        mean_importance,
        nucleotide_mean_importance,
        upstream,
        downstream
    )

    # Analyze important regions
    regions_df = analyze_important_regions(mean_importance, upstream, downstream)

    # Save data required for TF-MoDISco
    if save_modisco_data:
        os.makedirs('modisco', exist_ok=True)
        with h5py.File(f'modisco/{val_chrom}_scores.h5', 'w') as h5_data:
            h5_data.create_dataset('contrib_scores', data=contrib_scores)
            h5_data.create_dataset('hypothetical_scores', data=hypothetical_scores)
            h5_data.create_dataset('one_hots', data=x_val)

    return (all_importance_scores, mean_importance,
            nucleotide_mean_importance, regions_df, gene_ids,
            contrib_scores, hypothetical_scores, x_val)


# 2. Add error handling in main function
def main():
    try:
        # Set parameters
        model_path = r''
        fasta_path = ''
        gtf_path = ''
        tpm_path = ''
        val_chrom = ''

        # Run sequence importance analysis
        results = analyze_model_predictions(
            model_path=model_path,
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            tpm_path=tpm_path,
            val_chrom=val_chrom
        )

        (importance_scores, mean_importance,
         nucleotide_mean_importance, regions_df,
         gene_ids, contrib_scores,
         hypothetical_scores, one_hots) = results

        # Save results
        np.save('importance_scores5.npy', importance_scores)
        np.save('mean_importance5.npy', mean_importance)

        for nucleotide, importance in nucleotide_mean_importance.items():
            np.save(f'{nucleotide}_importance5.npy', importance)

        if regions_df is not None:
            regions_df.to_csv('important_regions5.csv', index=False)

        print("\nAnalysis complete! Results have been saved to files.")

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
