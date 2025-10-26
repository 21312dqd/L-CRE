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
# 确保在主程序开始时就设置多进程方法
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
    # 在进行索引之前，添加以下调试信息
    print(f"准备进行索引操作前的 x_val 形状: {x_val.shape}")
    print(f"x_val 的数据类型: {x_val.dtype}")
    print(f"x_val 的维度数: {x_val.ndim}")
    gene_ids = np.concatenate([np.take(gene_ids, low_val, axis=0), np.take(gene_ids, high_val, axis=0)])
    x_val[:, upstream:upstream+3, :] = 0
    x_val[:, upstream+(downstream*2)+17:upstream+(downstream*2)+20, :] = 0
    return x_val, y_val, gene_ids

def plot_importance_analysis(all_importance_scores, mean_importance,
                             nucleotide_mean_importance, upstream, downstream):
    """绘制重要性分析图"""
    plt.figure(figsize=(20, 16))

    # 1. 平均重要性得分
    plt.subplot(4, 1, 1)
    plt.plot(mean_importance)
    plt.axvline(x=upstream, color='r', linestyle='--', label='TSS')
    plt.axvline(x=len(mean_importance) - upstream, color='g', linestyle='--', label='TTS')
    plt.title('Average Sequence Importance Score')
    plt.xlabel('Position')
    plt.ylabel('Importance Score')
    plt.legend()
    plt.grid(True)

    # 2. 重要性得分热图
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

    # 3. 累积重要性
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

    # 4. 核苷酸重要性
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
    """分析重要序列区域"""
    # 使用Z-score标准化找出显著区域
    z_scores = (mean_importance - np.mean(mean_importance)) / np.std(mean_importance)
    significant_positions = np.where(z_scores > 2)[0]  # 选择Z-score > 2的位置

    if len(significant_positions) == 0:
        print("未发现显著重要的区域")
        return None

    # 将连续的重要位置分组为区域
    regions = []
    current_region = [significant_positions[0]]

    for pos in significant_positions[1:]:
        if pos - current_region[-1] <= 10:  # 如果位置间隔小于10bp，认为是同一区域
            current_region.append(pos)
        else:
            if len(current_region) >= 5:  # 只保留长度≥5bp的区域
                regions.append(current_region)
            current_region = [pos]

    if len(current_region) >= 5:
        regions.append(current_region)

    # 整理区域信息
    region_info = []
    for i, region in enumerate(regions):
        region_start = region[0]
        region_end = region[-1]
        mean_score = np.mean(mean_importance[region_start:region_end + 1])

        # 确定区域位置（相对于TSS/TTS）
        if region_start < upstream:
            location = f"TSS上游{upstream - region_start}bp"
        elif region_start > len(mean_importance) - upstream:
            location = f"TTS下游{region_start - (len(mean_importance) - upstream)}bp"
        else:
            if region_start < len(mean_importance) / 2:
                location = f"TSS下游{region_start - upstream}bp"
            else:
                location = f"TTS上游{len(mean_importance) - upstream - region_start}bp"

        region_info.append({
            'Region': f'Region {i + 1}',
            'Start': region_start,
            'End': region_end,
            'Length': len(region),
            'Mean Score': mean_score,
            'Location': location
        })

    regions_df = pd.DataFrame(region_info)
    print("\n重要序列区域概要:")
    print(regions_df.to_string(index=False))

    return regions_df
def analyze_model_predictions(model_path, fasta_path, gtf_path, tpm_path, val_chrom,
                              upstream=10000, downstream=500, batch_size=16, num_steps=50,
                              pkey='sol', save_modisco_data=True):
    """
    分析模型预测的序列和核苷酸重要性，并准备TF-MoDISco分析
    """
    print("加载数据...")
    model = tf.keras.models.load_model(model_path)

    print("准备验证序列...")
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

    print(f"总共加载了 {len(x_val)} 个验证序列")
    print(f"序列形状: {x_val.shape}")
    print(f"标签分布:\n{pd.Series(y_val).value_counts()}")

    print("\n计算序列重要性得分...")
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

        # 计算假设贡献分数（可以根据需要调整）
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

    # 可视化结果
    plot_importance_analysis(
        all_importance_scores,
        mean_importance,
        nucleotide_mean_importance,
        upstream,
        downstream
    )

    # 分析重要区域
    regions_df = analyze_important_regions(mean_importance, upstream, downstream)

    # 保存TF-MoDISco所需数据
    if save_modisco_data:
        os.makedirs('modisco', exist_ok=True)
        with h5py.File(f'modisco/{val_chrom}_scores.h5', 'w') as h5_data:
            h5_data.create_dataset('contrib_scores', data=contrib_scores)
            h5_data.create_dataset('hypothetical_scores', data=hypothetical_scores)
            h5_data.create_dataset('one_hots', data=x_val)

    return (all_importance_scores, mean_importance,
            nucleotide_mean_importance, regions_df, gene_ids,
            contrib_scores, hypothetical_scores, x_val)




# 2. 主函数中添加错误处理
def main():
    try:
        # 设置参数
        model_path = r''
        fasta_path = ''
        gtf_path = ''
        tpm_path = ''
        val_chrom = ''

        # 运行序列重要性分析
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


        # 保存结果
        np.save('importance_scores5.npy', importance_scores)
        np.save('mean_importance5.npy', mean_importance)

        for nucleotide, importance in nucleotide_mean_importance.items():
            np.save(f'{nucleotide}_importance5.npy', importance)

        if regions_df is not None:
            regions_df.to_csv('important_regions5.csv', index=False)

        print("\n分析完成！结果已保存到文件中。")

    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()