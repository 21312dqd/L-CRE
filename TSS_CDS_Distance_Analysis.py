import pandas as pd
import pyranges as pr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File path settings
gene_list_csv = r"tpm_counts/solanum_counts.csv"  # CSV file containing gene IDs
gtf_file = "gene_models/Solanum_lycopersicum.SL3.0.52.gtf"  # Genome annotation GTF file
output_file = "tss_cds_distance_resultsH.csv"  # Output results file
output_plot = "tss_cds_distance_plotH.svg"  # Visualization output file

# Read gene list
genes_df = pd.read_csv(gene_list_csv)
gene_id_col = genes_df.columns[0]  # Get the name of the first column
gene_ids = set(genes_df[gene_id_col].tolist())

print(f"Total {len(gene_ids)} gene IDs read")

# Read GTF file
gtf = pr.read_gtf(gtf_file)

# Extract gene and CDS information
genes = gtf[(gtf.Feature == 'gene') & (gtf.gene_id.isin(gene_ids))].df
cds = gtf[gtf.Feature == 'CDS'].df

print(f"Found {len(genes)} matching genes in the GTF file")

# Calculate distance between TSS and nearest CDS
results = []
count_within_500bp = 0

for _, gene in genes.iterrows():
    gene_id = gene['gene_id']
    chrom = gene['Chromosome']
    strand = gene['Strand']

    # Determine TSS position (based on strand direction)
    if strand == '+':
        tss = gene['Start']
    else:  # strand == '-'
        tss = gene['End']

    # Get all CDS for this gene
    gene_cds = cds[cds.gene_id == gene_id]

    if len(gene_cds) == 0:
        # No CDS for this gene
        min_distance = np.nan
    else:
        # Calculate distance from TSS to each CDS
        if strand == '+':
            # Positive strand: Distance from TSS to CDS start
            distances = gene_cds['Start'] - tss
        else:
            # Negative strand: Distance from TSS to CDS end
            distances = tss - gene_cds['End']

        # Get minimum distance (closest CDS)
        min_distance = min(distances)

    results.append({
        'gene_id': gene_id,
        'chromosome': chrom,
        'strand': strand,
        'tss': tss,
        'min_distance_to_cds': min_distance
    })

    # Count genes with distance < 500bp
    if not np.isnan(min_distance) and min_distance < 500:
        count_within_500bp += 1

# Create results DataFrame
results_df = pd.DataFrame(results)

# Calculate percentage
total_genes_with_cds = len(results_df.dropna(subset=['min_distance_to_cds']))
percentage_within_500bp = (count_within_500bp / total_genes_with_cds) * 100 if total_genes_with_cds > 0 else 0

# Save results
results_df.to_csv(output_file, index=False)

# Print results
print("\nStatistics:")
print(f"Total genes: {len(results_df)}")
print(f"Genes with CDS: {total_genes_with_cds}")
print(f"Genes with TSS-CDS distance < 500bp: {count_within_500bp}")
print(f"Percentage of genes with CDS: {percentage_within_500bp:.2f}%")
print(f"Detailed results saved to: {output_file}")

# Set global font size
plt.rcParams.update({'font.size': 14})  # Increase global font size

# Create plot
plt.figure(figsize=(12, 8))  # Increase figure size for larger fonts
# Set style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)  # Increase font size with font_scale

# Filter valid data
valid_distances = results_df['min_distance_to_cds'].dropna()
# Remove outliers for better visualization
distance_cutoff = np.percentile(valid_distances, 95)  # Use 95th percentile as upper limit
filtered_distances = valid_distances[valid_distances <= distance_cutoff]

# Create histogram
ax = sns.histplot(filtered_distances, bins=30, kde=True, color='steelblue')

# Add vertical line at 500bp
plt.axvline(x=500, color='red', linestyle='--',
            label=f'500bp threshold ({percentage_within_500bp:.1f}%)')

# Set plot labels and title
plt.xlabel('Length (bp)', fontsize=22)  # Increase x-axis label font size
plt.ylabel('Counts', fontsize=22)      # Increase y-axis label font size
plt.title('Distribution of TSS to CDS Distances', fontsize=24)  # Add title with larger font

# Increase legend font size
plt.legend(fontsize=18)

# Increase tick label font size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Save as SVG file
plt.tight_layout()
plt.savefig(output_plot, format='svg')
print(f"Visualization chart has been saved to: {output_plot}")

# Display plot
plt.show()