import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
from scipy import stats

# Attempt to import pearsonr
try:
    from scipy.stats import pearsonr
    USE_SCIPY = True
except ImportError:
    USE_SCIPY = False
    print("Warning: scipy not installed, using custom Pearson correlation calculation")

    def pearsonr(x, y):
        """Custom Pearson correlation coefficient calculation (with p-value)"""
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)

        if n < 3:
            return 0, 1

        mx = x.mean()
        my = y.mean()
        xm = x - mx
        ym = y - my
        r_num = np.sum(xm * ym)
        r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))

        if r_den == 0:
            return 0, 1

        r = r_num / r_den
        r = max(min(r, 1.0), -1.0)

        # Calculate p-value
        t = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2 + 1e-10)
        p = 2 * (1 - stats.t.cdf(abs(t), n - 2))

        return r, p

# Function: Parse MEME file and extract motif matrices
def parse_meme_file(file_path):
    """Parse MEME file, return list of motifs"""
    motifs = []
    current_motif = None
    in_matrix = False
    matrix_data = []

    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('MOTIF'):
            if current_motif and matrix_data:
                motifs.append((current_motif['name'], np.array(matrix_data).T))
            current_motif = {'name': line.split()[1]}
            matrix_data = []
            in_matrix = False
        elif 'letter-probability matrix' in line:
            w = int(re.search(r'w=\s*(\d+)', line).group(1))
            current_motif['w'] = w
            in_matrix = True
        elif in_matrix and line and not line.startswith('URL'):
            probs = [float(x) for x in line.split()]
            if len(probs) == 4:
                matrix_data.append(probs)

    if current_motif and matrix_data:
        motifs.append((current_motif['name'], np.array(matrix_data).T))

    return motifs

# Function: Extract file prefix (e.g., 82flower, Mleaf)
def extract_file_prefix(filename):
    """Extract prefix from filename"""
    # Remove .meme suffix
    basename = os.path.basename(filename)
    if basename.endswith('.meme'):
        return basename[:-5]
    return basename

# Function: Format motif name
def format_motif_name(file_prefix, motif_name):
    """Format motif name, e.g., 82flowerM10"""
    # Extract number from motif_name
    match = re.search(r'(\d+)', motif_name)
    if match:
        motif_num = match.group(1)
        return f"{file_prefix}M{motif_num}"
    return f"{file_prefix}_{motif_name}"

# Function: Compute all correlations and return multiple matches
def compute_multi_matches_with_pvalue(your_motifs, official_motifs, file_prefix, match_threshold=0.5):
    """Compute all matches above threshold for each motif (allowing multiple matches)"""
    motif_results = []

    for your_name, your_matrix in your_motifs:
        # Store all matches
        all_matches = []

        for official_name, official_matrix in official_motifs:
            L = official_matrix.shape[1]
            num_subseq = your_matrix.shape[1] - L + 1
            if num_subseq <= 0:
                continue

            best_r_for_this_official = -1
            best_p_for_this_official = 1
            best_start_for_this_official = 0

            # Find best match position for this official motif
            for start in range(num_subseq):
                subseq = your_matrix[:, start:start + L]
                x = subseq.flatten()
                y = official_matrix.flatten()
                r, p = pearsonr(x, y)

                if r > best_r_for_this_official:
                    best_r_for_this_official = r
                    best_p_for_this_official = p
                    best_start_for_this_official = start

            # Record if correlation exceeds threshold
            if best_r_for_this_official >= match_threshold:
                all_matches.append({
                    'official_name': official_name,
                    'pearson_r': best_r_for_this_official,
                    'p_value': best_p_for_this_official,
                    'match_position': best_start_for_this_official
                })

        # Sort matches by Pearson correlation in descending order
        all_matches.sort(key=lambda x: x['pearson_r'], reverse=True)

        # Format motif name
        formatted_name = format_motif_name(file_prefix, your_name)

        # If matches exist
        if all_matches:
            best_r = all_matches[0]['pearson_r']

            # Determine category (based on best match)
            if best_r > 0.7:
                category = "High Match"
            elif best_r > 0.5:
                category = "Moderate Match"
            else:
                category = "Potential Novel Motif"

            # Create record for each match
            for idx, match in enumerate(all_matches):
                motif_results.append({
                    'motif_id': formatted_name,
                    'original_name': your_name,
                    'file_name': file_prefix,
                    'category': category,
                    'match_rank': idx + 1,  # Match rank
                    'total_matches': len(all_matches),  # Total matches
                    'best_match': match['official_name'],
                    'pearson_r': match['pearson_r'],
                    'p_value': match['p_value'],
                    'match_position': match['match_position']
                })
        else:
            # No matches found
            motif_results.append({
                'motif_id': formatted_name,
                'original_name': your_name,
                'file_name': file_prefix,
                'category': "Potential Novel Motif",
                'match_rank': 0,
                'total_matches': 0,
                'best_match': 'N/A',
                'pearson_r': 0,
                'p_value': 1,
                'match_position': 'N/A'
            })

    return motif_results

# Function: Analyze a single MEME file
def analyze_single_file(meme_file, official_motifs, match_threshold=0.5):
    """Analyze a single MEME file"""
    print(f"  Analyzing: {os.path.basename(meme_file)}")

    file_prefix = extract_file_prefix(meme_file)
    your_motifs = parse_meme_file(meme_file)
    total_motifs = len(your_motifs)

    if total_motifs == 0:
        print(f"    Warning: No motifs found")
        return None

    # Compute all matches for each motif
    motif_results = compute_multi_matches_with_pvalue(your_motifs, official_motifs, file_prefix, match_threshold)

    # Count categories (each motif counted once, using best match category)
    unique_motifs = {}
    for m in motif_results:
        motif_id = m['motif_id']
        if motif_id not in unique_motifs or m['match_rank'] == 1:
            unique_motifs[motif_id] = m['category']

    matched_count = sum(1 for cat in unique_motifs.values() if cat == "High Match")
    weakly_count = sum(1 for cat in unique_motifs.values() if cat == "Moderate Match")
    novel_count = sum(1 for cat in unique_motifs.values() if cat == "Potential Novel Motif")

    return {
        'file_prefix': file_prefix,
        'total_motifs': total_motifs,
        'matched_count': matched_count,
        'weakly_count': weakly_count,
        'novel_count': novel_count,
        'motif_details': motif_results
    }

# Main function: Batch analysis
def batch_analyze_motifs(meme_files, official_meme_dir, output_excel,
                         high_threshold=0.7, moderate_threshold=0.5, match_threshold=0.5):
    """Batch analyze multiple MEME files and generate Excel report

    Parameters:
        match_threshold: Minimum correlation threshold for recording a match
    """

    print("=" * 80)
    print("Batch Motif Matching Analysis (Multi-Match Version)")
    print("=" * 80)

    # Read known motifs from database
    print("\n1. Reading MEME files from database...")
    official_motifs = []
    for filename in os.listdir(official_meme_dir):
        if filename.endswith('.meme'):
            file_path = os.path.join(official_meme_dir, filename)
            motifs = parse_meme_file(file_path)
            for name, matrix in motifs:
                official_motifs.append((f"{filename[:-5]}_{name}", matrix))
    print(f"   Read {len(official_motifs)} known motifs from database")

    # Analyze each file
    print(f"\n2. Analyzing {len(meme_files)} MEME files...")
    print(f"   Match threshold: r >= {match_threshold}")
    results = []
    all_motif_details = []

    for meme_file in meme_files:
        if os.path.exists(meme_file):
            result = analyze_single_file(meme_file, official_motifs, match_threshold)
            if result:
                results.append(result)
                all_motif_details.extend(result['motif_details'])
        else:
            print(f"  Warning: File not found - {meme_file}")

    if not results:
        print("Error: No files successfully analyzed")
        return

    # Generate Excel report
    print(f"\n3. Generating Excel report...")
    generate_detailed_excel_report(results, all_motif_details, output_excel,
                                  high_threshold, moderate_threshold, len(official_motifs), match_threshold)

    print(f"\n✓ Analysis complete! Results saved to: {output_excel}")
    print("=" * 80)

# Function: Generate detailed Excel report
def generate_detailed_excel_report(results, all_motif_details, output_file,
                                  high_threshold, moderate_threshold, official_motifs_count, match_threshold):
    """Generate detailed Excel report"""

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # Sheet 1: All motif details (including all matches)
        df_all_motifs = pd.DataFrame(all_motif_details)
        df_all_motifs = df_all_motifs[[
            'motif_id', 'file_name', 'original_name', 'category', 'match_rank', 'total_matches',
            'best_match', 'pearson_r', 'p_value', 'match_position'
        ]]
        df_all_motifs.columns = [
            'Motif Name', 'Source File', 'Original Name', 'Category', 'Match Rank', 'Total Matches',
            'Matched Motif', 'Pearson Coefficient', 'P-value', 'Match Position'
        ]
        # Format values
        df_all_motifs['Pearson Coefficient'] = df_all_motifs['Pearson Coefficient'].apply(lambda x: f"{x:.4f}")
        df_all_motifs['P-value'] = df_all_motifs['P-value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")

        df_all_motifs.to_excel(writer, sheet_name='All Motif Details', index=False)

        # Sheet 2: Best match only for each motif (rank 1 or 0)
        best_matches = [m for m in all_motif_details if m['match_rank'] == 1 or m['match_rank'] == 0]
        df_best = pd.DataFrame(best_matches)
        df_best = df_best[[
            'motif_id', 'file_name', 'original_name', 'category', 'total_matches',
            'best_match', 'pearson_r', 'p_value', 'match_position'
        ]]
        df_best.columns = [
            'Motif Name', 'Source File', 'Original Name', 'Category', 'Total Matches',
            'Best Matched Motif', 'Pearson Coefficient', 'P-value', 'Match Position'
        ]
        df_best['Pearson Coefficient'] = df_best['Pearson Coefficient'].apply(lambda x: f"{x:.4f}")
        df_best['P-value'] = df_best['P-value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
        df_best.to_excel(writer, sheet_name='Best Match Summary', index=False)

        # Sheet 3: Summary statistics by file
        summary_data = []
        for result in results:
            total = result['total_motifs']
            matched = result['matched_count']
            weakly = result['weakly_count']
            novel = result['novel_count']

            # Calculate total matches for this file
            file_matches = [m for m in all_motif_details if m['file_name'] == result['file_prefix']]
            total_match_count = len(file_matches)

            summary_data.append({
                'File Name': result['file_prefix'],
                'Total Motifs': total,
                'Total Match Records': total_match_count,
                f'High Matches (r>{high_threshold})': matched,
                'High Match Proportion': f"{matched / total * 100:.1f}%",
                f'Moderate Matches ({moderate_threshold}<r≤{high_threshold})': weakly,
                'Moderate Match Proportion': f"{weakly / total * 100:.1f}%",
                f'Potential Novel Motifs (r≤{moderate_threshold})': novel,
                'Potential Novel Proportion': f"{novel / total * 100:.1f}%"
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary by File', index=False)

        # Sheet 4: High match motifs (all match records)
        high_matches = [m for m in all_motif_details if m['category'] == "High Match"]
        if high_matches:
            df_high = pd.DataFrame(high_matches)
            df_high = df_high[[
                'motif_id', 'file_name', 'match_rank', 'best_match', 'pearson_r', 'p_value'
            ]]
            df_high.columns = ['Motif Name', 'Source File', 'Match Rank', 'Matched Motif', 'Pearson Coefficient', 'P-value']
            df_high['Pearson Coefficient'] = df_high['Pearson Coefficient'].apply(lambda x: f"{x:.4f}")
            df_high['P-value'] = df_high['P-value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
            df_high.to_excel(writer, sheet_name='High Match Motifs', index=False)

        # Sheet 5: Potential novel motifs
        novel_motifs = [m for m in all_motif_details if
                        m['category'] == "Potential Novel Motif" and (m['match_rank'] == 1 or m['match_rank'] == 0)]
        if novel_motifs:
            df_novel = pd.DataFrame(novel_motifs)
            df_novel = df_novel[[
                'motif_id', 'file_name', 'best_match', 'pearson_r', 'p_value'
            ]]
            df_novel.columns = ['Motif Name', 'Source File', 'Closest Motif', 'Pearson Coefficient', 'P-value']
            df_novel['Pearson Coefficient'] = df_novel['Pearson Coefficient'].apply(lambda x: f"{x:.4f}")
            df_novel['P-value'] = df_novel['P-value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
            df_novel.to_excel(writer, sheet_name='Potential Novel Motifs', index=False)

        # Sheet 6: Moderate match motifs (all match records)
        weak_matches = [m for m in all_motif_details if m['category'] == "Moderate Match"]
        if weak_matches:
            df_weak = pd.DataFrame(weak_matches)
            df_weak = df_weak[[
                'motif_id', 'file_name', 'match_rank', 'best_match', 'pearson_r', 'p_value'
            ]]
            df_weak.columns = ['Motif Name', 'Source File', 'Match Rank', 'Matched Motif', 'Pearson Coefficient', 'P-value']
            df_weak['Pearson Coefficient'] = df_weak['Pearson Coefficient'].apply(lambda x: f"{x:.4f}")
            df_weak['P-value'] = df_weak['P-value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
            df_weak.to_excel(writer, sheet_name='Moderate Match Motifs', index=False)

        # Sheet 7: Multi-match statistics (motifs with multiple matches)
        multi_match_motifs = [m for m in all_motif_details if m['total_matches'] > 1 and m['match_rank'] == 1]
        if multi_match_motifs:
            df_multi = pd.DataFrame(multi_match_motifs)
            df_multi = df_multi[[
                'motif_id', 'file_name', 'category', 'total_matches', 'best_match', 'pearson_r'
            ]]
            df_multi.columns = ['Motif Name', 'Source File', 'Category', 'Match Count', 'Best Match', 'Highest Pearson Coefficient']
            df_multi['Highest Pearson Coefficient'] = df_multi['Highest Pearson Coefficient'].apply(lambda x: f"{x:.4f}")
            df_multi = df_multi.sort_values('Match Count', ascending=False)
            df_multi.to_excel(writer, sheet_name='Multi-Match Motif Statistics', index=False)

        # Sheet 8: Overall statistics overview
        total_motifs = sum(r['total_motifs'] for r in results)
        total_matched = sum(r['matched_count'] for r in results)
        total_weakly = sum(r['weakly_count'] for r in results)
        total_novel = sum(r['novel_count'] for r in results)
        total_match_records = len(all_motif_details)

        overview_data = {
            'Statistic Item': [
                'Analysis Date',
                'Number of Files Analyzed',
                'Number of Known Motifs in Database',
                'Match Threshold',
                '',
                'Total Predicted Motifs',
                'Total Match Records',
                'Average Matches per Motif',
                '',
                f'High Match Motifs (r > {high_threshold})',
                'High Match Proportion',
                f'Moderate Match Motifs ({moderate_threshold} < r ≤ {high_threshold})',
                'Moderate Match Proportion',
                f'Potential Novel Motifs (r ≤ {moderate_threshold})',
                'Potential Novel Proportion',
                '',
                'P-value Explanation',
                'Significance Level (α=0.05)',
                'Highly Significant (p<0.001)',
                'Moderately Significant (0.001≤p<0.05)',
                'Not Significant (p≥0.05)'
            ],
            'Value/Description': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f"{len(results)} files",
                f"{official_motifs_count} motifs",
                f"r >= {match_threshold}",
                '',
                f"{total_motifs} motifs",
                f"{total_match_records} records",
                f"{total_match_records / total_motifs:.1f}",
                '',
                f"{total_matched} motifs",
                f"{total_matched / total_motifs * 100:.1f}%",
                f"{total_weakly} motifs",
                f"{total_weakly / total_motifs * 100:.1f}%",
                f"{total_novel} motifs",
                f"{total_novel / total_motifs * 100:.1f}%",
                '',
                '',
                'Statistical significance test for correlation coefficient',
                '*** (Highly Significant)',
                '* or ** (Significant)',
                'ns (Not Significant)'
            ]
        }

        df_overview = pd.DataFrame(overview_data)
        df_overview.to_excel(writer, sheet_name='Overall Statistics Overview', index=False)

        # Sheet 9: Analysis Explanation
        explanation_data = {
            'Explanation Item': [
                'Analysis Purpose',
                'Matching Method',
                'Multi-Match Explanation',
                '',
                'Classification Criteria',
                'High Match',
                'Moderate Match',
                'Potential Novel Motif',
                '',
                'Motif Naming Convention',
                '',
                'Pearson Correlation Coefficient',
                'P-value',
                'Match Rank',
                '',
                'Recommendations',
                'High Match Motifs',
                'Multi-Match Motifs',
                'Potential Novel Motifs'
            ],
            'Detailed Description': [
                'Compare predicted EPM motifs with known database motifs to identify new regulatory elements',
                'Use Pearson correlation coefficient to measure motif similarity',
                f'Each motif can match multiple database motifs (r >= {match_threshold}), sorted by correlation in descending order',
                '',
                f'High Match: r > {high_threshold}; Moderate Match: {moderate_threshold} < r ≤ {high_threshold}; Potential Novel Motif: r ≤ {moderate_threshold}',
                'Highly similar to known database motifs, likely representing known transcription factor binding sites',
                'Somewhat similar to known motifs, possibly variants or partial matches of known motifs',
                'Low correlation with known motifs, potentially novel regulatory elements, species-specific motifs, or condition-specific motifs',
                '',
                'Format: [file prefix]M[number], e.g., 82flowerM10 indicates the 10th motif in 82flower.meme file',
                '',
                'Measures similarity between two motifs, ranging from -1 to 1, higher values indicate greater similarity',
                'Tests statistical significance of correlation, p<0.05 indicates significant correlation',
                'For motifs with multiple matches, rank 1 indicates the highest correlation match',
                '',
                '',
                'These motifs match known TF binding sites; refer to database annotations for functional studies',
                'A motif matching multiple database motifs may indicate: (1) a common binding site for multiple TFs, or (2) similar binding preferences among different TFs',
                'Recommendations: (1) Experimentally validate biological function, (2) Analyze genomic distribution, (3) Compare with additional databases, (4) Check for complex patterns'
            ]
        }

        df_explanation = pd.DataFrame(explanation_data)
        df_explanation.to_excel(writer, sheet_name='Analysis Explanation', index=False)

# Main program entry
if __name__ == "__main__":
    # Configuration parameters
    meme_files = [
        "82flower.meme",
        "82fruit.meme",
        "82leaf.meme",
        "82root.meme",
        "Hflower.meme",
        "Hleaf.meme",
        "Hroot.meme",
        "Hfruit.meme",
        "Mflower.meme",
        "Mleaf.meme",
        "Mroot.meme",
        "Mfruit.meme"
    ]

    # Official MEME files directory (database)
    official_meme_dir = "official_meme_files"

    # Output Excel file path
    output_excel = "motif_multi_match_analysis_report.xlsx"

    # Threshold settings
    high_threshold = 0.7  # High match threshold
    moderate_threshold = 0.5  # Moderate match threshold
    match_threshold = 0.5  # Minimum threshold for recording matches

    # Run batch analysis
    batch_analyze_motifs(
        meme_files=meme_files,
        official_meme_dir=official_meme_dir,
        output_excel=output_excel,
        high_threshold=high_threshold,
        moderate_threshold=moderate_threshold,
        match_threshold=match_threshold  # Adjustable, e.g., set to 0.3 or 0.4
    )