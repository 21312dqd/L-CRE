import pandas as pd
import numpy as np
import os
from collections import defaultdict

# Gene ID mapping: (chromosome number, row index) -> gene ID
GENE_MAPPING = {

}

# Biologically significant regions (negative coordinates adjusted by adding 1000 to map to 0-999 indices)
BIOLOGICAL_REGIONS = {

}

def get_gene_id(i_value, row_idx):
    """
    Retrieve gene ID based on chromosome number and row index

    Parameters:
    i_value: Chromosome number
    row_idx: Row index

    Returns:
    Gene ID string, or empty string if not found
    """
    return GENE_MAPPING.get((i_value, row_idx), '')

def calculate_overlap(window_start, window_end, region_start, region_end):
    """
    Calculate the overlap length between two intervals

    Parameters:
    window_start, window_end: Window interval
    region_start, region_end: Biological region interval

    Returns:
    Overlap length
    """
    overlap_start = max(window_start, region_start)
    overlap_end = min(window_end, region_end)

    if overlap_start <= overlap_end:
        return overlap_end - overlap_start + 1
    else:
        return 0

def find_all_overlapping_regions(gene_id, window_start, window_end):
    """
    Find all biologically significant regions that overlap with the window

    Parameters:
    gene_id: Gene ID
    window_start, window_end: Window interval

    Returns:
    List of overlapping biological regions [(region, overlap_length), ...]
    """
    if gene_id not in BIOLOGICAL_REGIONS:
        return []

    overlapping_regions = []

    for region in BIOLOGICAL_REGIONS[gene_id]:
        region_start, region_end = region
        overlap = calculate_overlap(window_start, window_end, region_start, region_end)

        if overlap > 0:
            overlapping_regions.append((region, overlap))

    return overlapping_regions

def analyze_csv_differences(i_values=range(1, 13), window_size=50, top_n=15):
    """
    Analyze differences between outputHq0i.csv and outputCTRq0i.csv files for corresponding rows,
    calculate the first 1000 positions, and use a sliding window to find positions with the largest continuous differences.

    Parameters:
    i_values: Range of i values to analyze, default is 1 to 12 (chromosomes 1-11)
    window_size: Sliding window size, default is 50
    top_n: Calculate top N windows with largest differences for each gene, default is 15

    Returns:
    results: Dictionary containing maximum difference information for each i value and row
    """
    results = defaultdict(list)

    for i in i_values:
        # Construct filenames - adjust format for i>=10
        if i < 10:
            h_file = f"outputHq0{i}.csv"
            ctr_file = f"outputCTRq0{i}.csv"
        else:
            h_file = f"outputHq{i}.csv"
            ctr_file = f"outputCTRq{i}.csv"

        # Check if files exist
        if not os.path.exists(h_file) or not os.path.exists(ctr_file):
            print(f"Warning: File {h_file} or {ctr_file} does not exist, skipping i={i}")
            continue

        try:
            # Read CSV files
            h_data = pd.read_csv(h_file, header=None)
            ctr_data = pd.read_csv(ctr_file, header=None)

            # Ensure both files have the same number of rows
            min_rows = min(len(h_data), len(ctr_data))

            # Analyze each row
            for row_idx in range(min_rows):
                # Skip empty rows
                if h_data.iloc[row_idx].isnull().all() or ctr_data.iloc[row_idx].isnull().all():
                    continue

                # Get total number of columns
                total_cols = min(h_data.shape[1], ctr_data.shape[1])

                # Ensure enough columns for analysis
                if total_cols < 1000:
                    print(f"Warning: i={i}, row={row_idx} has fewer than 1000 columns ({total_cols}), using available columns")

                # Analyze first 1000 positions
                first_1000_cols = min(1000, total_cols)
                if first_1000_cols > window_size:
                    h_first = h_data.iloc[row_idx, :first_1000_cols].values
                    ctr_first = ctr_data.iloc[row_idx, :first_1000_cols].values

                    # Calculate differences
                    diff_first = h_first - ctr_first

                    # Analyze differences in the first 1000 positions
                    analyze_diff_section(
                        diff_first,
                        i,
                        row_idx,
                        window_size,
                        top_n,
                        results[i]
                    )

        except Exception as e:
            print(f"Error processing files for i={i}: {str(e)}")

    return results

def analyze_diff_section(diff, i, row_idx, window_size, top_n, result_list):
    """
    Analyze difference section, find top N windows with largest differences, keep all windows overlapping with biological regions

    Parameters:
    diff: Difference array
    i: Current i value being processed
    row_idx: Current row index being processed
    window_size: Sliding window size
    top_n: Calculate top N windows with largest differences
    result_list: List to store results
    """
    # Ensure difference array length is sufficient
    if len(diff) < window_size:
        print(f"Warning: i={i}, row={row_idx}, length ({len(diff)}) is less than window size ({window_size}), skipping")
        return

    # Get gene ID
    gene_id = get_gene_id(i, row_idx)

    # Calculate differences for all possible windows
    all_windows = []
    for start in range(len(diff) - window_size + 1):
        end = start + window_size - 1  # End is the last included position
        window_sum = np.sum(diff[start:start + window_size])
        avg_diff = window_sum / window_size

        all_windows.append({
            'start': start,
            'end': end,
            'total_diff': window_sum,
            'avg_diff': avg_diff
        })

    # Sort by total difference and select top N
    top_windows = sorted(all_windows, key=lambda x: x['total_diff'], reverse=True)[:top_n]

    # Find all windows that overlap with biological regions
    windows_with_overlap = []
    covered_regions = set()  # Track covered biological regions

    for rank, window in enumerate(top_windows, 1):
        # Get all biological regions overlapping with this window
        all_overlapping = find_all_overlapping_regions(gene_id, window['start'], window['end'])

        if all_overlapping:  # If there is overlap
            # Find the maximum overlap
            max_overlap = max(all_overlapping, key=lambda x: x[1])
            overlap_region, overlap_length = max_overlap
            overlap_pct = (overlap_length / window_size * 100)

            # Format all overlapping regions
            overlapping_regions_str = "; ".join([f"{r}" for r, _ in all_overlapping])

            windows_with_overlap.append({
                'row': row_idx,
                'gene_id': gene_id,
                'rank': rank,
                'window_start': window['start'],
                'window_end': window['end'],
                'window_size': window_size,
                'total_diff': window['total_diff'],
                'avg_diff': window['avg_diff'],
                'overlap_length': overlap_length,
                'max_overlap_region': f"({overlap_region[0]}, {overlap_region[1]})",
                'overlap_percentage': overlap_pct,
                'num_overlapping_regions': len(all_overlapping),
                'all_overlapping_regions': overlapping_regions_str
            })

            # Record covered regions
            for region, _ in all_overlapping:
                covered_regions.add(region)

    # Add all overlapping windows to results
    for window_info in windows_with_overlap:
        result_list.append(window_info)

        gene_info = f", gene={window_info['gene_id']}" if window_info['gene_id'] else ""
        print(f"i={i}, row={row_idx}{gene_info}, rank={window_info['rank']}: "
              f"window[{window_info['window_start']}:{window_info['window_end']}], "
              f"total_diff={window_info['total_diff']:.2f}, "
              f"covers {window_info['num_overlapping_regions']} regions")

def save_results_to_csv(results, output_file="difference_analysis.csv"):
    """
    Save analysis results to a CSV file

    Parameters:
    results: Analysis results dictionary
    output_file: Output filename
    """
    all_data = []
    covered_regions_by_gene = defaultdict(set)  # Track covered regions by gene

    for i, row_infos in results.items():
        for row_info in row_infos:
            gene_id = row_info['gene_id']

            # Parse all covered regions for this window
            if row_info['all_overlapping_regions']:
                # Parse regions from string
                regions_str = row_info['all_overlapping_regions']
                for region_str in regions_str.split('; '):
                    # Parse format like "(56, 69)"
                    region_str = region_str.strip('()')
                    start, end = map(int, region_str.split(', '))
                    covered_regions_by_gene[gene_id].add((start, end))

            all_data.append({
                'chromosome': i,
                'row': row_info['row'],
                'gene_id': row_info['gene_id'],
                'rank': row_info['rank'],
                'window_start': row_info['window_start'],
                'window_end': row_info['window_end'],
                'window_size': row_info['window_size'],
                'total_diff': row_info['total_diff'],
                'avg_diff': row_info['avg_diff'],
                'overlap_length': row_info['overlap_length'],
                'max_overlap_region': row_info['max_overlap_region'],
                'overlap_percentage': row_info['overlap_percentage'],
                'num_overlapping_regions': row_info['num_overlapping_regions'],
                'all_overlapping_regions': row_info['all_overlapping_regions']
            })

    # Create DataFrame and save
    column_order = ['chromosome', 'row', 'gene_id', 'rank', 'window_start', 'window_end',
                    'window_size', 'total_diff', 'avg_diff',
                    'overlap_length', 'max_overlap_region', 'overlap_percentage',
                    'num_overlapping_regions', 'all_overlapping_regions']

    df = pd.DataFrame(all_data)

    # Count total and covered biological regions
    total_biological_regions = sum(len(regions) for regions in BIOLOGICAL_REGIONS.values())
    total_covered_regions = sum(len(regions) for regions in covered_regions_by_gene.values())
    coverage_rate = (total_covered_regions / total_biological_regions * 100) if total_biological_regions > 0 else 0

    if not df.empty:
        df = df[column_order]

        print(f"\n{'=' * 70}")
        print(f"Biological Region Coverage Statistics")
        print(f"{'=' * 70}")
        print(f"Total biological regions: {total_biological_regions}")
        print(f"Covered regions (deduplicated): {total_covered_regions}")
        print(f"Coverage rate: {coverage_rate:.2f}%")
        print(f"Total windows found: {len(df)}")

        # Statistics by gene
        print(f"\n{'=' * 70}")
        print(f"Region Coverage by Gene")
        print(f"{'=' * 70}")
        for gene_id in sorted(BIOLOGICAL_REGIONS.keys()):
            total_regions = len(BIOLOGICAL_REGIONS[gene_id])
            covered_count = len(covered_regions_by_gene.get(gene_id, set()))
            coverage = (covered_count / total_regions * 100) if total_regions > 0 else 0

            # Count windows for this gene
            gene_windows = df[df['gene_id'] == gene_id]
            num_windows = len(gene_windows)

            print(f"{gene_id:20s}: {covered_count:2d}/{total_regions:2d} regions covered ({coverage:5.1f}%) - "
                  f"found {num_windows} windows")

        # List covered and uncovered regions for each gene
        print(f"\n{'=' * 70}")
        print(f"Detailed Region Coverage")
        print(f"{'=' * 70}")
        for gene_id in sorted(BIOLOGICAL_REGIONS.keys()):
            all_regions = set(BIOLOGICAL_REGIONS[gene_id])
            covered = covered_regions_by_gene.get(gene_id, set())
            uncovered = all_regions - covered

            print(f"\n{gene_id}:")
            if covered:
                print(f"  Covered regions ({len(covered)}/{len(all_regions)}):")
                for region in sorted(covered):
                    print(f"    {region}")
            if uncovered:
                print(f"  Uncovered regions ({len(uncovered)}/{len(all_regions)}):")
                for region in sorted(uncovered):
                    print(f"    {region}")
            if not uncovered:
                print(f"  ✓ All regions covered!")

    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

# Run analysis
if __name__ == "__main__":
    window_size = 8  # Window size
    top_n = 20  # Calculate top 20 windows with largest differences for each gene
    results = analyze_csv_differences(window_size=window_size, top_n=top_n)
    save_results_to_csv(results)