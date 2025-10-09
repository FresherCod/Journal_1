import pandas as pd
from scipy import stats
import warnings
import os

# ==============================================================================
# --- CONFIGURATION (EDIT THIS SECTION) ---
# ==============================================================================

# 1. Define the full paths to your 5 dataset Excel files.
# Use r"..." syntax to avoid issues with backslashes on Windows.
DATASET_FILES = {
    # "Display Name": r"Path\to\your\file.xlsx"
    "Proprietary (19 samples)": r"D:\scr\journal\results\dataset_19sample\report_dataset_19sample.xlsx",
    "MVTec AD (15 classes)":    r"D:\scr\journal\results\dataset_mvtec\report_dataset_mvtec.xlsx",
    "MVTec 2D LOCO (3 classes)":r"D:\scr\journal\results\dataset_mvtec2\report_dataset_mvtec2.xlsx",
    "Amazon (11 classes)":      r"D:\scr\journal\results\dataset_amazon\report_dataset_amazon.xlsx",
    "Kaggle (7 classes)":       r"D:\scr\journal\results\dataset_kaggle\report_dataset_kaggle.xlsx"
}

# 2. Define the output text report file name.
OUTPUT_TEXT_REPORT = "Statistical_Analysis_Report.txt"

# 3. Define the metric to analyze.
METRIC_TO_ANALYZE = 'NG_F1'

# 4. Define the significance level (alpha).
SIGNIFICANCE_LEVEL = 0.05

# 5. Define the baseline method and the methods to compare against it.
BASELINE_METHOD = 'One_Class_SVM'
METHODS_TO_COMPARE = ['AnoGAN', 'Single_Centroid', 'KMeans_Multi_Centroid', 'VAE']

# ==============================================================================
# --- END OF CONFIGURATION ---
# ==============================================================================


def perform_statistical_analysis(dataset_paths, metric, baseline, competitors, alpha):
    """
    Performs pairwise statistical analysis and returns a results DataFrame.
    """
    all_results = []
    warnings.filterwarnings('ignore', category=UserWarning)

    # Automatically clean up method names to prevent whitespace errors
    baseline = baseline.strip()
    competitors = [c.strip() for c in competitors]

    for dataset_name, file_path in dataset_paths.items():
        if not os.path.exists(file_path):
            print(f"WARNING: File not found for '{dataset_name}'. Skipping.")
            continue
        
        try:
            df_dataset = pd.read_excel(file_path)
            # Automatically clean the 'method' column
            df_dataset['method'] = df_dataset['method'].astype(str).str.strip()
        except Exception as e:
            print(f"ERROR: Could not read file for '{dataset_name}'. Error: {e}. Skipping.")
            continue

        wide_df = df_dataset.pivot_table(index='sample', columns='method', values=metric)

        for competitor in competitors:
            if baseline not in wide_df.columns or competitor not in wide_df.columns:
                continue

            combined = pd.concat([wide_df[baseline], wide_df[competitor]], axis=1).dropna()
            num_pairs = len(combined)
            p_value = float('nan')
            note = ""

            if num_pairs < 6:
                note = f"(N={num_pairs} is too small)"
            else:
                try:
                    stat, p_value = stats.wilcoxon(combined[baseline], combined[competitor], alternative='two-sided')
                except ValueError:
                    p_value = 1.0
                    note = "(Differences are all zero)"
            
            all_results.append({
                'Dataset': dataset_name,
                'Comparison': f"{baseline} vs. {competitor}",
                'p-value': p_value,
                'N_Pairs': num_pairs,
                'Note': note
            })

    if not all_results:
        return None
        
    results_df = pd.DataFrame(all_results)
    results_df['Significant'] = results_df['p-value'].apply(
        lambda p: 'Yes *' if pd.notna(p) and p < alpha else 'No'
    )
    results_df['p-value'] = results_df['p-value'].map('{:.4f}'.format).replace('nan', 'N/A')

    return results_df[['Dataset', 'Comparison', 'p-value', 'N_Pairs', 'Significant', 'Note']]


def generate_insights_text(results_df, baseline, alpha):
    """
    Generates a written analysis in English based on the statistical results.
    """
    if results_df is None or results_df.empty:
        return "No data available for analysis."

    insights = ["STATISTICAL ANALYSIS & CONCLUSIONS", 
                "="*40, 
                f"This report summarizes the statistical comparison of anomaly detection methods against a '{baseline}' baseline.",
                f"The metric analyzed is '{METRIC_TO_ANALYZE}'. A result is considered statistically significant if the p-value is less than {alpha}.",
                "\n",
                "--- DETAILED ANALYSIS PER DATASET ---"]

    for dataset_name, group in results_df.groupby('Dataset'):
        insights.append(f"\nAnalysis for: {dataset_name}")
        for _, row in group.iterrows():
            comp_name = row['Comparison'].split(' vs. ')[1]
            pval_str = f"p={row['p-value']}"
            
            if row['Note']:
                insights.append(f"- {comp_name}: The number of samples ({row['N_Pairs']}) was too small for a reliable statistical test.")
                continue

            if row['Significant'] == 'Yes *':
                # Note: Directional conclusion would require mean comparison, omitted for simplicity here.
                # This version focuses only on significance.
                insights.append(f"- {comp_name}: Performance difference was STATISTICALLY SIGNIFICANT ({pval_str}).")
            else:
                insights.append(f"- {comp_name}: There was NO statistically significant difference in performance ({pval_str}).")

    # --- Executive Summary ---
    insights.extend(["\n", "--- EXECUTIVE SUMMARY ---"])
    
    anogan_sig = results_df[(results_df['Comparison'].str.contains('AnoGAN')) & (results_df['Significant'] == 'Yes *')]
    if not anogan_sig.empty:
        insights.append(f"1. Performance of AnoGAN: The performance of AnoGAN was significantly different from the '{baseline}' baseline on the following dataset(s): {', '.join(anogan_sig['Dataset'].tolist())}. This warrants a closer look at the mean scores in the paper to determine superiority.")
    
    sc_no_diff = results_df[(results_df['Comparison'].str.contains('Single_Centroid')) & (results_df['Significant'] == 'No')]
    if len(sc_no_diff) > 1:
        insights.append(f"2. The Case for Simplicity: The simplest model, Single_Centroid, often showed no significant performance difference compared to the more complex '{baseline}'. This reinforces the argument that simpler, faster models are a highly pragmatic choice for production environments.")

    return "\n".join(insights)


if __name__ == '__main__':
    print("Starting statistical analysis...")

    final_table = perform_statistical_analysis(
        dataset_paths=DATASET_FILES,
        metric=METRIC_TO_ANALYZE,
        baseline=BASELINE_METHOD,
        competitors=METHODS_TO_COMPARE,
        alpha=SIGNIFICANCE_LEVEL
    )

    if final_table is not None and not final_table.empty:
        # Generate the analysis text
        analysis_summary = generate_insights_text(final_table, BASELINE_METHOD, SIGNIFICANCE_LEVEL)
        
        # Prepare the final output string
        output_lines = []
        output_lines.append("="*60)
        output_lines.append("--- FINAL STATISTICAL ANALYSIS REPORT ---")
        output_lines.append("="*60)
        output_lines.append(final_table.to_string(index=False))
        output_lines.append("\n" + "="*60)
        output_lines.append("\n" + analysis_summary)
        output_lines.append("\n" + "="*60)
        output_lines.append("\n* Notes:")
        output_lines.append("  - Test Used: Wilcoxon signed-rank test.")
        output_lines.append("  - 'N_Pairs': Number of valid sample/class pairs used for comparison.")
        output_lines.append("  - 'Significant': 'Yes *' indicates a statistically significant difference (p < 0.05).")

        final_output_string = "\n".join(output_lines)

        # Print to console
        print(final_output_string)

        # Save to text file
        try:
            with open(OUTPUT_TEXT_REPORT, 'w', encoding='utf-8') as f:
                f.write(final_output_string)
            print(f"\nSUCCESS: Report has been saved to '{OUTPUT_TEXT_REPORT}'")
        except Exception as e:
            print(f"\nERROR: Could not write to file '{OUTPUT_TEXT_REPORT}'. Error: {e}")

    else:
        print("\nANALYSIS FAILED: No results were generated. Please check your file paths and the 'method' column names in your Excel files.")