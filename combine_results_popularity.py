#!/usr/bin/env python3
"""
Combine all results_by_popularity.csv files into a single summary CSV.
For each (Model, PopularityCategory) pair, computes F1 from Precision and Recall rows.
"""

import argparse
import csv
import os
import sys


def compute_f1(precision_val, recall_val):
    if precision_val + recall_val == 0:
        return 0.0
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)


def combine_results_popularity(results_dir: str, output_path: str = None) -> str:
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    if output_path is None:
        output_path = os.path.join(results_dir, "combined_results_popularity.csv")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_rows = []
    files_found = 0

    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == "results_by_popularity.csv":
                csv_path = os.path.join(root, f)
                rel_path = os.path.relpath(csv_path, results_dir)
                parts = rel_path.split(os.sep)
                model_name = parts[0]

                try:
                    with open(csv_path, 'r', newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            row_copy = dict(row)
                            row_copy['Model'] = model_name
                            all_rows.append(row_copy)
                    files_found += 1
                    print(f"Found: {rel_path}")
                except Exception as e:
                    print(f"Warning: Could not read {csv_path}: {e}")

    if not all_rows:
        print("Warning: No results found to combine")
        with open(output_path, 'w', newline='') as f:
            f.write("")
        return output_path

    # Group by (Model, Category) to pair Precision and Recall rows
    grouped = {}
    for row in all_rows:
        key = (row['Model'], row['Category'])
        grouped.setdefault(key, []).append(row)

    metric_cols = ['Entailment', 'Contradiction', 'Neutral',
                   'Entailment_ratio', 'Contradiction_ratio', 'Neutral_ratio']

    final_rows = []
    for (model, category), rows in sorted(grouped.items()):
        precision_row = next((r for r in rows if 'Precision' in r.get('Metric', '')), None)
        recall_row = next((r for r in rows if 'Recall' in r.get('Metric', '')), None)

        combined = {
            'Model': model,
            'PopularityCategory': category,
            'Total #Triples': rows[0].get('Total #Triples', ''),
        }

        for col in metric_cols:
            prec_val = float(precision_row[col]) if precision_row else 0.0
            rec_val = float(recall_row[col]) if recall_row else 0.0
            combined[f'{col}_Precision'] = prec_val
            combined[f'{col}_Recall'] = rec_val

        prec_ent = float(precision_row['Entailment_ratio']) if precision_row else 0.0
        rec_ent = float(recall_row['Entailment_ratio']) if recall_row else 0.0
        combined['Entailment_ratio_F1'] = compute_f1(prec_ent, rec_ent)

        final_rows.append(combined)

    fieldnames = ['Model', 'PopularityCategory', 'Total #Triples',
                  'Entailment_Precision', 'Entailment_Recall',
                  'Contradiction_Precision', 'Contradiction_Recall',
                  'Neutral_Precision', 'Neutral_Recall',
                  'Entailment_ratio_Precision', 'Entailment_ratio_Recall', 'Entailment_ratio_F1',
                  'Contradiction_ratio_Precision', 'Contradiction_ratio_Recall',
                  'Neutral_ratio_Precision', 'Neutral_ratio_Recall']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"Combined {len(final_rows)} model/category pairs from {files_found} files")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Combine results_by_popularity.csv files into a single summary CSV"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="RESULTS/NEW_POPULARITY",
        help="Base results directory containing model subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: results_dir/combined_results_popularity.csv)"
    )

    args = parser.parse_args()
    combine_results_popularity(args.results_dir, args.output)


if __name__ == "__main__":
    main()
