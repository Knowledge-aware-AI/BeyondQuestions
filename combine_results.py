#!/usr/bin/env python3
"""
Standalone script to combine all evaluation results into a single summary CSV.
Can be run independently after evaluations complete (or partially complete).
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def compute_f1(precision_val, recall_val):
    if precision_val + recall_val == 0:
        return 0.0
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)


def combine_results(results_dir: str, output_path: str = None) -> str:
    """
    Discover all results files in the results directory and combine them.
    For each (Model, Setting, Domain) pair, compute F1 score from Precision and Recall rows.
    
    Args:
        results_dir: Base results directory containing model subdirectories
        output_path: Optional custom output path. Defaults to results_dir/combined_results_manual.csv
    
    Returns:
        Path to the combined summary file
    """
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    if output_path is None:
        output_path = os.path.join(results_dir, "combined_results_manual.csv")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    all_rows = []
    files_found = 0
    
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f in ["results.csv", "results_by_category.csv", "results_by_popularity.csv"]:
                csv_path = os.path.join(root, f)
                rel_path = os.path.relpath(csv_path, results_dir)
                
                parts = rel_path.split(os.sep)
                
                rag_eval_idx = None
                for i, part in enumerate(parts):
                    if part.endswith("_rag_eval"):
                        rag_eval_idx = i
                        break
                
                if rag_eval_idx is not None:
                    model_name = parts[0]
                    setting = os.sep.join(parts[1:rag_eval_idx]) if rag_eval_idx > 1 else ""
                elif len(parts) >= 2:
                    model_name = parts[0]
                    setting = os.sep.join(parts[1:]) if len(parts) > 1 else ""
                else:
                    model_name = "unknown"
                    setting = ""
                
                if not setting:
                    setting = "root"
                
                try:
                    with open(csv_path, 'r', newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            row_copy = dict(row)
                            row_copy['Model'] = model_name
                            row_copy['Setting'] = setting
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
    
    grouped = {}
    for row in all_rows:
        model = row.get('Model', '')
        setting = row.get('Setting', '')
        domain = row.get('Category', '') if setting == 'domains' else ''
        key = (model, setting, domain)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    
    final_rows = []
    for (model, setting, domain), rows in grouped.items():
        precision_row = None
        recall_row = None
        for row in rows:
            metric = row.get('Metric', '')
            if 'Precision' in metric:
                precision_row = row
            elif 'Recall' in metric:
                recall_row = row
        
        combined = {
            'Model': model,
            'Setting': setting,
            'Domain': domain if domain else '',
            'Total #Triples': rows[0].get('Total #Triples', '500') if rows else '500',
        }
        
        for col in ['Entailment', 'Contradiction', 'Neutral', 'Entailment_ratio', 'Contradiction_ratio', 'Neutral_ratio', 'Error_count', 'Error_ratio']:
            prec_val = float(precision_row.get(col, 0)) if precision_row else 0
            rec_val = float(recall_row.get(col, 0)) if recall_row else 0
            combined[f'{col}_Precision'] = prec_val
            combined[f'{col}_Recall'] = rec_val
        
        if precision_row and recall_row:
            prec_ent = float(precision_row.get('Entailment_ratio', 0))
            rec_ent = float(recall_row.get('Entailment_ratio', 0))
            combined['Entailment_F1'] = compute_f1(prec_ent, rec_ent)
        else:
            combined['Entailment_F1'] = 0.0
        
        final_rows.append(combined)
    
    fieldnames = ['Model', 'Setting', 'Domain', 'Total #Triples',
                  'Entailment_Precision', 'Entailment_Recall', 'Entailment_F1',
                  'Contradiction_Precision', 'Contradiction_Recall',
                  'Neutral_Precision', 'Neutral_Recall',
                  'Entailment_ratio_Precision', 'Entailment_ratio_Recall',
                  'Contradiction_ratio_Precision', 'Contradiction_ratio_Recall',
                  'Neutral_ratio_Precision', 'Neutral_ratio_Recall',
                  'Error_count_Precision', 'Error_count_Recall',
                  'Error_ratio_Precision', 'Error_ratio_Recall']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"Combined {len(final_rows)} model/setting pairs from {files_found} result files")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Combine all evaluation results into a single summary CSV"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results",
        help="Base results directory containing model subdirectories (default: results)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output path for combined results (default: results_dir/combined_results_manual.csv)"
    )
    
    args = parser.parse_args()
    
    combine_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
