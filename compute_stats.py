import os
import csv
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent / "elicited_triples"
OUTPUT_DIR = Path(__file__).parent

per_file_rows = []
per_model_accum = {}  # model -> accumulated data

def compute_file_stats(csv_path, model_name, setting):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    if total == 0:
        return None

    subjects = [r['subject'] for r in rows]
    predicates = [r['predicate'] for r in rows]
    objects = [r['object'] for r in rows]
    triples = [(r['subject'], r['predicate'], r['object']) for r in rows]

    unique_subjects = set(subjects)
    unique_predicates = set(predicates)
    unique_objects = set(objects)
    unique_triples = set(triples)

    duplicate_count = total - len(unique_triples)
    duplicate_pct = round(duplicate_count / total * 100, 2) if total > 0 else 0
    unique_ratio = round(len(unique_triples) / total, 4) if total > 0 else 0

    triples_per_subject = Counter(subjects)
    avg_tps = round(total / len(unique_subjects), 2) if unique_subjects else 0
    max_tps = max(triples_per_subject.values()) if triples_per_subject else 0
    min_tps = min(triples_per_subject.values()) if triples_per_subject else 0

    pred_counter = Counter(predicates)
    subj_counter = Counter(subjects)
    top3_pred = '; '.join(f"{p}({c})" for p, c in pred_counter.most_common(3))
    top3_subj = '; '.join(f"{s}({c})" for s, c in subj_counter.most_common(3))

    return {
        'model': model_name,
        'setting': setting,
        'file': str(csv_path.relative_to(BASE_DIR)),
        'total_triples': total,
        'unique_subjects': len(unique_subjects),
        'unique_predicates': len(unique_predicates),
        'unique_objects': len(unique_objects),
        'avg_triples_per_subject': avg_tps,
        'max_triples_per_subject': max_tps,
        'min_triples_per_subject': min_tps,
        'duplicate_triples': duplicate_count,
        'duplicate_pct': duplicate_pct,
        'unique_triple_ratio': unique_ratio,
        'top3_predicates': top3_pred,
        'top3_subjects': top3_subj,
        # store raw data for per-model aggregation
        '_subjects': unique_subjects,
        '_predicates': unique_predicates,
        '_objects': unique_objects,
        '_triples': unique_triples,
        '_all_subjects': subjects,
    }

# Walk all model directories
for model_dir in sorted(BASE_DIR.iterdir()):
    if not model_dir.is_dir():
        continue
    model_name = model_dir.name

    for csv_path in sorted(model_dir.rglob('*.csv')):
        # setting = relative path from model dir, without filename extension
        rel = csv_path.relative_to(model_dir)
        parts = list(rel.parts)
        if len(parts) == 1:
            setting = 'random'  # top-level file
        else:
            setting = '/'.join(parts[:-1])  # subdirectory path

        stats = compute_file_stats(csv_path, model_name, setting)
        if stats is None:
            continue

        per_file_rows.append(stats)

        # Accumulate for per-model
        if model_name not in per_model_accum:
            per_model_accum[model_name] = {
                'all_subjects': set(),
                'all_predicates': set(),
                'all_objects': set(),
                'all_triples': set(),
                'all_subjects_list': [],
                'total_triples': 0,
            }
        acc = per_model_accum[model_name]
        acc['all_subjects'].update(stats['_subjects'])
        acc['all_predicates'].update(stats['_predicates'])
        acc['all_objects'].update(stats['_objects'])
        acc['all_triples'].update(stats['_triples'])
        acc['all_subjects_list'].extend(stats['_all_subjects'])
        acc['total_triples'] += stats['total_triples']

# Write per-file CSV
per_file_fields = [
    'model', 'setting', 'file',
    'total_triples', 'unique_subjects', 'unique_predicates', 'unique_objects',
    'avg_triples_per_subject', 'max_triples_per_subject', 'min_triples_per_subject',
    'duplicate_triples', 'duplicate_pct', 'unique_triple_ratio',
    'top3_predicates', 'top3_subjects',
]
per_file_out = OUTPUT_DIR / 'stats_per_file.csv'
with open(per_file_out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=per_file_fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(per_file_rows)

print(f"Written: {per_file_out} ({len(per_file_rows)} rows)")

# Build per-model rows
per_model_rows = []
for model_name, acc in sorted(per_model_accum.items()):
    total = acc['total_triples']
    unique_subj = acc['all_subjects']
    unique_pred = acc['all_predicates']
    unique_obj = acc['all_objects']
    unique_triples = acc['all_triples']
    subj_list = acc['all_subjects_list']

    duplicate_count = total - len(unique_triples)
    duplicate_pct = round(duplicate_count / total * 100, 2) if total > 0 else 0
    unique_ratio = round(len(unique_triples) / total, 4) if total > 0 else 0

    tps = Counter(subj_list)
    avg_tps = round(total / len(unique_subj), 2) if unique_subj else 0
    max_tps = max(tps.values()) if tps else 0
    min_tps = min(tps.values()) if tps else 0

    per_model_rows.append({
        'model': model_name,
        'total_triples': total,
        'unique_subjects': len(unique_subj),
        'unique_predicates': len(unique_pred),
        'unique_objects': len(unique_obj),
        'avg_triples_per_subject': avg_tps,
        'max_triples_per_subject': max_tps,
        'min_triples_per_subject': min_tps,
        'duplicate_triples': duplicate_count,
        'duplicate_pct': duplicate_pct,
        'unique_triple_ratio': unique_ratio,
    })

per_model_fields = [
    'model',
    'total_triples', 'unique_subjects', 'unique_predicates', 'unique_objects',
    'avg_triples_per_subject', 'max_triples_per_subject', 'min_triples_per_subject',
    'duplicate_triples', 'duplicate_pct', 'unique_triple_ratio',
]
per_model_out = OUTPUT_DIR / 'stats_per_model.csv'
with open(per_model_out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=per_model_fields)
    writer.writeheader()
    writer.writerows(per_model_rows)

print(f"Written: {per_model_out} ({len(per_model_rows)} rows)")
