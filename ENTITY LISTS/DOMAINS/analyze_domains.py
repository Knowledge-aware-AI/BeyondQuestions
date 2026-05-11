import json
import math
import os

# Get base directory (parent of ENTITIES/DOMAINS directory)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_BASE_DIR, "wikipedia_entities_by_domain_1000.json")) as f:
    data = json.load(f)

def mean_std(lst):
    n = len(lst)
    m = sum(lst) / n
    variance = sum((x - m) ** 2 for x in lst) / n
    return m, math.sqrt(variance)

def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    return sorted_lst[mid]

def percentile(lst, p):
    sorted_lst = sorted(lst)
    k = (len(sorted_lst) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_lst[int(k)]
    return sorted_lst[f] * (c - k) + sorted_lst[c] * (k - f)

results = []
for domain, entities in data.items():
    lengths = [e["length"] for e in entities]
    mean_len, std_len = mean_std(lengths)
    med = median(lengths)
    p25 = percentile(lengths, 25)
    p75 = percentile(lengths, 75)
    min_len = min(lengths)
    max_len = max(lengths)
    results.append([domain, mean_len, std_len, med, min_len, max_len, p25, p75])

with open(os.path.join(_BASE_DIR, "domain_stats.csv"), "w") as f:
    f.write("domain,mean,std,median,min,max,p25,p75\n")
    for r in results:
        f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{r[4]},{r[5]},{r[6]:.2f},{r[7]:.2f}\n")