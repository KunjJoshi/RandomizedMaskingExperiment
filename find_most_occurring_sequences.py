import pandas as pd
import re
import json

base_path = 'results/gpt_base/110e3_leakage_discoverable_enron.csv'
rmft_path = 'results/gpt_rmft/110e3_leakage_discoverable_enron.csv'
dedup_path = 'results/gpt_dedup/110e3_leakage_discoverable_enron.csv'

base_df = pd.read_csv(base_path)
rmft_df = pd.read_csv(rmft_path)
dedup_df = pd.read_csv(dedup_path)

base_leaks = set(base_df['leaked_email'].to_list())
rmft_leaks = set(rmft_df['leaked_email'].to_list())
dedup_leaks = set(dedup_df['leaked_email'].to_list())

def get_most_occurring_substrings(list_set, min_len=5):
    # --- Frequency pass ---
    freq_map = {}
    for s in list_set:
        seen = set()
        for start in range(len(s)):
            for end in range(start + min_len, len(s) + 1):
                substr = s[start:end]
                if substr not in seen:
                    seen.add(substr)
                    freq_map[substr] = freq_map.get(substr, 0) + 1

    

    # --- Absorption pass ---
    keys = sorted(freq_map.keys(), key=len, reverse=True)  # longest first
    to_delete = set()

    for long_str in keys:
        if long_str in to_delete:  # already absorbed, skip
            continue
        L = len(long_str)
        for start in range(L):
            for end in range(start + 1, L + 1):
                sub = long_str[start:end]
                if sub == long_str:
                    continue
                if sub in freq_map and freq_map[sub] <= freq_map[long_str]:
                    to_delete.add(sub)

    for key in to_delete:
        del freq_map[key]
    
    freq_map = dict(sorted(freq_map.items(), key=lambda x: x[1], reverse=True))
    return freq_map

def print_top_substrings(leaks, label, exclude=("enron", "com"), top_n=5):
    base_map = get_most_occurring_substrings(leaks)
    filtered_keys = [
        key for key in base_map
        if not any(term in key for term in exclude)
    ]
    print(f"(ENRON EXCLUDED) Seeing which strings occur maximum number of times for {label}:")
    for key in filtered_keys[:top_n]:
        print(f"Key: {key} Occurring: {base_map[key]} times")

print_top_substrings(base_leaks, "Baseline")
print_top_substrings(rmft_leaks,     "RMFT")
print_top_substrings(dedup_leaks,    "Deduplication")

