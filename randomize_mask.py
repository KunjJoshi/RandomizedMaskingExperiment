import pandas as pd
import re
import json
from collections import defaultdict, Counter
import random
import os
from tqdm import tqdm

EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')


def save_all_email_locations(dataset_path, output_file, data_key):
    """
    Build a store of ALL email occurrences (including first)
    Format: {email: [(doc_idx, start, end), ...]}
    """
    email_occurrences = defaultdict(list)

    df = pd.read_csv(dataset_path)
    texts = df[data_key].dropna().astype(str).tolist()

    for i, text in tqdm(enumerate(texts), desc="Creating PII Store"):
        matches = [(m.group(), m.start(), m.end()) for m in EMAIL_REGEX.finditer(text)]

        for email, start, end in matches:
            email_occurrences[email.lower()].append((i, start, end))

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(email_occurrences, f, indent=4)
    
    return email_occurrences


def count_email_occurrences(dataset_path, data_key):
    """Count how many times each email appears"""
    df = pd.read_csv(dataset_path)
    texts = df[data_key].dropna().astype(str).tolist()
    
    email_counter = Counter()
    for text in texts:
        emails = [e.lower() for e in EMAIL_REGEX.findall(text)]
        email_counter.update(emails)
    
    return dict(email_counter)


def calculate_occurrence_stats(email_counts):
    """Calculate statistics about email occurrences"""
    if not email_counts:
        return {
            'total_unique_emails': 0,
            'total_occurrences': 0,
            'avg_occurrences': 0,
            'max_occurrences': 0,
            'min_occurrences': 0,
            'median_occurrences': 0
        }
    
    counts = list(email_counts.values())
    return {
        'total_unique_emails': len(email_counts),
        'total_occurrences': sum(counts),
        'avg_occurrences': sum(counts) / len(counts),
        'max_occurrences': max(counts),
        'min_occurrences': min(counts),
        'median_occurrences': sorted(counts)[len(counts)//2]
    }


with open('email_vals/first_part.json', 'r') as file:
    first_part = json.load(file)
FIRST_PART_STORE = first_part['values']

with open('email_vals/domain.json', 'r') as file:
    domain_part = json.load(file)
DOMAIN_STORE = domain_part['values']

with open('email_vals/tld.json', 'r') as file:
    tld_part = json.load(file)
TLD_STORE = tld_part['values']


def generate_masked_email(original_email, anchor=None, anchor_val=None):
    """Generate a single masked email"""
    first_part, last_part = original_email.split('@')
    
    # Generate first part
    if '.' in first_part:
        parts = first_part.split('.')
        fp = '.'.join([random.choice(FIRST_PART_STORE) for _ in parts])
    else:
        fp = random.choice(FIRST_PART_STORE)
    
    # Generate domain and TLD
    domain = random.choice(DOMAIN_STORE)
    tld = random.choice(TLD_STORE)
    
    # Apply anchor if specified
    if anchor == 'fp':
        return f'{anchor_val}@{domain}.{tld}'
    elif anchor == 'domain':
        last_part_splits = last_part.split('.')
        domain_orig = last_part_splits[0]
        return f'{fp}@{domain_orig}.{tld}'
    elif anchor == 'tld':
        tld_orig = '.'.join(last_part.split('.')[1:])
        return f'{fp}@{domain}.{tld_orig}'
    else:
        return f'{fp}@{domain}.{tld}'


def disect_email(email):
    """Extract components from email"""
    first_part, last_part = email.split('@')
    last_part_splits = last_part.split('.')
    domain = last_part_splits[0]
    tld = '.'.join(last_part_splits[1:])
    return first_part, domain, tld


def randomize_mask(dataset_path, output_path, data_key, pii_store_path=None):
    print("\n" + "="*70)
    print("RANDOMIZED MASKING - OCCURRENCE ANALYSIS")
    print("="*70)
    
    # BEFORE statistics
    print("\n📊 BEFORE MASKING:")
    before_counts = count_email_occurrences(dataset_path, data_key)
    before_stats = calculate_occurrence_stats(before_counts)
    
    print(f"  Total unique emails: {before_stats['total_unique_emails']}")
    print(f"  Total occurrences: {before_stats['total_occurrences']}")
    print(f"  Average occurrences per email: {before_stats['avg_occurrences']:.2f}")
    print(f"  Median occurrences: {before_stats['median_occurrences']}")
    print(f"  Max occurrences: {before_stats['max_occurrences']}")
    
    top_10_before = sorted(before_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top 10 most frequent emails BEFORE masking:")
    for email, count in top_10_before:
        print(f"    {email}: {count} times")
    
    # Build PII store
    if pii_store_path is None:
        pii_store_path = "jsons/pii_store.json"
        email_store = save_all_email_locations(dataset_path, pii_store_path, data_key)
    else:
        with open(pii_store_path, 'r') as file:
            email_store = json.load(file)
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    data = df[data_key].to_list()
    
    print(f"\n🎭 MASKING {len(email_store)} emails with multiple occurrences...")
    
    # Build replacement plan FIRST (before modifying any text)
    replacement_plan = []  # List of (doc_idx, start, end, original_email, masked_email)
    
    for email, occurrences in tqdm(email_store.items(), desc="Planning Replacements"):
        if len(occurrences) == 0:
            continue
        
        # Sort occurrences by (doc_idx, start) to get canonical first occurrence
        sorted_occs = sorted(occurrences, key=lambda x: (x[0], x[1]))
        
        # Get anchor from original email
        fp, dom, tld = disect_email(email)
        
        # Skip first occurrence, mask the rest
        for occ in sorted_occs[1:]:
            doc_idx, start, end = occ
            
            # Randomly select anchor
            anchor_choice = random.choice(['fp', 'domain', 'tld'])
            if anchor_choice == 'fp':
                masked_email = generate_masked_email(email, anchor='fp', anchor_val=fp)
            elif anchor_choice == 'domain':
                masked_email = generate_masked_email(email, anchor='domain', anchor_val=dom)
            else:
                masked_email = generate_masked_email(email, anchor='tld', anchor_val=tld)
            
            replacement_plan.append((doc_idx, start, end, email, masked_email))
    
    print(f"  Generated {len(replacement_plan)} replacements")
    
    # Group replacements by document to handle position shifts
    doc_replacements = defaultdict(list)
    for doc_idx, start, end, orig, masked in replacement_plan:
        doc_replacements[doc_idx].append((start, end, orig, masked))
    
    # Apply replacements document by document (reverse order to avoid position shifts)
    print("\n🔧 Applying replacements...")
    for doc_idx in tqdm(doc_replacements.keys(), desc="Masking Documents"):
        text = data[doc_idx]
        
        # Sort replacements in REVERSE order by start position
        # This way we replace from end to beginning, avoiding position shifts
        replacements = sorted(doc_replacements[doc_idx], key=lambda x: x[0], reverse=True)
        
        for start, end, orig_email, masked_email in replacements:
            # Verify the original email is actually at this position
            actual_text = text[start:end]
            if actual_text.lower() == orig_email.lower():
                text = text[:start] + masked_email + text[end:]
            else:
                print(f"    Warning: Expected '{orig_email}' at position {start}-{end} in doc {doc_idx}, "
                      f"found '{actual_text}'. Skipping.")
        
        data[doc_idx] = text
    
    # Save masked dataset
    df[data_key] = data
    
    # Determine output path
    if output_path:
        output_file = output_path
    else:
        dataset_folder = os.path.dirname(dataset_path)
        dataset_file = os.path.basename(dataset_path)
        output_file = os.path.join(dataset_folder, f"randomize_masked_{dataset_file}")
    
    df.to_csv(output_file, index=False)
    
    # AFTER statistics
    print("\n📊 AFTER MASKING:")
    after_counts = count_email_occurrences(output_file, data_key)
    after_stats = calculate_occurrence_stats(after_counts)
    
    print(f"  Total unique emails: {after_stats['total_unique_emails']}")
    print(f"  Total occurrences: {after_stats['total_occurrences']}")
    print(f"  Average occurrences per email: {after_stats['avg_occurrences']:.2f}")
    print(f"  Median occurrences: {after_stats['median_occurrences']}")
    print(f"  Max occurrences: {after_stats['max_occurrences']}")
    
    original_emails_remaining = set(before_counts.keys()) & set(after_counts.keys())
    print(f"\n  Original emails still present: {len(original_emails_remaining)} / {len(before_counts)}")
    
    top_10_after = sorted(after_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top 10 most frequent emails AFTER masking:")
    for email, count in top_10_after:
        print(f"    {email}: {count} times")
    
    # Summary
    print("\n" + "="*70)
    print("📈 REDUCTION SUMMARY:")
    print("="*70)
    print(f"  Avg occurrences reduced: {before_stats['avg_occurrences']:.2f} → {after_stats['avg_occurrences']:.2f}")
    reduction_pct = ((before_stats['avg_occurrences'] - after_stats['avg_occurrences']) / 
                     before_stats['avg_occurrences'] * 100)
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  Max occurrences reduced: {before_stats['max_occurrences']} → {after_stats['max_occurrences']}")
    print(f"  New unique emails created: {after_stats['total_unique_emails'] - before_stats['total_unique_emails']}")
    
    # Check if goal achieved
    emails_with_multiple_occurrences = sum(1 for count in after_counts.values() if count > 1)
    print(f"\n⚠️  Emails still appearing >1 time: {emails_with_multiple_occurrences}")
    print("="*70 + "\n")
    
    print(f"✅ Masked dataset saved to: {output_file}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomize and mask email occurrences in dataset"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to input CSV dataset"
    )

    parser.add_argument(
        "--data_key",
        type=str,
        required=True,
        help="Column name containing text data"
    )

    parser.add_argument(
        "--pii_store_path",
        type=str,
        default=None,
        help="Optional path to precomputed PII store JSON (if not provided, will be generated)"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output CSV path (default: randomize_masked_<dataset_path>)"
    )

    args = parser.parse_args()

    # Run masking
    randomize_mask(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        data_key=args.data_key,
        pii_store_path=args.pii_store_path
    )