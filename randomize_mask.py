import pandas as pd
import re
import json
from collections import defaultdict
import random
import os
from tqdm import tqdm

EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')


def save_all_email_locations(dataset_path, output_file, data_key):
    email_occurrences = defaultdict(list)

    df = pd.read_csv(dataset_path)
    texts = df[data_key].dropna().astype(str).tolist()

    for i, text in tqdm(enumerate(texts), desc = "Creating PII Store"):
        # Find all email matches ONCE
        matches = [(m.group(), m.start(), m.end()) for m in EMAIL_REGEX.finditer(text)]

        # Track occurrences per email
        email_positions = defaultdict(list)
        for email, start, end in matches:
            email_positions[email].append((start, end))

        # Skip first occurrence, store rest
        for email, positions in email_positions.items():
            for start, end in positions[1:]:  # skip first
                email_occurrences[email].append((i, start, end))

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(email_occurrences, f, indent=4)
            

with open('email_vals/first_part.json', 'r') as file:
    first_part = json.load(file)

FIRST_PART_STORE = first_part['values']

with open('email_vals/domain.json', 'r') as file:
    domain_part = json.load(file)

DOMAIN_STORE = domain_part['values']

with open('email_vals/tld.json', 'r') as file:
    tld_part = json.load(file)

TLD_STORE = tld_part['values']

def mask_text(text, email, start, end, anchor, anchor_val):
    first_part, last_part = email.split('@')
    if '.' in first_part:
        parts = first_part.split('.')
        for i in range(len(parts)):
            parts[i] = random.sample(FIRST_PART_STORE, 1)[0]
        fp = '.'.join(parts)
    else:
        fp = random.sample(FIRST_PART_STORE, 1)[0]
    domain = random.sample(DOMAIN_STORE, 1)[0]
    tld = random.sample(TLD_STORE, 1)[0]

    email = ""
    if anchor == 'fp':
        email = f'{anchor_val}@{domain}.{tld}'
    elif anchor == 'domain':
        email = f'{fp}@{anchor_val}.{tld}'
    elif anchor == 'tld':
        email = f'{fp}@{domain}.{anchor_val}'


    updated_text = text[:start] + email + text[end:]
    return updated_text

def disect_email(email):
    first_part, last_part = email.split('@')
    last_part_splits = last_part.split('.')
    domain = last_part_splits[0]
    tld = ''
    for split in last_part_splits[1:]:
        tld += tld + split + '.'
    tld = tld[:-1]
    return first_part, domain, tld
    
def randomize_mask(dataset_path, output_path, data_key, pii_store_path = None):
    if pii_store_path == None:
        pii_store_path = "jsons/pii_store.json"
        save_all_email_locations(dataset_path, pii_store_path, data_key)


    with open(pii_store_path, 'r') as file:
        email_store = json.load(file)

    df = pd.read_csv(dataset_path)
    data = df[data_key].to_list()
    emails = email_store.keys()
    for email in tqdm(emails, desc = "Masking Emails"):
        fp, dom, tld = disect_email(email)
        occurrences = email_store[email]
        for occ in occurrences[1:]:
            index, start, end = occ
            text = data[index]
            anchor = random.sample(['fp', 'domain','tld'], 1)[0]
            if anchor == 'fp':
                masked_text = mask_text(text, email, start, end, anchor, fp)
            elif anchor == 'domain':
                masked_text = mask_text(text, email, start, end, anchor, dom)
            elif anchor == 'tld':
                masked_text = mask_text(text, email, start, end, anchor, tld)

            data[index] = masked_text

    df[data_key] = data
    dataset_folder = os.path.dirname(dataset_path)
    dataset_file = os.path.basename(dataset_path)

    output_file = os.path.join(
        dataset_folder,
        f"randomize_masked_{dataset_file}"
    )

    df.to_csv(output_file, index=False)


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
                