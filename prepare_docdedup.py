# prepare_enron.py
import pandas as pd
import sys

# Load your Enron CSV (after document-level dedup)
input_csv = '../datasets/enron/deduplicated_docs_enron_training_split.csv'  # Change this to your input file
output_txt = '../datasets/enron/enron_emails_docdedup.txt'

print(f"Loading {input_csv}...")
df = pd.read_csv(input_csv)

print(f"Found {len(df)} emails")
print(f"Writing to {output_txt}...")

# Write one email per line (newlines within emails become spaces)
with open(output_txt, 'w', encoding='utf-8') as f:
    for idx, email in enumerate(df['message']):
        # Replace internal newlines with space, write as single line
        email_single_line = str(email).replace('\n', ' ').replace('\r', ' ')
        f.write(email_single_line + '\n')
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} emails...")

print(f"Done! Converted {len(df)} emails to {output_txt}")
print(f"Each email is now a single line in the text file.")