# back_to_csv.py
import pandas as pd

# Input/output files
input_txt = '../datasets/enron/enron_emails_full_dedup.txt'  # Deduplicated text file
output_csv = '../datasets/enron/deduplicated_train_split.csv'  # Output CSV
original_csv = '../datasets/enron/deduplicated_docs_enron_training_split.csv'  # Original for comparison

print(f"Reading deduplicated emails from {input_txt}...")

# Read deduplicated emails (one per line)
with open(input_txt, 'r', encoding='utf-8') as f:
    dedup_emails = [line.strip() for line in f]

print(f"Found {len(dedup_emails)} deduplicated emails")

# Load original to compare
df_original = pd.read_csv(original_csv)
print(f"Original had {len(df_original)} emails")

# Create new dataframe with deduplicated emails
df_dedup = pd.DataFrame({'message': dedup_emails})

# Optional: If you have other columns (email_id, sender, date, etc.) 
# and want to preserve them, copy them over
# Assuming same order is maintained:
# for col in df_original.columns:
#     if col != 'body':
#         df_dedup[col] = df_original[col].iloc[:len(dedup_emails)]

# Save to CSV
print(f"Saving to {output_csv}...")
df_dedup.to_csv(output_csv, index=False)

print(f"\n{'='*60}")
print(f"Deduplication Complete!")
print(f"{'='*60}")

# Report statistics
original_chars = df_original['message'].astype(str).str.len().sum()
dedup_chars = df_dedup['message'].astype(str).str.len().sum()
reduction_pct = (1 - dedup_chars/original_chars) * 100

print(f"Original emails: {len(df_original):,}")
print(f"Deduplicated emails: {len(df_dedup):,}")
print(f"Emails removed: {len(df_original) - len(df_dedup):,}")
print(f"\nOriginal total characters: {original_chars:,}")
print(f"Deduplicated total characters: {dedup_chars:,}")
print(f"Character reduction: {reduction_pct:.1f}%")
print(f"\nSaved to: {output_csv}")