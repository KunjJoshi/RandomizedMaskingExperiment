import pandas as pd
import re
import json

def extract_emails(text):
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    emails = [email.lower() for email in emails]
    return set(emails)

ORIGINAL_SET = set()
training_df = pd.read_csv('../datasets/enron/train_split.csv')
from_list = training_df['from'].to_list()
for email in from_list:
    ORIGINAL_SET.add(email)

def extract_emails_from_list(key):
    column = training_df[key].to_list()
    for em_list in column:
        emails = extract_emails(em_list)
        for email in emails:
            if email.strip() != '':
                ORIGINAL_SET.add(email)

extract_emails_from_list('to')
extract_emails_from_list('cc')
extract_emails_from_list('bcc')

print(len(ORIGINAL_SET))

base_path = 'results/gpt_base/110e3_leakage_discoverable_enron.csv'
rmft_path = 'results/gpt_rmft/110e3_leakage_discoverable_enron.csv'
dedup_path = 'results/gpt_dedup/110e3_leakage_discoverable_enron.csv'

base_df = pd.read_csv(base_path)
rmft_df = pd.read_csv(rmft_path)
dedup_df = pd.read_csv(dedup_path)

base_leaks = set(base_df['leaked_email'].to_list()) & ORIGINAL_SET
rmft_leaks = set(rmft_df['leaked_email'].to_list()) & ORIGINAL_SET
dedup_leaks = set(dedup_df['leaked_email'].to_list()) & ORIGINAL_SET

base_logprobs = base_df['logprob'].to_list()
rmft_logprobs = rmft_df['logprob'].to_list()
dedup_logprobs = dedup_df['logprob'].to_list()

prompts = pd.read_csv("../datasets/enron/discoverable_prompts.csv")['prompt'].to_list()

base_leak_prompts = set(base_df['prompt'].to_list())
rmft_leak_prompts = set(rmft_df['prompt'].to_list())
dedup_leak_prompts = set(dedup_df['prompt'].to_list())

print(f"=======SUMMARY==========")
print(f"\n Base Email Leaks: {(len(base_leaks)/len(ORIGINAL_SET))*100}")
print(f"\n RMFT Email Leaks: {(len(rmft_leaks)/len(ORIGINAL_SET))*100}")
print(f"\n Dedup Email Leaks: {(len(dedup_leaks)/len(ORIGINAL_SET))*100}")


print(f"Average Logprob Confidence per generation for Baseline Finetuning: {sum(base_logprobs)/len(base_logprobs)}")
print(f"Average Logprob Confidence per generation for RMFT: {sum(rmft_logprobs)/len(rmft_logprobs)}")
print(f"Average Logprob Confidence per generation for Deduplication: {sum(dedup_logprobs)/len(dedup_logprobs)}")

print(f"Number of prompts that leaked Emails for Baseline Finetuning: {len(base_leak_prompts)}/{len(prompts)}")
print(f"Number of prompts that leaked Emails for RMFT: {len(rmft_leak_prompts)}/{len(prompts)}")
print(f"Number of prompts that leaked Emails for Dedup: {len(dedup_leak_prompts)}/{len(prompts)}")
