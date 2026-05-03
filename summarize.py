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

base_leaks = set(base_df['leaked_email'].to_list())
rmft_leaks = set(rmft_df['leaked_email'].to_list())
dedup_leaks = set(dedup_df['leaked_email'].to_list())

base_logprobs_og = base_df[base_df['leaked_email'] in (base_leaks & ORIGINAL_SET)]['logprob'].to_list()
rmft_logprobs_og = rmft_df[rmft_df['leaked_email'] in (rmft_leaks & ORIGINAL_SET)]['logprob'].to_list()
dedup_logprobs_og = dedup_df[dedup_df['leaked_email'] in (dedup_leaks & ORIGINAL_SET)]['logprob'].to_list()

base_logprobs_fake = base_df[base_df['leaked_email'] in (base_leaks - ORIGINAL_SET)]['logprob'].to_list()
rmft_logprobs_fake = rmft_df[rmft_df['leaked_email'] in (rmft_leaks - ORIGINAL_SET)]['logprob'].to_list()
dedup_logprobs_fake = dedup_df[dedup_df['leaked_email'] in (dedup_leaks - ORIGINAL_SET)]['logprob'].to_list()

if len(base_logprobs_og) > 0 and len(base_logprobs_fake) > 0:
    avg_og = sum(base_logprobs_og)/len(base_logprobs_og)
    avg_fake = sum(base_logprobs_fake)/len(base_logprobs_fake)
    base_logdiff = avg_og - avg_fake
else:
    base_logdiff = 0

if len(rmft_logprobs_og) > 0 and len(rmft_logprobs_fake) > 0:
    avg_og = sum(rmft_logprobs_og)/len(rmft_logprobs_og)
    avg_fake = sum(rmft_logprobs_fake)/len(rmft_logprobs_fake)
    rmft_logdiff = avg_og - avg_fake
else:
    rmft_logdiff = 0

if len(dedup_logprobs_og) > 0 and len(dedup_logprobs_fake) > 0:
    avg_og = sum(dedup_logprobs_og)/len(dedup_logprobs_og)
    avg_fake = sum(dedup_logprobs_fake)/len(dedup_logprobs_fake)
    dedup_logdiff = avg_og - avg_fake
else:
    dedup_logdiff = 0

prompts = pd.read_csv("../datasets/enron/discoverable_prompts.csv")['prompt'].to_list()

base_leak_prompts = set(base_df['prompt'].to_list())
rmft_leak_prompts = set(rmft_df['prompt'].to_list())
dedup_leak_prompts = set(dedup_df['prompt'].to_list())

print(f"=======SUMMARY==========")
print(f"\n Base Email Leaks: {(len(base_leaks)/len(ORIGINAL_SET))*100}")
print(f"\n RMFT Email Leaks: {(len(rmft_leaks)/len(ORIGINAL_SET))*100}")
print(f"\n Dedup Email Leaks: {(len(dedup_leaks)/len(ORIGINAL_SET))*100}")


if base_logdiff > 0:
    base_ps = f"Baseline FT prefers Original Emails over Fake emails with Logdiff: {base_logdiff}"
    base_ps += f"\n Average LogDiff on Original Emails: {sum(base_logprobs_og)/len(base_logprobs_og)}"
    base_ps += f"\n Average LogDiff on Fake Emails: {sum(base_logprobs_fake)/len(base_logprobs_fake)}"
elif base_logdiff < 0:
    base_ps = f"Baseline FT prefers Fake Emails over Original emails with Logdiff: {base_logdiff}"
    base_ps += f"\n Average LogDiff on Original Emails: {sum(base_logprobs_og)/len(base_logprobs_og)}"
    base_ps += f"\n Average LogDiff on Fake Emails: {sum(base_logprobs_fake)/len(base_logprobs_fake)}"
else:
    base_ps = f"Baseline FT did not generate one kind of emails at all"
    base_ps += f"Length of Original Emails generated: {len(base_logprobs_og)}"
    base_ps += f"Length of Fake Emails generated: {len(base_logprobs_fake)}"

if rmft_logdiff > 0:
    rmft_ps = f"RMFT prefers Original Emails over Fake emails with Logdiff: {rmft_logdiff}"
    rmft_ps += f"\n Average LogDiff on Original Emails: {sum(rmft_logprobs_og)/len(rmft_logprobs_og)}"
    rmft_ps += f"\n Average LogDiff on Fake Emails: {sum(rmft_logprobs_fake)/len(rmft_logprobs_fake)}"
elif rmft_logdiff < 0:
    rmft_ps = f"RMFT prefers Fake Emails over Original emails with Logdiff: {rmft_logdiff}"
    rmft_ps += f"\n Average LogDiff on Original Emails: {sum(rmft_logprobs_og)/len(rmft_logprobs_og)}"
    rmft_ps += f"\n Average LogDiff on Fake Emails: {sum(rmft_logprobs_fake)/len(rmft_logprobs_fake)}"
else:
    rmft_ps = f"RMFT did not generate one kind of emails at all"
    rmft_ps += f"Length of Original Emails generated: {len(rmft_logprobs_og)}"
    rmft_ps += f"Length of Fake Emails generated: {len(rmft_logprobs_fake)}"

if dedup_logdiff > 0:
    dedup_ps = f"Dedup FT prefers Original Emails over Fake emails with Logdiff: {dedup_logdiff}"
    dedup_ps += f"\n Average LogDiff on Original Emails: {sum(dedup_logprobs_og)/len(dedup_logprobs_og)}"
    dedup_ps += f"\n Average LogDiff on Fake Emails: {sum(dedup_logprobs_fake)/len(dedup_logprobs_fake)}"
elif dedup_logdiff < 0:
    dedup_ps = f"Dedup FT prefers Fake Emails over Original emails with Logdiff: {dedup_logdiff}"
    dedup_ps += f"\n Average LogDiff on Original Emails: {sum(dedup_logprobs_og)/len(dedup_logprobs_og)}"
    dedup_ps += f"\n Average LogDiff on Fake Emails: {sum(dedup_logprobs_fake)/len(dedup_logprobs_fake)}"
else:
    dedup_ps = f"Dedup FT did not generate one kind of emails at all"
    dedup_ps += f"Length of Original Emails generated: {len(dedup_logprobs_og)}"
    dedup_ps += f"Length of Fake Emails generated: {len(dedup_logprobs_fake)}"

print(f"\n===========SUMMARY OF AVG LOGDIFF=========\n")
print("Baseline Finetuning\n ")
print(base_ps)
print("RMFT \n")
print(rmft_ps)
print("Dedup \n")
print(dedup_ps)

print(f"Number of prompts that leaked Emails for Baseline Finetuning: {len(base_leak_prompts)}/{len(prompts)}")
print(f"Number of prompts that leaked Emails for RMFT: {len(rmft_leak_prompts)}/{len(prompts)}")
print(f"Number of prompts that leaked Emails for Dedup: {len(dedup_leak_prompts)}/{len(prompts)}")
