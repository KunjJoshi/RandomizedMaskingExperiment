import pandas as pd
import re
from transformers import AutoTokenizer
import json
from tqdm import tqdm

def email_occurrence_map(all_emails):
    occ_map = {}
    for email in all_emails:
        if email in occ_map.keys():
            occ_map[email] += 1
        else:
            occ_map[email] = 1
    occ_map = dict(sorted(occ_map.items(), key=lambda x: x[1]))
    return occ_map

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def generate_stages(tokens):
    full_word = ""
    stages = []
    for tok in tokens:
        full_word += tok
        stages.append(full_word)
    return stages
def generate_tokens_map(emails, model_name: str = 'gpt2-xl'):
    tokenizer = load_tokenizer(model_name)
    tokens_map = []
    for email in tqdm(emails, desc = "Tokens Map through Emails"):
        email_map = {}
        email_map['email'] = email
        inputs = tokenizer.encode(email, return_tensors='pt')
        tokens = list(tokenizer.convert_ids_to_tokens(inputs[0]))
        stages = generate_stages(tokens)
        email_map['stages'] = stages
        tokens_map.append(email_map)
    return tokens_map

def pipeline():
    base_df = pd.read_csv("results/gpt_base/110e3_leakage_discoverable_enron.csv")
    leaked_emails = base_df['leaked_email'].to_list()
    occ_map = email_occurrence_map(leaked_emails)
    email_keys = list(occ_map.keys())
    if len(email_keys) > 100:
        email_keys = email_keys[:100]
    tokens_map = generate_tokens_map(email_keys)
    with open('jsons/context_extraction.json', 'w') as file:
        json.dump(tokens_map, file, indent=4)

if __name__ == "__main__":
    pipeline()

        
