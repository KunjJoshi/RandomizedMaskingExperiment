import pandas as pd
import random
import re
import os
import json
import torch.nn.functional as F
import math

def sample_random_prompts(texts, num_samples=25000, prompt_length=10):
    prompts = []
    for text in texts:
        tokens = text.split()
        
        if len(tokens) >= prompt_length:
            start_index = random.randint(0, len(tokens) - prompt_length)
            prompt = tokens[start_index:start_index + prompt_length]
            prompts.append(' '.join(prompt))
        
        if len(prompts) >= num_samples:
            break
    
    return prompts

import re

def parse_wet_file(file_path, max_docs=None):
    """
    Parse a WET file and return English language documents.
    
    Args:
        file_path (str): Path to the .wet file
        max_docs (int, optional): limit how many docs to parse (useful for sampling)
    
    Returns:
        list of str: extracted English documents
    """
    texts = []
    doc_lines = []
    keep_doc = False
    doc_started = False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("WARC/1.0"):
                # save the previous doc if it's English
                if keep_doc and doc_lines:
                    texts.append("".join(doc_lines).strip())
                    if max_docs and len(texts) >= max_docs:
                        break
                # reset for new doc
                doc_lines = []
                keep_doc = False
                doc_started = True

            if "WARC-Identified-Content-Language: eng" in line:
                keep_doc = True

            if doc_started and not line.startswith("WARC/") and not line.startswith("Content-Length"):
                doc_lines.append(line)

        # last doc
        if keep_doc and doc_lines:
            texts.append("".join(doc_lines).strip())

    return texts


def save_prompts(save_path):
    texts = parse_wet_file('crawl.wet')
    prompts = sample_random_prompts(texts)
    promptDF = {'prompt': prompts}
    df = pd.DataFrame(promptDF)
    df.to_csv(save_path, index = False)


import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

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

def num_times_email_leaked(emails, email):
    num_times = 0
    for mail in emails:
        if mail == email:
            num_times += 1
    return num_times


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

import gc

def get_token_indices(tokenizer, text, char_start, char_end):
    enc     = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    tok_start, tok_end = None, None
    for idx, (cs, ce) in enumerate(offsets):
        if tok_start is None and cs >= char_start:
            tok_start = idx
        if ce <= char_end:
            tok_end = idx
    return tok_start, tok_end

def find_all_indices(text, substring):
    occurrences = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(substring)))
        start = idx + 1
    return occurrences

def compute_logprob_for_span(model, tokenizer, output_ids, tok_start, tok_end, device):
    """
    Compute sum of log-probs for tokens [tok_start, tok_end] (inclusive)
    by running a forward pass on only that single sequence — avoids
    materialising the full [B, T, V] tensor for the whole batch.
    """
    # output_ids: 1-D tensor (single sequence)
    seq = output_ids.unsqueeze(0).to(device)          # [1, T]
    with torch.no_grad():
        logits = model(seq).logits                    # [1, T, V]
        # Shift: logits[t] predicts token[t+1]
        log_probs = F.log_softmax(logits[0], dim=-1)  # [T, V]  — still single seq
    
    token_ids  = output_ids[tok_start : tok_end + 1]  # [span]
    score_pos  = log_probs[tok_start - 1 : tok_end]   # [span, V]
    per_tok_lp = score_pos.gather(1, token_ids.unsqueeze(1).to(device)).squeeze(1)
    
    result = per_tok_lp.sum().item()
    
    # Free immediately — don't let logits linger on GPU
    del seq, logits, log_probs, score_pos, per_tok_lp
    torch.cuda.empty_cache()
    
    return result

def re_evaluate(model_coll, token_word, testing_dataset_path, test_key, test_split_name, batch_size=60, max_length=64):  # ← batch_size 250 → 8
    models    = [os.path.join(model_coll, m) for m in os.listdir(model_coll)]
    prompt_df = pd.read_csv(testing_dataset_path)
    prompts   = prompt_df[test_key].to_list()
    if len(prompts) > 2500:
        prompts = prompts[:2500]

    for ckpt in models:
        try:
            print(f'Processing {ckpt}')
            modelname = ckpt.split('/')[-1]

            tokenizer = load_tokenizer(ckpt)
            model = AutoModelForCausalLM.from_pretrained(
                ckpt,
                return_dict=True,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            model.eval()
            device = next(model.parameters()).device  # ← respect device_map

            directory = f'results/{token_word}'
            os.makedirs(directory, exist_ok=True)
            leakageCsv = f'results/{token_word}/{modelname}_leakage_{test_split_name}.csv'
            summary = {'prompt': [], 'generation': [], 'leaked_email': [], 'logprob': []}

            for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting Emails"):
                batched_prompts = prompts[i : i + batch_size]

                inputs = tokenizer(
                    batched_prompts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=True,               # ← needed for batched generation
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,  # ← max_new_tokens avoids re-counting prompt
                        do_sample=False,
                    )
                outputs_cpu = outputs.cpu()         # ← pull to CPU before the scoring loop
                del outputs
                torch.cuda.empty_cache()

                generations = tokenizer.batch_decode(outputs_cpu, skip_special_tokens=True)

                for no, generation in enumerate(generations):
                    emails_leaked = extract_emails(generation)
                    for email in emails_leaked:
                        for char_start, char_end in find_all_indices(generation, email):
                            tok_start, tok_end = get_token_indices(
                                tokenizer, generation, char_start, char_end
                            )
                            if tok_start is None or tok_start == 0:
                                continue

                            logprob = compute_logprob_for_span(
                                model, tokenizer, outputs_cpu[no], tok_start, tok_end, device
                            )

                            summary['prompt'].append(batched_prompts[no])
                            summary['generation'].append(generation)
                            summary['leaked_email'].append(email)
                            summary['logprob'].append(logprob)

                del inputs, outputs_cpu, generations
                gc.collect()
                torch.cuda.empty_cache()

            pd.DataFrame(summary).to_csv(leakageCsv, index=False)

            # Free model between checkpoints
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Exception Encountered: {e}. Skipping Checkpoint {ckpt}")
            continue




import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Data using a Testing Dataset")

    parser.add_argument("--model_collection_path", type=str, required=True, help="Collection of models")
    parser.add_argument("--training_type", type=str, default="gpt_base", help="Type of training/ token word")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to testing CSV file")
    parser.add_argument("--test_key", type=str, required=True, help="Column name for training text")
    parser.add_argument("--test_split", type=str, required=True, help="Token word you want to give for this testing dataset")
    parser.add_argument("--batch_size", type=int, help="Batch Size for Email Extraction")
    args = parser.parse_args()

    re_evaluate(model_coll = args.model_collection_path, token_word = args.training_type, testing_dataset_path = args.test_dataset_path, test_key = args.test_key, test_split_name = args.test_split, batch_size = args.batch_size)