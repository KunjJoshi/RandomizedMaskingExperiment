import pandas as pd
import random
import re

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


prompt_df_path = "../../NGramMemorization/compExp/promptsAndGenerations.csv"

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

print(f"Number of Unique Emails found: {len(ORIGINAL_SET)}")
def num_times_email_leaked(emails, email):
    num_times = 0
    for mail in emails:
        if mail == email:
            num_times += 1
    return num_times

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
tokenizer.pad_token = tokenizer.eos_token

from typing import List, Dict
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd

def load_model_and_tokenizer(model_name="gpt2-xl", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # GPT2 tokenizer has no pad token by default -> set to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return tokenizer, model, device

def calculate_batch_perplexity(
    prompts: List[str],
    model_name: str = "gpt2-xl",
    batch_size: int = 8,
    device: str = None,
    truncate_long: bool = True,      # if True, truncate tokenized prompt to model max length
    max_length_override: int = None  # if provided, override model config max length
) -> pd.DataFrame:
    tokenizer, model, device = load_model_and_tokenizer(model_name, device)
    model_max_len = model.config.max_position_embeddings
    if max_length_override is not None:
        model_max_len = min(model_max_len, max_length_override)

    results = []
    # We'll process prompts in batches, but first tokenize per prompt to know lengths
    # For simplicity we truncate very long prompts (you can request sliding-window approach)
    encodings = []
    for p in prompts:
        enc = tokenizer.encode(p, add_special_tokens=False)
        if len(enc) == 0:
            # avoid zero-length input
            enc = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        if truncate_long and len(enc) > model_max_len:
            enc = enc[-model_max_len:]  # keep last tokens (right-truncate) — typical for causal models
        encodings.append(torch.tensor(enc, dtype=torch.long))

    # Process in batches
    for i in tqdm(range(0, len(encodings), batch_size), desc="Batches"):
        batch_enc = encodings[i : i + batch_size]
        # pad to same length in batch
        lengths = [e.size(0) for e in batch_enc]
        batch_max_len = max(lengths)
        input_ids = torch.full((len(batch_enc), batch_max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_enc), batch_max_len), dtype=torch.long)
        for j, e in enumerate(batch_enc):
            input_ids[j, : e.size(0)] = e
            attention_mask[j, : e.size(0)] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            # Get logits: (batch, seq_len, vocab)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()  # (batch, seq_len-1, vocab)
            shift_labels = input_ids[..., 1:].contiguous()   # (batch, seq_len-1)
            shift_mask = attention_mask[..., 1:].contiguous()  # (batch, seq_len-1)

            b, seqm1, vocab = shift_logits.shape
            # Flatten for cross_entropy
            flat_logits = shift_logits.view(-1, vocab)
            flat_labels = shift_labels.view(-1)

            # Compute per-token loss (no reduction)
            # cross_entropy expects class indices in [0..vocab-1]; we will mask pad tokens below
            losses_flat = F.cross_entropy(flat_logits, flat_labels, reduction='none')  # (b*(seqm1),)
            losses = losses_flat.view(b, seqm1)  # (batch, seq_len-1)

            # Mask out tokens where shift_labels == pad_token_id (we don't want to count pad tokens)
            pad_mask = (shift_labels == tokenizer.pad_token_id)
            losses = losses.masked_fill(pad_mask, 0.0)
            token_counts = shift_mask.sum(dim=1)  # number of tokens contributing to loss per sample

            # Sum nll per sequence
            nll_per_seq = losses.sum(dim=1).cpu().tolist()
            token_counts = token_counts.cpu().tolist()

            # Compute perplexity
            for j in range(len(batch_enc)):
                nll = float(nll_per_seq[j])
                n_tokens = int(token_counts[j])
                if n_tokens <= 0:
                    # fallback (shouldn't happen for non-empty prompts)
                    ppl = float("inf")
                else:
                    ppl = math.exp(nll / n_tokens)
                prompt_text = tokenizer.decode(batch_enc[j].cpu().tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True)
                results.append({
                    "prompt_index": i + j,
                    "prompt": prompt_text,
                    "n_tokens": n_tokens,
                    "nll": nll,
                    "ppl": ppl
                })

    sorted_data = sorted(results, key=lambda x: x["prompt_index"])
    return sorted_data

def re_eval_perp(model_coll, token_word, prompt_path, prompt_key, batch_size = 50, max_length = 256):
    prompts = pd.read_csv(prompt_path)[prompt_key].to_list()
    prompts = [str(prompt) for prompt in prompts if type(prompt)==str and prompt.strip() != ""]
    models = [os.path.join(model_coll, model) for model in os.listdir(model_coll)]
    perplexities = {}
    for model in models:
        modelname = model.split('/')[-1]
        print(f'Processing {modelname}')
        outputs = calculate_batch_perplexity(prompts, model_name = model, batch_size = 5, device = 'cuda', max_length_override = 4096)
        perplexities[modelname] = outputs
    with open(f'jsons/perplexities_{token_word}.json', 'w') as file:
        json.dump(perplexities, file, indent=4)

re_eval_perp('../models/gpt_base', 'gpt_base', '../datasets/enron/test_split.csv', 'message')
re_eval_perp('../models/gpt_rmft', 'gpt_rmft', '../datasets/enron/test_split.csv', 'message')
re_eval_perp('../models/gpt_dedup', 'gpt_dedup', '../datasets/enron/test_split.csv', 'message')

