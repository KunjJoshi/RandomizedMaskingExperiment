import pandas as pd
import random
import re
import json

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

from typing import List, Dict
import math
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd

def load_model_and_tokenizer(model_name, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available() and (
        device is None or str(device).startswith("cuda")
    )
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model.eval()
        actual_device = next(model.parameters()).device
    else:
        dev = device if device is not None else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        model.to(dev)
        actual_device = torch.device(dev) if isinstance(dev, str) else dev

    return tokenizer, model, actual_device

def calculate_batch_perplexity(
    prompts: List[str],
    model_name: str,
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
                    # On very OOD text, (nll / n_tokens) can be large enough that exp() overflows to inf.
                    # Cap in log-space at the maximum finite float exponent.
                    log_ppl = nll / n_tokens
                    max_log = math.log(sys.float_info.max)  # ~709.78
                    ppl = math.exp(min(log_ppl, max_log))
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

def re_eval_perp(
    ckpt,
    token_word,
    prompt_path,
    prompt_key,
    test_split,
    batch_size=50,
    max_length=256,
):
    prompts = pd.read_csv(prompt_path)[prompt_key].to_list()
    prompts = [
        str(prompt)
        for prompt in prompts
        if isinstance(prompt, str) and prompt.strip() != ""
    ]
    modelname = os.path.basename(os.path.normpath(ckpt))
    print(f"Processing {modelname} ({ckpt})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = calculate_batch_perplexity(
        prompts,
        model_name=ckpt,
        batch_size=batch_size,
        device=device,
        max_length_override=max_length,
    )
    perplexities = {modelname: outputs}
    os.makedirs("jsons", exist_ok=True)
    out_path = f"jsons/perplexities_{token_word}_{test_split}.json"
    with open(out_path, "w") as file:
        json.dump(perplexities, file, indent=4)
    print(f"Wrote {out_path}")


import argparse
import gc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perplexity for a single checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path or HF id for one model checkpoint",
    )
    parser.add_argument(
        "--training_type",
        type=str,
        default="gpt_base",
        help="Tag for output filename (token_word)",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="CSV with prompts",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        required=True,
        help="Column name for prompt text",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        required=True,
        help="Tag for output filename",
    )
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    re_eval_perp(
        ckpt=args.checkpoint,
        token_word=args.training_type,
        prompt_path=args.prompt_path,
        prompt_key=args.prompt_key,
        test_split=args.test_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    torch.cuda.empty_cache()
    gc.collect()

