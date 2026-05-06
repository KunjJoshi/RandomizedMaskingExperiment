import pandas as pd
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import math
import torch
from tqdm import tqdm

CONTROLLED_PROMPTING_SETUP = "jsons/context_extraction.json"
with open(CONTROLLED_PROMPTING_SETUP, 'r') as file:
    controlled_prompts = json.load(file)

def extract_emails(text):
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    emails = [email.lower() for email in emails]
    return set(emails)

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

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

def count_leading_pad_tokens(row_1d, pad_id):
    """Left-padded batches: real tokens start after any leading pad_token_id cells."""
    if pad_id is None:
        return 0
    n = 0
    for t in row_1d:
        if int(t.item()) != int(pad_id):
            break
        n += 1
    return n

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

def leakage_context(ckpt, token_word, prompt_list, max_length = 64):
    try:
        print(f'Processing {ckpt}')

        tokenizer = load_tokenizer(ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            return_dict=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        model.eval()
        device = next(model.parameters()).device

        for i in tqdm(range(prompt_list), desc = "Controlled Extraction"):
            email_struct = prompt_list[i]
            email_to_leak = email_struct['email']
            stages = email_struct['stages']
            leak_dict = {}
            fake_logprobs = []
            true_logprobs = []
            inputs = tokenizer(stages, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=False)
            outputs_cpu = outputs.cpu()
            del outputs
            torch.cuda.empty_cache()

            generations = tokenizer.batch_decode(outputs_cpu, skip_special_tokens=True)
            pad_id = tokenizer.pad_token_id

            for no, generation in enumerate(generations):
                row_ids = outputs_cpu[no]
                leading_pad = count_leading_pad_tokens(row_ids, pad_id)
                emails_leaked = extract_emails(generation)
                for email in emails_leaked:
                    if email.lower().strip() == email_to_leak.lower().strip():
                        leak_dict['stage'] = no
                        avg_logprobs = 0
                        if len(fake_logprobs) >= 1:
                            avg_logprobs = sum(fake_logprobs)/len(fake_logprobs)
                        leak_dict['fake_logprobs'] = avg_logprobs
                        for char_start, char_end in find_all_indices(generation, email):
                            tok_start, tok_end = get_token_indices(
                                tokenizer, generation, char_start, char_end
                            )
                            if tok_start is None or tok_end is None:
                                continue
                            a_start = leading_pad + tok_start
                            a_end = leading_pad + tok_end
                            # Causal LM: no predecessor logit for the first position of the full sequence
                            if a_start == 0:
                                continue

                            logprob = compute_logprob_for_span(
                                model, tokenizer, row_ids, a_start, a_end, device
                            )
                            true_logprobs.append(logprob)
                        avg_logprob = 0
                        if len(true_logprobs) >= 1:
                            avg_logprob = sum(true_logprobs)/len(true_logprobs)
                        leak_dict['true_logprobs'] = avg_logprob
                        email_struct[token_word] = leak_dict
                        prompt_list[i] = email_struct
                    else:
                        email_logprobs = []
                        for char_start, char_end in find_all_indices(generation, email):
                            tok_start, tok_end = get_token_indices(
                                tokenizer, generation, char_start, char_end
                            )
                            if tok_start is None or tok_end is None:
                                continue
                            a_start = leading_pad + tok_start
                            a_end = leading_pad + tok_end
                            # Causal LM: no predecessor logit for the first position of the full sequence
                            if a_start == 0:
                                continue

                            logprob = compute_logprob_for_span(
                                model, tokenizer, row_ids, a_start, a_end, device
                            )
                            email_logprobs.append(logprob)
                        logprobs = sum(email_logprobs)/len(email_logprobs)
                        fake_logprobs.append(logprobs)
        return prompt_list
    except Exception as e:
        print(f"Procedure failed due to {str(e)}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Data using a Testing Dataset")

    parser.add_argument("--checkpoint", type=str, required=True, help="Collection of models")
    parser.add_argument("--token_word", type=str, default="gpt_base", help="Type of training/ token word")
    parser.add_argument("--max_length", type=int, help="Max Length")
    args = parser.parse_args()

    token_pl = leakage_context(ckpt = parser.checkpoint, token_word = parser.token_word, prompt_list = controlled_prompts, max_length = parser.max_length)
    with open(f'jsons/context_extraction_{parser.token_word}.json', 'w') as file:
        json.dump(token_pl, file, indent=4)


