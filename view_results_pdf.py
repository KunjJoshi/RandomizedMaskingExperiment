import pandas as pd
import random
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import json
import numpy as np

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

perplexity_path = "jsons"
data_seen_path = "../data_seen"
extraction_path = "results"

def data_seen_per_checkpoint(token_word):
    direc = os.path.join(data_seen_path, token_word)
    files = [os.path.join(direc, file) for file in os.listdir(direc)]
    checkpoints = {}
    emails_seen = set()
    for filename in files:
        fn = filename.split('.')[2]
        ckpt = fn.split('/')[-1]
        with open(filename, 'r') as file:
            data = file.read()
        emails_at_this_ckpt = extract_emails(data)
        emails_seen = emails_seen | emails_at_this_ckpt
        checkpoints[ckpt] = emails_seen
    return checkpoints

def convert_to_numerical(ckpt):
    iter,epoch = ckpt.split('e')
    number = (int(epoch) - 1) * 100 + int(iter)
    return number

def graph_friendly_data_ppl(token_word, test_split):
    json_file = os.path.join(perplexity_path, f'perplexities_{token_word}_{test_split}.json')
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    checkpoints = list(data.keys())
    ckpt_map = {}
    for i in range(len(checkpoints)):
        number = convert_to_numerical(checkpoints[i])
        ckpt_map[checkpoints[i]] = number
        checkpoints[i] = number
    checkpoints = sorted(checkpoints)
    avg_perplexities = {}
    for ckpt in list(data.keys()):
        collection = data[ckpt]
        ppl = 0
        for coll in collection:
            ppl += coll['ppl']
        avg_ppl = ppl/len(collection)
        num = ckpt_map[ckpt]
        avg_perplexities[num] = avg_ppl
    avg_perplexities = dict(sorted(avg_perplexities.items()))
    return avg_perplexities


def mdp_graph_friendly_data(base_word, model_word, test_split):
    base_file = os.path.join(perplexity_path, f'perplexities_{base_word}_{test_split}.json')
    rmft_file = os.path.join(perplexity_path, f'perplexities_{model_word}_{test_split}.json')

    with open(base_file, 'r') as file:
        base_data = json.load(file)
    with open(rmft_file, 'r') as file:
        rmft_data = json.load(file)
    mdp_values = {}
    for key in list(base_data.keys()):
        number = convert_to_numerical(key)
        base_ppls = base_data[key]
        model_ppls = rmft_data[key]
        ppl_diffs = []
        for i in range(len(base_ppls)):
            diff = base_ppls[i]['ppl'] - model_ppls[i]['ppl']
            ppl_diffs.append(diff)
        mdp_values[number] = sum(ppl_diffs)/len(ppl_diffs)
    mdp_values = dict(sorted(mdp_values.items()))
    return mdp_values

def get_emails_from_a_column(df, column):
    col = df[column].to_list()
    email_set = set()
    for text in col:
        emails = extract_emails(text)
        for em in emails:
            email_set.add(em)
    return email_set

def extraction_rates(token_word, test_split):
    seen_data = data_seen_per_checkpoint('gpt_base')
    extraction_rates = {}
    directory = os.path.join(extraction_path, token_word)
    for epoch in range(1,4):
        for iter in range(10,110,10):
            name = f'{iter}e{epoch}'
            key = f'{epoch}e{iter}'
            number = convert_to_numerical(name)
            leakage_file = os.path.join(directory,f'{name}_leakage_{test_split}.csv')
            df = pd.read_csv(leakage_file)
            leaked_emails = set(df['leaked_email'].to_list())
            ter_leaked = leaked_emails & ORIGINAL_SET
            ter = (len(ter_leaked)/len(ORIGINAL_SET))*100
            ckpt_seen = seen_data[key]
            ser_leaked = leaked_emails & ckpt_seen
            ser = (len(ser_leaked)/len(ckpt_seen)) * 100
            er = {'ter':ter, 'ser':ser}
            extraction_rates[number] = er
    extraction_rates = dict(sorted(extraction_rates.items()))
    return extraction_rates

def tegr(token_word, test_split):
    extraction_rates = {}
    directory = os.path.join(extraction_path, token_word)
    for epoch in range(1,4):
        for iter in range(10,110,10):
            name = f'{iter}e{epoch}'
            key = f'{epoch}e{iter}'
            number = convert_to_numerical(name)
            leakage_file = os.path.join(directory,f'{name}_leakage_{test_split}.csv')
            df = pd.read_csv(leakage_file)
            leaked_emails = set(df['leaked_email'].to_list())
            if len(leaked_emails)>0:
                ter_leaked = leaked_emails & ORIGINAL_SET
                tegr = (len(ter_leaked)/len(leaked_emails))*100
            else:
                tegr = 0
            extraction_rates[number] = tegr
    extraction_rates = dict(sorted(extraction_rates.items()))
    return extraction_rates



def logprobs(token_word, test_split):
    logprobs_gp = {}
    directory = os.path.join(extraction_path, token_word)
    for epoch in range(1,4):
        for iter in range(10,110,10):
            name = f'{iter}e{epoch}'
            key = f'{epoch}e{iter}'
            number = convert_to_numerical(name)
            leakage_file = os.path.join(directory,f'{name}_leakage_{test_split}.csv')
            df = pd.read_csv(leakage_file)
            logprobs_col = df['logprob'].to_list()
            emails = df['leaked_email'].to_list()
            
            og_email_probs = []
            non_og_email_probs = []
            
            for i in range(len(logprobs_col)):
                if emails[i] in ORIGINAL_SET:
                    og_email_probs.append(logprobs_col[i])  # ← RAW logprob (negative)
                else:
                    non_og_email_probs.append(logprobs_col[i])
            
            # Calculate averages (handle empty lists)
            if len(og_email_probs) > 0:
                og_avg_probs = sum(og_email_probs) / len(og_email_probs)
            else:
                og_avg_probs = None  # ← Use None instead of np.NaN
            
            if len(non_og_email_probs) > 0:
                non_og_avg_probs = sum(non_og_email_probs) / len(non_og_email_probs)
            else:
                non_og_avg_probs = None
            
            lp = {
                'true_emails': og_avg_probs, 
                'fake_emails': non_og_avg_probs
            }
            logprobs_gp[number] = lp
            
    logprobs_gp = dict(sorted(logprobs_gp.items()))
    return logprobs_gp

def maxtegr(tegr_values, mdp_values, taus):
    tegr_values = np.array(tegr_values)
    mdp_values = np.array(mdp_values)

    results = []
    for tau in taus:
        valid_indices = np.where(mdp_values <= tau)[0]

        if len(valid_indices) == 0:
            results.append(np.nan)
        else:
            results.append(np.max(tegr_values[valid_indices]))

    return results

def fetch_lists(rates_dict, key1, key2):
    ter_list = []
    ser_list = []
    for key in list(rates_dict.keys()):
        er = rates_dict[key]
        ter_list.append(er[key1])
        ser_list.append(er[key2])
    return ter_list, ser_list
    
def plot_all(base_token_word, rmft_token_word, dedup_token_word, test_split, model_name):
    try:
        base_ppl = graph_friendly_data_ppl(base_token_word, test_split)
        rmft_ppl = graph_friendly_data_ppl(rmft_token_word, test_split)
        dedup_ppl = graph_friendly_data_ppl(dedup_token_word, test_split)

        base_ppl = {key: val for key, val in base_ppl.items() if key != 10}
        rmft_ppl = {key: val for key, val in rmft_ppl.items() if key != 10}
        dedup_ppl = {key: val for key, val in dedup_ppl.items() if key != 10}
    
        plt.figure(figsize=(10,6))
        plt.plot(base_ppl.keys(), base_ppl.values(), marker = 'o', label = 'Base Finetuning')
        plt.plot(rmft_ppl.keys(), rmft_ppl.values(), marker='o', label = 'Randomized Masked Finetuning')
        plt.plot(dedup_ppl.keys(), dedup_ppl.values(), marker='o', label='Deduplicated Finetuning')
        plt.xlabel('Training Checkpoints', fontsize=20)
        plt.ylabel(f'Perplexity on {test_split}', fontsize=20)
        plt.title(f'Comparing {model_name} Perplexities on {test_split}', fontsize=25)
        plt.legend(fontsize = 18, frameon = True)
        plt.tick_params(labelsize=16)
        plt.savefig(f'pdfs/perplexity_{test_split}_{model_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Failed to draw the Perplexity Graph: {e}")

    try:
        mdp_rmft = mdp_graph_friendly_data(base_token_word, rmft_token_word, test_split)
        mdp_dedup = mdp_graph_friendly_data(base_token_word, dedup_token_word, test_split)

        mdp_rmft = {key: val for key, val in mdp_rmft.items() if key != 10}
        mdp_dedup = {key: val for key, val in mdp_dedup.items() if key != 10}

        plt.figure(figsize=(10,6))
        plt.plot(mdp_rmft.keys(), mdp_rmft.values(), marker='o', label = 'Randomized Masked Finetuning')
        plt.plot(mdp_dedup.keys(), mdp_dedup.values(), marker='o', label='Deduplicated Finetuning')
        plt.xlabel('Training Checkpoints', fontsize=20)
        plt.ylabel(f'Mean Delta Perplexity on {test_split}', fontsize=20)
        plt.title(f'Comparing {model_name} MDP on {test_split}', fontsize=25)
        plt.legend(fontsize = 18, frameon = True)
        plt.tick_params(labelsize=16)
        plt.savefig(f'pdfs/mdp_{test_split}_{model_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Failed to draw MDP Graph: {e}")

    try:
        lp_base = logprobs(base_token_word, test_split)
        lp_rmft = logprobs(rmft_token_word, test_split)
        lp_dedup = logprobs(dedup_token_word, test_split)
    
        checkpoints = sorted(lp_base.keys())
    
        # Extract values, filtering out None
        def extract_values(lp_dict, key):
            values = []
            ckpts = []
            for ckpt in checkpoints:
                val = lp_dict[ckpt][key]
                if val is not None:  # Skip None values
                    values.append(val)
                    ckpts.append(ckpt)
            return ckpts, values
    
        base_ckpts_true, base_true = extract_values(lp_base, 'true_emails')
        base_ckpts_fake, base_fake = extract_values(lp_base, 'fake_emails')
    
        rmft_ckpts_true, rmft_true = extract_values(lp_rmft, 'true_emails')
        rmft_ckpts_fake, rmft_fake = extract_values(lp_rmft, 'fake_emails')
    
        dedup_ckpts_true, dedup_true = extract_values(lp_dedup, 'true_emails')
        dedup_ckpts_fake, dedup_fake = extract_values(lp_dedup, 'fake_emails')
    
    # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Plot 1: Baseline
        axes[0].plot(base_ckpts_true, base_true, 'o-', linewidth=2.5, markersize=6,
                 label='Original Emails', color='#3498db', alpha=0.85)
        axes[0].plot(base_ckpts_fake, base_fake, 's-', linewidth=2.5, markersize=6,
                 label='Fake Emails', color='#e67e22', alpha=0.85)
        axes[0].set_title('Baseline Fine-Tuning', fontsize=15, fontweight='bold')
        axes[0].set_xlabel('Training Checkpoints', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Avg Log-Probability\n(Higher/Less Negative = More Confident)', 
                       fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='best', fontsize=11, framealpha=0.95)
        axes[0].tick_params(labelsize=11)
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
    
        # Plot 2: RMFT
        axes[1].plot(rmft_ckpts_true, rmft_true, 'o-', linewidth=2.5, markersize=6,
                 label='Original Emails', color='#3498db', alpha=0.85)
        axes[1].plot(rmft_ckpts_fake, rmft_fake, 's-', linewidth=2.5, markersize=6,
                 label='Fake Emails', color='#e67e22', alpha=0.85)
        axes[1].set_title('RMFT', fontsize=15, fontweight='bold')
        axes[1].set_xlabel('Training Checkpoints', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='best', fontsize=11, framealpha=0.95)
        axes[1].tick_params(labelsize=11)
        axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
    
        # Plot 3: Deduplication
        axes[2].plot(dedup_ckpts_true, dedup_true, 'o-', linewidth=2.5, markersize=6,
                 label='Original Emails', color='#3498db', alpha=0.85)
        axes[2].plot(dedup_ckpts_fake, dedup_fake, 's-', linewidth=2.5, markersize=6,
                 label='Fake Emails', color='#e67e22', alpha=0.85)
        axes[2].set_title('Deduplication', fontsize=15, fontweight='bold')
        axes[2].set_xlabel('Training Checkpoints', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].legend(loc='best', fontsize=11, framealpha=0.95)
        axes[2].tick_params(labelsize=11)
        axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
    
        # Add interpretation box
        fig.text(0.5, 0.02, 
             '📊 Interpretation: Original line HIGHER than Fake line = Model memorized originals (BAD privacy)\n' +
             '                  Original and Fake lines SIMILAR = Model treats equally (GOOD privacy)',
             ha='center', fontsize=11, style='italic', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
        plt.suptitle(f'Model Confidence on Original vs Fake Emails - {test_split}', 
                 fontsize=17, fontweight='bold', y=0.98)
    
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(f'pdfs/logprobs_sidebyside_{test_split}_{model_name}.pdf', 
                format='pdf', bbox_inches='tight', dpi=300)
        plt.show()

    except Exception as e:
            print(f"Failed to draw Logprobs Graph: {e}")

    try:
        er_base = extraction_rates(base_token_word, test_split)
        er_rmft = extraction_rates(rmft_token_word, test_split)
        er_dedup = extraction_rates(dedup_token_word, test_split)

        ter_base, ser_base = fetch_lists(er_base, 'ter', 'ser')
        ter_rmft, ser_rmft = fetch_lists(er_rmft, 'ter', 'ser')
        ter_dedup, ser_dedup = fetch_lists(er_dedup, 'ter', 'ser')

    

        plt.figure(figsize=(10,6))
        plt.plot(er_base.keys(), ter_base, marker = 'o', label = 'Base Finetuning')
        plt.plot(er_rmft.keys(), ter_rmft, marker='o', label = 'Randomized Masked Finetuning')
        plt.plot(er_dedup.keys(), ter_dedup, marker='o', label='Deduplicated Finetuning')
        plt.xlabel('Training Checkpoints', fontsize=20)
        plt.ylabel(f'TER on {test_split}', fontsize=20)
        plt.title(f'Comparing {model_name} TER on {test_split}', fontsize=25)
        plt.legend(fontsize = 18, frameon = True)
        plt.tick_params(labelsize=16)
        plt.savefig(f'pdfs/ter_{test_split}_{model_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)

        plt.figure(figsize=(10,6))
        plt.plot(er_base.keys(), ser_base, marker = 'o', label = 'Base Finetuning')
        plt.plot(er_rmft.keys(), ser_rmft, marker='o', label = 'Randomized Masked Finetuning')
        plt.plot(er_dedup.keys(), ser_dedup, marker='o', label='Deduplicated Finetuning')
        plt.xlabel('Training Checkpoints', fontsize=20)
        plt.ylabel(f'SER on {test_split}', fontsize=20)
        plt.title(f'Comparing {model_name} SER on {test_split}', fontsize=25)
        plt.legend(fontsize = 18, frameon = True)
        plt.tick_params(labelsize=16)
        plt.savefig(f'pdfs/ser_{test_split}_{model_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Failed to Draw Extraction Rates Graph: {e}")

    try:
        tegr_base = tegr(base_token_word, test_split)
        tegr_rmft = tegr(rmft_token_word, test_split)
        tegr_dedup = tegr(dedup_token_word, test_split)

        plt.figure(figsize=(10,6))
        plt.plot(tegr_base.keys(), tegr_base.values(), marker='o', label = 'Baseline Finetuning')
        plt.plot(tegr_rmft.keys(), tegr_rmft.values(), marker='o', label = 'Randomized Masked Finetuning')
        plt.plot(tegr_dedup.keys(), tegr_dedup.values(), marker='o', label='Deduplicated Finetuning')
        plt.xlabel('Training Checkpoints', fontsize=20)
        plt.ylabel(f'TEGR on {test_split} (Lower the better)', fontsize=20)
        plt.title(f'Comparing {model_name} TEGR on {test_split}', fontsize=25)
        plt.legend(fontsize = 18, frameon = True)
        plt.tick_params(labelsize=16)
        plt.savefig(f'pdfs/tegr_{test_split}_{model_name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Failed to draw TEGR Graph: {e}")

    # ========== SUMMARY SECTION ==========
    print("\n" + "="*80)
    print("📋 EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    # ========== EXTRACTION DATA SUMMARY ==========
    try:
        print("\n🔒 PRIVACY METRICS (Extraction Rates):")
        print("-" * 80)
        
        er_base = extraction_rates(base_token_word, test_split)
        er_rmft = extraction_rates(rmft_token_word, test_split)
        er_dedup = extraction_rates(dedup_token_word, test_split)
        
        # Get final checkpoint values
        final_ckpt = max(er_base.keys())
        
        base_final = er_base[final_ckpt]
        rmft_final = er_rmft[final_ckpt]
        dedup_final = er_dedup[final_ckpt]

        # Calculate averages
        base_ter_avg = sum(er_base[k]['ter'] for k in er_base.keys()) / len(er_base)
        rmft_ter_avg = sum(er_rmft[k]['ter'] for k in er_rmft.keys()) / len(er_rmft)
        dedup_ter_avg = sum(er_dedup[k]['ter'] for k in er_dedup.keys()) / len(er_dedup)
        
        base_ser_avg = sum(er_base[k]['ser'] for k in er_base.keys()) / len(er_base)
        rmft_ser_avg = sum(er_rmft[k]['ser'] for k in er_rmft.keys()) / len(er_rmft)
        dedup_ser_avg = sum(er_dedup[k]['ser'] for k in er_dedup.keys()) / len(er_dedup)
        
        print(f"\n  TOTAL EXTRACTION RATE (TER):")
        print(f"    Baseline:      Final: {base_final['ter']:.3f}%  |  Avg: {base_ter_avg:.3f}%")
        print(f"    RMFT:          Final: {rmft_final['ter']:.3f}%  |  Avg: {rmft_ter_avg:.3f}%")
        print(f"    Deduplication: Final: {dedup_final['ter']:.3f}%  |  Avg: {dedup_ter_avg:.3f}%")
        
        print(f"\n  SEEN EXTRACTION RATE (SER):")
        print(f"    Baseline:      Final: {base_final['ser']:.3f}%  |  Avg: {base_ser_avg:.3f}%")
        print(f"    RMFT:          Final: {rmft_final['ser']:.3f}%  |  Avg: {rmft_ser_avg:.3f}%")
        print(f"    Deduplication: Final: {dedup_final['ser']:.3f}%  |  Avg: {dedup_ser_avg:.3f}%")
        
        # Calculate reduction percentages
        ter_reduction_rmft = ((base_ter_avg - rmft_ter_avg) / base_ter_avg) * 100
        ter_reduction_dedup = ((base_ter_avg - dedup_ter_avg) / base_ter_avg) * 100
        
        print(f"\n  PRIVACY IMPROVEMENT vs Baseline:")
        print(f"    RMFT:          {ter_reduction_rmft:.2f}% TER reduction")
        print(f"    Deduplication: {ter_reduction_dedup:.2f}% TER reduction")
        
    except Exception as e:
        print(f"\n  ❌ Failed to generate Extraction summary: {e}")

    try:
        print("\n🔒 PRIVACY METRICS (TEGR):")
        print("-" * 80)
        
        er_base = tegr(base_token_word, test_split)
        er_rmft = tegr(rmft_token_word, test_split)
        er_dedup = tegr(dedup_token_word, test_split)
        
        # Get final checkpoint values
        final_ckpt = max(er_base.keys())
        
        base_final = er_base[final_ckpt]
        rmft_final = er_rmft[final_ckpt]
        dedup_final = er_dedup[final_ckpt]
        
        # Calculate averages
        base_ter_avg = sum(er_base[k] for k in er_base.keys()) / len(er_base)
        rmft_ter_avg = sum(er_rmft[k] for k in er_rmft.keys()) / len(er_rmft)
        dedup_ter_avg = sum(er_dedup[k] for k in er_dedup.keys()) / len(er_dedup)
        
        
        print(f"\n  TRUE EMAIL GENERATION RATE (TEGR):")
        print(f"    Baseline:      Final: {base_final:.3f}%  |  Avg: {base_ter_avg:.3f}%")
        print(f"    RMFT:          Final: {rmft_final:.3f}%  |  Avg: {rmft_ter_avg:.3f}%")
        print(f"    Deduplication: Final: {dedup_final:.3f}%  |  Avg: {dedup_ter_avg:.3f}%")
        
        # Calculate reduction percentages
        ter_reduction_rmft = ((base_ter_avg - rmft_ter_avg) / base_ter_avg) * 100
        ter_reduction_dedup = ((base_ter_avg - dedup_ter_avg) / base_ter_avg) * 100
        
        print(f"\n  PRIVACY IMPROVEMENT vs Baseline:")
        print(f"    RMFT:          {ter_reduction_rmft:.2f}% TEGR reduction")
        print(f"    Deduplication: {ter_reduction_dedup:.2f}% TEGR reduction")
        
    except Exception as e:
        print(f"\n  ❌ Failed to generate Extraction summary: {e}")
    
    # ========== LOGPROBS SUMMARY ==========
    try:
        print("\n🧠 CONFIDENCE METRICS (Logprobs):")
        print("-" * 80)
        
        lp_base = logprobs(base_token_word, test_split)
        lp_rmft = logprobs(rmft_token_word, test_split)
        lp_dedup = logprobs(dedup_token_word, test_split)
        
        # Calculate average confidence gap across all checkpoints
        def calc_avg_gap(lp_dict):
            gaps = []
            for ckpt in lp_dict.keys():
                true_val = lp_dict[ckpt]['true_emails']
                fake_val = lp_dict[ckpt]['fake_emails']
                if true_val is not None and fake_val is not None:
                    gap = true_val - fake_val  # Positive = memorization
                    gaps.append(gap)
            return sum(gaps) / len(gaps) if gaps else None
        
        base_gap = calc_avg_gap(lp_base)
        rmft_gap = calc_avg_gap(lp_rmft)
        dedup_gap = calc_avg_gap(lp_dedup)
        
        print(f"\n  AVERAGE CONFIDENCE GAP (Original - Fake logprobs):")
        print(f"    [Lower Value = Better Privacy]")
        if base_gap is not None:
            print(f"    Baseline:      {base_gap:+.3f}")
        if rmft_gap is not None:
            print(f"    RMFT:          {rmft_gap:+.3f}")
        if dedup_gap is not None:
            print(f"    Deduplication: {dedup_gap:+.3f}")
        
        # Final checkpoint details
        final_ckpt = max(lp_base.keys())
        print(f"\n  FINAL CHECKPOINT ({final_ckpt}):")
        
        base_final = lp_base[final_ckpt]
        rmft_final = lp_rmft[final_ckpt]
        dedup_final = lp_dedup[final_ckpt]
        
        if base_final['true_emails'] is not None:
            print(f"    Baseline:      Original: {base_final['true_emails']:.3f}  |  Fake: {base_final['fake_emails']:.3f}")
        if rmft_final['true_emails'] is not None:
            print(f"    RMFT:          Original: {rmft_final['true_emails']:.3f}  |  Fake: {rmft_final['fake_emails']:.3f}")
        if dedup_final['true_emails'] is not None:
            print(f"    Deduplication: Original: {dedup_final['true_emails']:.3f}  |  Fake: {dedup_final['fake_emails']:.3f}")
        
    except Exception as e:
        print(f"\n  ❌ Failed to generate Logprobs summary: {str(e)}")

    # ========== SUMMARY SECTION ==========
    print("\n" + "="*80)
    print("📋 EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    # ========== PERPLEXITY SUMMARY ==========
    try:
        print("\n📈 PERFORMANCE METRICS (Perplexity):")
        print("-" * 80)
        
        base_ppl = graph_friendly_data_ppl(base_token_word, test_split)
        rmft_ppl = graph_friendly_data_ppl(rmft_token_word, test_split)
        dedup_ppl = graph_friendly_data_ppl(dedup_token_word, test_split)
        
        base_ppl = {key: val for key, val in base_ppl.items() if key != 10}
        rmft_ppl = {key: val for key, val in rmft_ppl.items() if key != 10}
        dedup_ppl = {key: val for key, val in dedup_ppl.items() if key != 10}
        
        # Calculate averages
        base_ppl_avg = sum(base_ppl.values()) / len(base_ppl)
        rmft_ppl_avg = sum(rmft_ppl.values()) / len(rmft_ppl)
        dedup_ppl_avg = sum(dedup_ppl.values()) / len(dedup_ppl)
        
        # Get final values
        final_ckpt = max(base_ppl.keys())
        base_ppl_final = base_ppl[final_ckpt]
        rmft_ppl_final = rmft_ppl[final_ckpt]
        dedup_ppl_final = dedup_ppl[final_ckpt]
        
        print(f"\n  PERPLEXITY (Lower = Better):")
        print(f"    Baseline:      Final: {base_ppl_final:.3f}  |  Avg: {base_ppl_avg:.3f}")
        print(f"    RMFT:          Final: {rmft_ppl_final:.3f}  |  Avg: {rmft_ppl_avg:.3f}")
        print(f"    Deduplication: Final: {dedup_ppl_final:.3f}  |  Avg: {dedup_ppl_avg:.3f}")
        
        # Calculate performance degradation
        ppl_increase_rmft = ((rmft_ppl_avg - base_ppl_avg) / base_ppl_avg) * 100
        ppl_increase_dedup = ((dedup_ppl_avg - base_ppl_avg) / base_ppl_avg) * 100
        
        print(f"\n  PERFORMANCE DEGRADATION vs Baseline:")
        print(f"    RMFT:          {ppl_increase_rmft:+.2f}% perplexity increase")
        print(f"    Deduplication: {ppl_increase_dedup:+.2f}% perplexity increase")
        
        # MDP if available
        try:
            mdp_rmft = mdp_graph_friendly_data(base_token_word, rmft_token_word, test_split)
            mdp_dedup = mdp_graph_friendly_data(base_token_word, dedup_token_word, test_split)
            
            mdp_rmft = {key: val for key, val in mdp_rmft.items() if key != 10}
            mdp_dedup = {key: val for key, val in mdp_dedup.items() if key != 10}
            
            mdp_rmft_avg = sum(mdp_rmft.values()) / len(mdp_rmft)
            mdp_dedup_avg = sum(mdp_dedup.values()) / len(mdp_dedup)
            
            print(f"\n  MEAN DELTA PERPLEXITY (Lower = Better):")
            print(f"    RMFT:          {mdp_rmft_avg:.3f}")
            print(f"    Deduplication: {mdp_dedup_avg:.3f}")
        except:
            pass
            
    except Exception as e:
        print(f"\n  ❌ Failed to generate Perplexity summary: {e}")
    
    print("\n" + "="*80)
    print("✅ SUMMARY COMPLETE")
    print("="*80 + "\n")
    


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Perplexities, Extraction Rates and Logprobs")

    parser.add_argument("--base_token_word", type=str, required=True, help="Token Word for Base you used while training and extraction")
    parser.add_argument("--rmft_token_word", type=str, required=True, help="Token Word for RMFT you used while training and extraction")
    parser.add_argument("--dedup_token_word", type=str, required=True, help="Token Word for Dedup you used while training and extraction")
    parser.add_argument("--test_split", type=str, required=True, help="Test Split you want to evaluate on")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name is used to name the output files")

    args = parser.parse_args()

    plot_all(
        base_token_word=args.base_token_word,
        rmft_token_word=args.rmft_token_word,
        dedup_token_word=args.dedup_token_word,
        test_split=args.test_split,
        model_name = args.model_name
    )
    

    













    