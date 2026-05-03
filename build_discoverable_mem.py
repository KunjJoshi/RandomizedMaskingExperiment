import pandas as pd
import json
import random

def get_prompts(text_list, positions):
    prompt_set = {}
    keys = list(positions.keys())
    print("Number of Emails: ", len(keys))
    all_prompts = []
    for key in keys:
        prompt_set[key] = []
        posn = positions[key]
        for pos in posn:
            idx, start, end = pos
            text = text_list[idx]
            prompt = text[0:start]
            prompt_set[key].append(prompt)
            all_prompts.append(prompt)
    print(f"Number of email occurrences: {len(all_prompts)}")
    return prompt_set

def sample(prompt_set, sample_size=2):
    keys = prompt_set.keys()
    prem_mapping = {}
    all_prompts = []
    for key in keys:
        prompts = prompt_set[key]
        sampled_prompts = random.sample(prompts, sample_size)
        for pr in sampled_prompts:
            prem_mapping[pr] = key
        all_prompts.extend(sampled_prompts)
    return all_prompts, prem_mapping

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample Discoverable Memorization Dataset"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to input CSV dataset"
    )

    parser.add_argument(
        "--data_key",
        type=str,
        required=True,
        help="Column name containing text data"
    )

    parser.add_argument(
        "--positions_path",
        type=str,
        required=True,
        help="Optional path to precomputed PII store JSON (if not provided, will be generated)"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Optional output CSV path (default: randomize_masked_<dataset_path>)"
    )

    parser.add_argument(
        "--sample_size",
        type=bool,
        default=2,
        help=""
    )
    args = parser.parse_args()

    text_list = pd.read_csv(args.dataset_path)[args.data_key].to_list()
    with open(args.positions_path, 'r') as file:
        positions = json.load(file)
    prompt_set = get_prompts(text_list, positions)
    all_prompts, mapping = sample(prompt_set, args.sample_size)
    with open(f'jsons/prompt_mapping.json', 'w') as file:
        json.dump(mapping, file, indent=4)
    df = pd.DataFrame({"prompt":all_prompts})
    df.to_csv(args.output_path, index=False)
