import pandas as pd
from datasets import load_dataset
import random

def get_dataset_from_hf(dataset_name, split, key):
    ds = load_dataset(dataset_name)
    split = ds[split].to_pandas()
    prompts = split[key].to_list()
    return prompts

def save_df(dataset_name, split, key, save_path, n = 2500):
    prompts = get_dataset_from_hf(dataset_name, split, key)
    prompts = random.sample(prompts, n)
    df = pd.DataFrame({key: prompts})
    df.to_csv(save_path, index=False)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Data using a Testing Dataset")

    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset ID from HuggingFace")
    parser.add_argument("--split", type=str, default="gpt_base", help="Dataset Split to be used")
    parser.add_argument("--key", type=str, required=True, help="Path to testing CSV file")
    parser.add_argument("--save_path", type=str, required=True, help="Where you nwant to save the dataset")
    args = parser.parse_args()

    save_df(args.dataset_name, args.split, args.key, args.save_path)