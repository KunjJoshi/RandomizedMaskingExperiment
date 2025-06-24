from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from datasets import load_dataset, Dataset, DatasetDict
import torch
import pandas as pd
import numpy as np
import os
from huggingface_hub import login
import re
import random
import time
from tqdm import tqdm

EMAIL_MAKER_PATH = 'email_vals'

with open(os.path.join(EMAIL_MAKER_PATH, 'first_names.txt'), 'r') as file:
    first_names = file.readlines()

with open(os.path.join(EMAIL_MAKER_PATH, 'last_names.txt'), 'r') as file:
    last_names = file.readlines()

with open(os.path.join(EMAIL_MAKER_PATH, 'domains.txt'), 'r') as file:
    domains = file.readlines()

with open(os.path.join(EMAIL_MAKER_PATH, 'tld.txt'), 'r') as file:
    tlds = file.readlines()

first_names = [fn.strip() for fn in first_names]
last_names = [ln.strip() for ln in last_names]
domains = [dmn.strip() for dmn in domains]
tlds = [tld.strip() for tld in tlds]

hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)

MODEL_CHECKPOINT = "gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, return_tensors = 'pt')
block_size = 128

def tokenize_function(examples):
    return tokenizer(examples['text'])
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
def get_email_locations(text, email):
    escaped = re.escape(email)
    matches = [(mt.group(), mt.start(), mt.end()) for mt in re.finditer(escaped, text)]
    matches = sorted(matches, key =lambda x: x[2], reverse = True)
    return matches[:-1]
def randomize_mask_text(email):
        name, domain = email.split('@')
        try:
            first_name, last_name = name.split('.')
        except:
            first_name, last_name = name, ''
        try:
            domain, tld = domain.split('.')
        except:
            tld = 'com'
        fixer = random.randint(1,3)
        rand_fn = random.sample(first_names, 1)[0]
        rand_ln = random.sample(last_names, 1)[0]
        rand_dmn = random.sample(domains, 1)[0]
        rand_tld = random.sample(tlds, 1)[0]
        if fixer == 1:
            masker = f"{first_name}.{rand_ln}@{rand_dmn}.{rand_tld}"
        elif fixer == 2:
            masker = f"{rand_fn}.{last_name}@{rand_dmn}.{rand_tld}"
        else:
            masker = f"{rand_fn}.{rand_ln}@{domain}.{rand_tld}"
        return masker

def mask(text):
    email_occurences = {}
    replacements = []
    all_emails = [(mt.group(), mt.start(), mt.end()) for mt in re.finditer(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)]
    all_emails = sorted(all_emails, key = lambda x: x[2], reverse = True)
    for email_set in tqdm(all_emails, desc = "Masking Emails"):
        email, start, end = email_set
        if email not in email_occurences.keys():
            email_occurences[email] = 1
        else:
            email_occurences[email] += 1
            masker = randomize_mask_text(email)
            replacements.append((start, end, masker))
    
    text_list = list(text)
    for replacement in tqdm(replacements, desc = "Adjusting Replacements"):
        start, end, masker = replacement
        text_list[start:end] = list(masker)
    return ''.join(text_list)
class CustomTrainer(Trainer):
    def __init__(self, *args, tokenizer = None, save_dir = ".", **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.steps_per_epoch = len(self.get_train_dataloader())
        self.steps_per_interval = self.steps_per_epoch // 10
        self.current_epoch = 0
        self.current_interval = 0
        self.data_seen = []
        self.data_processed = []
        
        os.makedirs(self.save_dir, exist_ok = True)
    
    def on_epoch_begin(self):
        self.current_interval = 0
        
    def training_step(self, model, inputs, num_items_in_batch):
        if self.state.global_step // self.steps_per_interval > self.current_interval:
            self.save_data_interval()
            self.current_interval += 1
        self.data_seen.append(inputs)
        return super().training_step(model, inputs, num_items_in_batch)
    
    def on_epoch_end(self):
        self.save_data_interval(final = True)
        self.data_seen = []
        self.current_epoch += 1
    
    def save_data_interval(self, final = False):
        interval_label = "final" if final else f"{(self.current_interval+1) * 10}%"
        filename = f"epoch_{self.current_epoch + 1}_{interval_label}.txt"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding = 'utf-8') as file:
            for batch in self.data_seen:
                decoded_texts = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens = True)
                for text in decoded_texts:
                    file.write(text + '\n')
        self.data_seen = []

def get_training_args(lm_datasets):
    """
    Returns training arguments  
    """
    
    base_args = {
        "output_dir": "models",
        "evaluation_strategy": "no",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "push_to_hub": False,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8, #adjust batch size according to your compute 
        "per_device_eval_batch_size": 8,
        "fp16": True,
        "report_to": "none",
        "lr_scheduler_type": "linear",
        "warmup_steps": 500,
        "seed": 42
    }
    
    base_args['save_strategy'] = "steps"
    checkpoint = (len(lm_datasets["train"]) // base_args['per_device_train_batch_size']) * base_args['num_train_epochs']
    checkpoint = checkpoint // 30
    base_args["save_steps"] = checkpoint
    print(base_args)
    return TrainingArguments(**base_args)

def get_trainer(model, training_args, lm_datasets, tokenizer = None):
    return CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = lm_datasets['train'],
    eval_dataset = lm_datasets['validation'],
    tokenizer = tokenizer,
    save_dir = "data")


def parse_custom_text_files(textFilePath, separator = "\n\n", randomize = False):
    with open(textFilePath, 'r') as file:
        data = file.read()
    
    if randomize:
        start_time = time.time()
        data = mask(data)
        end_time = time.time()
        print(f"Masked the whole data in {end_time - start_time} seconds")
        num_times = len(re.findall('kay.mann@enron.com', data))
        print(f"Target Email appearing {num_times} number of times")
    content = data.split(separator)
    content = [con.strip() for con in content if con.strip()]
    return content

def train(train_file_path, val_file_path, randomizer = False): 
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT, device_map="auto", low_cpu_mem_usage=True)

    training_content = parse_custom_text_files(train_file_path, "\n**--ByKunj--**\n", randomize = randomizer)
    validation_content = parse_custom_text_files(val_file_path, "\n**--ByKunj--**\n", randomize = randomizer)

    training_dataset = Dataset.from_dict({
                "text": training_content
            })
    
    val_dataset = Dataset.from_dict({
                "text": validation_content
            })

    datasets = DatasetDict(
        {
            "train" : training_dataset,

            "validation" : val_dataset, 
        }
    )
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4,)
    training_args = get_training_args(lm_datasets)
    
    trainer = get_trainer(model, training_args, lm_datasets, tokenizer) 
    trainer.train() 

if __name__ == "__main__":
    train("datasets/enron/training.txt", "datasets/enron/validation.txt", randomizer = True)