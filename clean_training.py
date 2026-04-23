from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import pandas as pd
import os


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, block_size=128):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def get_training_args(output_dir, lm_datasets):
    base_args = {
        "output_dir": output_dir,
        "evaluation_strategy": "steps",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "push_to_hub": False,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 50,
        "per_device_eval_batch_size": 50,
        "fp16": True,
        "report_to": "none",
        "lr_scheduler_type": "linear",
        "warmup_steps": 500,
        "seed": 42,
        "save_strategy": "steps"
    }

    checkpoint = (len(lm_datasets["train"]) // base_args["per_device_train_batch_size"]) \
                 * base_args["num_train_epochs"]
    checkpoint = max(checkpoint // 30, 1)

    base_args["save_steps"] = checkpoint

    return TrainingArguments(**base_args)


def load_csv_dataset(train_path, val_path, train_key, val_key):
    # Load CSVs
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Extract column
    train_data = train_df[train_key].dropna().astype(str).tolist()
    val_data = val_df[val_key].dropna().astype(str).tolist()

    # Convert to HF datasets
    train_dataset = Dataset.from_dict({"text": train_data})
    val_dataset = Dataset.from_dict({"text": val_data})

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

class CustomTrainer(Trainer):
    def __init__(self, *args, tokenizer = None, save_dir = "../data_seen", **kwargs):
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


def train_model(
    model_name,
    training_type,
    train_dataset_path,
    val_dataset_path,
    model_save_path,
    train_key,
    val_key
):
    # Create save directory
    os.makedirs(model_save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Load datasets
    datasets = load_csv_dataset(
        train_dataset_path,
        val_dataset_path,
        train_key,
        val_key
    )

    # Tokenization
    tokenized_datasets = datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # Grouping
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000
    )

    # Training args
    training_args = get_training_args(model_save_path, lm_datasets)

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model from CSV datasets")

    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name (e.g., gpt2)")
    parser.add_argument("--training_type", type=str, default="causal_lm", help="Type of training (currently unused)")
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--val_dataset_path", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--model_save_path", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--train_key", type=str, required=True, help="Column name for training text")
    parser.add_argument("--val_key", type=str, required=True, help="Column name for validation text")

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        training_type=args.training_type,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        model_save_path=args.model_save_path,
        train_key=args.train_key,
        val_key=args.val_key
    )