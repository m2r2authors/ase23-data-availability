import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import argparse
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

from torch.nn import DataParallel


class T5RerankDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        # https://github.com/huggingface/transformers/commit/e94d63f6cbf5efe288e41d9840a96a5857090617
        return {
            'input_ids': text,
            'labels': sample[2],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='t5-base', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="data dir")
    parser.add_argument("--output_model_path", default=None, type=str, required=True,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=128, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--per_device_eval_batch_size", default=256, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Number of epochs to train")
    parser.add_argument("--save_total_limit", default=5, type=int,
                        help="Number of total checkpoints")
    parser.add_argument("--max_length", default=200, type=int,
                        help="Max sequence length of templated query and document")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--fsdp', action='store_true',
                        help='Use FSDP for model parallelism')
    parser.add_argument('--top_k', default=10, type=int,
                        help='Number of top k documents to use for training')

    device = torch.device('cuda')
    torch.manual_seed(123)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    with open(os.path.join(args.data_dir, f'reranker_train_top_{args.top_k}.json'), 'r') as f:
        train_samples = json.load(f)
    with open(os.path.join(args.data_dir, f'reranker_eval_top_{args.top_k}.json'), 'r') as f:
        eval_samples = json.load(f)

    assert len(train_samples) > 0
    assert len(eval_samples) > 0
    
    print(args.epochs)

    def smart_batching_collate_text_only(batch):
        texts = [example['input_ids'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first',
                              return_tensors='pt', max_length=args.max_length)
        tokenized['labels'] = tokenizer(
            [example['labels'] for example in batch], return_tensors='pt')['input_ids']

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized

    train_dataset = T5RerankDataset(train_samples)
    eval_dataset = T5RerankDataset(eval_samples)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        save_strategy=strategy,
        evaluation_strategy='steps',
        save_steps=steps,
        eval_steps=steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=5e-5,
        num_train_epochs=args.epochs,
        warmup_steps=1000,
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        predict_with_generate=True,
        dataloader_pin_memory=False,
        save_total_limit=args.save_total_limit,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
    )

#     trainer = DataParallel(trainer)

#     trainer.module.train()

#     trainer.module.save_model(args.output_model_path)
#     trainer.module.save_state()

    trainer.train()
    trainer.save_model(args.output_model_path)
    trainer.save_state()


if __name__ == "__main__":
    main()
