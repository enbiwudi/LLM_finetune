import os
import sys
import torch
import pickle
import random
import json
import argparse
import logging
import deepspeed
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence
import datasets
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, AutoTokenizer, AutoModel
from deal_data import jload, SupervisedDataset, DataCollatorForSupervisedDataset
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
    TaskType
)

os.environ['WANDB_MODE'] = 'dryrun'

# Parameters
IGNORE_INDEX = -100
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
# Model cach dir
MODEL_PATH = {
    "Llama-7b": YOUR_MODEL_PATH
}
# Lora target modules for Llama
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "down_proj",
    "gate_proj",
    "up_proj",
]

DEVICE_MAP = "auto"
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DDP = WORLD_SIZE != 1
if DDP:
    DEVICE_MAP = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    
def load_model(base_path="Llama-7b", use_deepspeed=True, use_lora=True, lora_path=None):
    logging.warning("Loading base model...")
    total_params, params = 0, 0
    base_path = MODEL_PATH.get(base_path, base_path)
    # 不用Lora
    if not use_lora:
        model = LlamaForCausalLM.from_pretrained(base_path)
        logging.warning("Not use Lora, maybe you should check your bach size in case OOM.")
    else:
        if use_deepspeed:

            model = LlamaForCausalLM.from_pretrained(base_path, device_map=DEVICE_MAP)
        else:
            model = LlamaForCausalLM.from_pretrained(base_path, load_in_8bit=True, device_map=DEVICE_MAP)
            model = prepare_model_for_int8_training(model)
        if lora_path is None:
            logging.warning("Use Lora model without pretrain.")
            config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
        else:
            logging.warning("Use pretrained Lora model.")
            model = PeftModel.from_pretrained(model, lora_path)

    for n, p in model.model.named_parameters():
        if any([x in n for x in ["lora"]]):
            total_params += p.numel()
            p.requires_grad = True
        else:
            p.requires_grad = False
        params += p.numel()
    logging.warning(
        "Total number of parameters: {}M, rate: {}%".format(
            total_params // 1000 / 1000, round(total_params / params * 100, 2)
        )
    )
    return model

def load_tokenizer(base_path="Llama-7b", max_length=256):
    base_path = MODEL_PATH.get(base_path, base_path)
    tokenizer = LlamaTokenizer.from_pretrained(base_path)
    tokenizer.model_max_length = max_length
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "</s>",
        "unk_token": "</s>",
        "pad_token": "[PAD]"})
    return tokenizer
    
def load_data(dataset_path=None, load_data_from_dataset=True, load_data_from_json=False, tokenizer=None):
    if dataset_path is None or tokenizer is None:
        raise Exception("Please make sure you pase dataset path and tokenizer!")
    if load_data_from_dataset and load_data_from_json:
        raise Exception("Do not choose load_data_from_dataset and load_data_from_json at same time!")
    if load_data_from_dataset:
        print("loading dataset ...")
        train_dataset = datasets.load_from_disk(dataset_path)
    elif load_data_from_json:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=dataset_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return train_dataset, data_collator

def load_deepspeed_config(args):
    deepspeed_config = jload(args.deepspeed_config)
    deepspeed_config['train_micro_batch_size_per_gpu'] = args.micro_batch_size
    deepspeed_config['gradient_accumulation_steps'] = args.batch_size // args.micro_batch_size
    deepspeed_config['optimizer']['params']['lr'] = args.lr
    return deepspeed_config

def train(args):
    logging.warning("Loading tokenizer...")
    tokenizer = load_tokenizer(base_path=args.base_path, max_length=args.max_length)
    logging.warning("Loading data...")
    train_dataset, data_collator = load_data(
        dataset_path=args.dataset_path, load_data_from_dataset=args.load_data_from_dataset, 
        load_data_from_json=args.load_data_from_json, tokenizer=tokenizer)
    logging.warning("Loading model...")
    model = load_model(base_path=args.base_path, use_deepspeed=args.use_deepspeed, use_lora=args.use_lora, lora_path=args.lora_path)
    deepspeed_config = load_deepspeed_config(args) if args.use_deepspeed else None
    # Training
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_first_step=True,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            output_dir=args.save_model_path,
            save_total_limit=args.save_total_limit,
            ddp_find_unused_parameters=False if DDP else None,
            deepspeed=deepspeed_config
        ),
        data_collator=data_collator,
    )
    model.config.use_cache = False
    if args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))
    trainer.train()
    model.save_pretrained(args.save_model_path)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro_batch_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_path', type=str, default='Llama-7b')
    parser.add_argument('--lora_path', type=str, default='ft_models/models')
    parser.add_argument('--save_model_path', type=str, default="./ft_models/models_llama-7b/")
    parser.add_argument('--use_lora', type=bool, default=True)
    parser.add_argument('--use_deepspeed', type=bool, default=True)
    parser.add_argument('--deepspeed_config', type=str, default='./config/ds_config.json')
    parser.add_argument('--load_data_from_dataset', type=bool, default=False)
    parser.add_argument('--load_data_from_json', type=bool, default=True)
    parser.add_argument('--dataset_path', type=str, default='dataset/data.json')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--save_strategy', type=str, default="epoch")
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=2, help="how many steps to log")
    parser.add_argument('--warmup_ratio', type=int, default=0.01)
    parser.add_argument('--lr', type=float, default=9e-6)
    parser.add_argument('--local_rank', default=-1, type=int, required=True)
    args = parser.parse_args()
    train(args)
