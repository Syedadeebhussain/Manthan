from typing import Tuple, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def load_text_classification_dataset(
    dataset_name: str,
    subset: str,
    text_field: str,
    label_field: str,
    tokenizer_name: str,
    max_length: int,
):
    tokenizer = load_tokenizer(tokenizer_name)
    ds = load_dataset(dataset_name, subset)

    def preprocess(examples):
        return tokenizer(
            examples[text_field],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(preprocess, batched=True)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", label_field],
    )
    ds = ds.rename_column(label_field, "labels")
    return ds, tokenizer


def load_ner_dataset(
    dataset_name: str,
    text_field: str,
    label_field: str,
    tokenizer_name: str,
    max_length: int,
    label_all_tokens: bool = False,
):
    tokenizer = load_tokenizer(tokenizer_name)
    ds = load_dataset(dataset_name)

    def tokenize_and_align(example):
        tokenized = tokenizer(
            example[text_field],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        word_ids = tokenized.word_ids()
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(example[label_field][word_id])
            else:
                label_ids.append(example[label_field][word_id] if label_all_tokens else -100)
            prev_word_id = word_id
        tokenized["labels"] = label_ids
        return tokenized

    ds = ds.map(tokenize_and_align)
    ds.set_format(type="torch")
    return ds, tokenizer


def get_dataset_for_config(cfg) -> Tuple[Any, Any]:
    ds_cfg = cfg.dataset
    if cfg.task in ["text_classification", "intent_classification"]:
        ds, tokenizer = load_text_classification_dataset(
            dataset_name=ds_cfg["name"],
            subset=ds_cfg.get("subset"),
            text_field=ds_cfg["text_field"],
            label_field=ds_cfg["label_field"],
            tokenizer_name=cfg.student_model_name,
            max_length=cfg.max_length,
        )
    elif cfg.task == "ner":
        ds, tokenizer = load_ner_dataset(
            dataset_name=ds_cfg["name"],
            text_field=ds_cfg["text_field"],
            label_field=ds_cfg["label_field"],
            tokenizer_name=cfg.student_model_name,
            max_length=cfg.max_length,
        )
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")
    return ds, tokenizer
