"""
Distillation training script using Hugging Face Trainer.

Supports:
- text_classification / intent_classification
- NER (token classification)
"""
import argparse
from typing import Dict, Any

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_metric

from .config import load_config
from .data import get_dataset_for_config
from .modeling import get_teacher_student_models


class DistillationTrainer(Trainer):
    def __init__(
        self,
        teacher,
        temperature: float,
        alpha_ce: float,
        alpha_nll: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_nll = alpha_nll
        self.ce_loss_fct = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Student forward
        outputs_student = model(**inputs)
        logits_student = outputs_student.logits

        # Teacher forward
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            logits_teacher = teacher_outputs.logits

        # CE (hard labels)
        loss_nll = self.ce_loss_fct(logits_student.view(-1, logits_student.size(-1)), labels.view(-1))

        # KL / soft labels
        t = self.temperature
        log_p = torch.nn.functional.log_softmax(logits_student / t, dim=-1)
        q = torch.nn.functional.softmax(logits_teacher / t, dim=-1)
        loss_ce = torch.nn.functional.kl_div(log_p, q, reduction="batchmean") * (t ** 2)

        loss = self.alpha_ce * loss_ce + self.alpha_nll * loss_nll
        return (loss, outputs_student) if return_outputs else loss


def build_training_args(cfg) -> TrainingArguments:
    tr = cfg.training
    return TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=tr.get("epochs", 3),
        learning_rate=tr.get("learning_rate", 3e-5),
        weight_decay=tr.get("weight_decay", 0.0),
        warmup_ratio=tr.get("warmup_ratio", 0.0),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy" if cfg.task != "ner" else "f1",
    )


def build_compute_metrics(cfg):
    if cfg.task == "ner":
        metric = load_metric("seqeval")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = logits.argmax(-1)
            # Remove ignored indices
            true_predictions = []
            true_labels = []
            for pred, lab in zip(predictions, labels):
                tp = []
                tl = []
                for p, l in zip(pred, lab):
                    if l != -100:
                        tp.append(int(p))
                        tl.append(int(l))
                true_predictions.append(tp)
                true_labels.append(tl)
            return metric.compute(predictions=true_predictions, references=true_labels)

    else:
        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = logits.argmax(-1)
            return metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds, tokenizer = get_dataset_for_config(cfg)
    teacher, student = get_teacher_student_models(cfg)
    teacher.eval()
    teacher.to("cuda" if torch.cuda.is_available() else "cpu")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_args(cfg)
    dist_cfg: Dict[str, Any] = cfg.distillation
    trainer = DistillationTrainer(
        teacher=teacher,
        temperature=dist_cfg.get("temperature", 2.0),
        alpha_ce=dist_cfg.get("alpha_ce", 0.5),
        alpha_nll=dist_cfg.get("alpha_nll", 0.5),
        model=student,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", ds.get("test")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
