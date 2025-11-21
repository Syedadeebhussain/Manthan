from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)


def get_teacher_student_models(cfg):
    if cfg.task in ["text_classification", "intent_classification"]:
        teacher = AutoModelForSequenceClassification.from_pretrained(
            cfg.teacher_model_name,
            num_labels=cfg.num_labels,
        )
        student = AutoModelForSequenceClassification.from_pretrained(
            cfg.student_model_name,
            num_labels=cfg.num_labels,
        )
    elif cfg.task == "ner":
        teacher = AutoModelForTokenClassification.from_pretrained(
            cfg.teacher_model_name,
            num_labels=cfg.num_labels,
        )
        student = AutoModelForTokenClassification.from_pretrained(
            cfg.student_model_name,
            num_labels=cfg.num_labels,
        )
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")
    return teacher, student
