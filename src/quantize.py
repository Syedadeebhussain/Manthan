"""
Quantization helpers:
- Dynamic quantization (easy, good for CPU & ONNX export).
"""
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification


def dynamic_quantize(model_dir: str, task: str, out_dir: str):
    if task in ["text_classification", "intent_classification"]:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    elif task == "ner":
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    else:
        raise ValueError(f"Unsupported task: {task}")

    qmodel = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    qmodel.save_pretrained(out_dir)
    print(f"Dynamic quantized model saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--task", required=True, choices=["text_classification", "intent_classification", "ner"])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    dynamic_quantize(args.model_dir, args.task, args.out_dir)


if __name__ == "__main__":
    main()
