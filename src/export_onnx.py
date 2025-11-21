"""
Export a fine-tuned (possibly distilled/pruned/quantized) model to ONNX.
"""
import argparse
import os
import torch
import onnx

# Try to import onnx-simplifier, but make it optional
try:
    from onnxsim import simplify
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)



def export_to_onnx(model_path: str, task: str, out_path: str, max_length: int = 128):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if task in ["text_classification", "intent_classification"]:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    elif task == "ner":
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported task: {task}")

    model.eval()

    dummy = tokenizer(
        "This is a sample input",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    inputs = (dummy["input_ids"], dummy["attention_mask"])
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.onnx.export(
        model,
        inputs,
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )

        # Optional: simplify the ONNX graph if onnx-simplifier is available
    if HAS_SIMPLIFIER:
        model_onnx = onnx.load(out_path)
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, out_path)
            print(f"Simplified ONNX saved to {out_path}")
        else:
            print("ONNX simplification failed; original model saved")
    else:
        print("onnx-simplifier not installed; saved raw ONNX model to", out_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task", required=True, choices=["text_classification", "intent_classification", "ner"])
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.task, args.out_path, args.max_length)


if __name__ == "__main__":
    main()
