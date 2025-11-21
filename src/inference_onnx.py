"""
Simple ONNX Runtime inference script for text inputs.
"""
import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def run_inference(onnx_path: str, tokenizer_path: str, text: str):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="np",
    )
    ort_inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }

    logits = sess.run(None, ort_inputs)[0]
    pred = int(np.argmax(logits, axis=-1)[0])
    print("Text:", text)
    print("Predicted label index:", pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    run_inference(args.onnx, args.tokenizer, args.text)


if __name__ == "__main__":
    main()
