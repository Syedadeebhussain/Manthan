"""
Latency evaluation for PyTorch and ONNX models.

Power measurement is device-specific; here we only measure latency.
"""
import time
import argparse
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def measure_pytorch_latency(model_dir: str, text: str, runs: int = 50):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    # warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(**enc)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**enc)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    print(f"PyTorch latency over {runs} runs: mean={np.mean(times):.2f}ms, std={np.std(times):.2f}ms")


def measure_onnx_latency(onnx_path: str, tokenizer_dir: str, text: str, runs: int = 50):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    enc = tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)

    ort_inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    # warmup
    for _ in range(5):
        _ = sess.run(None, ort_inputs)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = sess.run(None, ort_inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    print(f"ONNX latency over {runs} runs: mean={np.mean(times):.2f}ms, std={np.std(times):.2f}ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pytorch", "onnx"], required=True)
    parser.add_argument("--model_or_onnx", required=True)
    parser.add_argument("--tokenizer", required=False)
    parser.add_argument("--text", default="This is a latency test.")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "pytorch":
        measure_pytorch_latency(args.model_or_onnx, args.text, args.runs)
    else:
        if not args.tokenizer:
            raise ValueError("Tokenizer path required in ONNX mode")
        measure_onnx_latency(args.model_or_onnx, args.tokenizer, args.text, args.runs)


if __name__ == "__main__":
    main()
