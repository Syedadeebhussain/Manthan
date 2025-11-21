# Low-Latency On-Device NLP Using Efficient Transformers

This repository demonstrates a complete pipeline for **edge-ready NLP models** using efficient Transformers:

- Tasks: intent classification, sentiment analysis, NER, summarization (classification + token classification examples coded).
- Techniques: knowledge distillation, structured pruning, quantization, ONNX export.
- Targets: Raspberry Pi, Jetson Nano, Android (via TFLite or ONNX Runtime).

## Goals

- Minimize **inference latency** and **energy** on constrained devices.
- Maintain as much **accuracy** as possible relative to a large teacher model.
- Provide an **end-to-end experimental pipeline**:

1. Train teacher / load pre-trained teacher.
2. Distill to smaller student.
3. Prune attention heads and feed-forward layers.
4. Quantize (dynamic or QAT).
5. Export to ONNX.
6. Evaluate latency & accuracy on edge targets.

## Quickstart

```bash
# 1. Create env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. Run SST-2 distillation
python -m src.train_distill --config configs/sst2.yaml

# 4. Export student to ONNX
python -m src.export_onnx --model_path output/sst2_student --out_path output/sst2_student.onnx

# 5. Run ONNX inference
python -m src.inference_onnx --onnx output/sst2_student.onnx --tokenizer output/sst2_student \
    --text "I loved this movie!"
