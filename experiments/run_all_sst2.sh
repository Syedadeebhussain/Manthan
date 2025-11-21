#!/usr/bin/env bash
set -e

CONFIG=configs/sst2.yaml

# 1. Distillation
python -m src.train_distill --config "$CONFIG"

# 2. Optional pruning (edit prune.py to control strategy)
# Example usage in Python:
# python - << 'EOF'
# from transformers import AutoModelForSequenceClassification
# from src.prune import example_prune_student
# model = AutoModelForSequenceClassification.from_pretrained("output/sst2_student")
# model = example_prune_student(model)
# model.save_pretrained("output/sst2_student_pruned")
# EOF

# 3. Dynamic quantization
python -m src.quantize --model_dir output/sst2_student --task text_classification --out_dir output/sst2_student_q

# 4. Export ONNX
python -m src.export_onnx --model_path output/sst2_student_q --task text_classification \
    --out_path output/sst2_student_q.onnx

# 5. Latency evaluation
python -m src.eval_latency --mode onnx --model_or_onnx output/sst2_student_q.onnx \
    --tokenizer output/sst2_student_q \
    --text "This is a benchmark sentence." \
    --runs 50
