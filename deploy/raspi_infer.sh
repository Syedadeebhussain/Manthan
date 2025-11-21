#!/usr/bin/env bash
# Example script to run ONNX inference on Raspberry Pi.

# Assumes:
#   - Python and pip installed
#   - onnxruntime for ARM installed
#   - model and tokenizer are present in ../output

python3 -m src.inference_onnx \
  --onnx output/sst2_student.onnx \
  --tokenizer output/sst2_student \
  --text "I really enjoyed this!"
