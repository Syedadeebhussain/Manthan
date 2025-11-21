#!/usr/bin/env bash
# Example Jetson Nano / Xavier script.
# Pre-step: convert ONNX to TensorRT engine if desired using `trtexec`.

# If you want to just use ONNX Runtime with TensorRT EP:
# export ORT_TENSORRT_OPTIONS... (depends on install) and run:

python3 -m src.inference_onnx \
  --onnx output/sst2_student.onnx \
  --tokenizer output/sst2_student \
  --text "The movie was terrible."
