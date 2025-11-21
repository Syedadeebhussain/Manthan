# Android Deployment Notes

Two main options:

1. **TensorFlow Lite (recommended for broad support)**
   - Convert your ONNX or PyTorch model to TFLite (int8 or float16).
   - Bundle the `.tflite` file in `app/src/main/assets`.
   - Use the `Interpreter` API in Kotlin/Java to run inference.

2. **ONNX Runtime for Android**
   - Use ONNX Runtime Mobile with an optimized build.
   - Ship the `.onnx` model and call inference using the ORT Java API.

Key considerations:

- Tokenization must exactly match training.
- For efficient int8 TFLite, you need a representative dataset for calibration.
- Measure latency using Android's `SystemClock` or similar API, and profile battery usage with Android Studio tools.
