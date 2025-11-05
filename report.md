Edge AI Prototype Report

Overview

This repository contains a minimal Edge-AI prototype pipeline showing how to train a lightweight image classification model, convert it to TensorFlow Lite, and evaluate the TFLite models. The code uses CIFAR-10 as a proxy dataset; replace it with a recyclable-items dataset (images organized by class) for a production experiment.

Files

- `src/train.py` — train a MobileNetV2-based lightweight classifier on CIFAR-10 and save SavedModel + .h5.
- `src/convert_tflite.py` — convert SavedModel to TFLite: float32 baseline, dynamic-range quantized, and (attempt) full-int8 quantization using a representative dataset.
- `src/evaluate_tflite.py` — run inference with the TFLite interpreter on the test set and compute accuracy.
- `requirements.txt` — packages to install in Colab or a Python environment.

How to run (Colab recommended)

1) In Colab, create a Python 3 runtime. Install packages:

```bash
pip install -r requirements.txt
```

2) Train the model (example):

```bash
python src/train.py --epochs 8 --batch_size 64 --model_dir outputs/model
```

Expected output: `outputs/model/saved_model` (SavedModel) and `outputs/model/final_model.h5`.

3) Convert to TFLite:

```bash
python src/convert_tflite.py --saved_model_dir outputs/model/saved_model --out_dir outputs/tflite
```

This produces `model_float32.tflite`, `model_dynamic_quant.tflite`, and (if representative dataset allowed) `model_int8_fullquant.tflite` in `outputs/tflite`.

4) Evaluate a TFLite model:

```bash
python src/evaluate_tflite.py --tflite_model outputs/tflite/model_dynamic_quant.tflite
```

Accuracy metrics

- The code prints test accuracy after training (Keras evaluation) and when evaluating TFLite files.
- Because I could not run training in this environment, I did not produce measured numbers here. Typical results on CIFAR-10 with a small MobileNetV2 trained for ~8-12 epochs (no pretraining) often achieve ~50-75% depending on hyperparameters; replacing with your recyclable-items dataset and tuning will produce task-specific metrics.

Deployment steps for Raspberry Pi / Edge device

1) Pick TFLite model to deploy: prefer `model_dynamic_quant.tflite` or `model_int8_fullquant.tflite` for smaller size and faster inference.
2) Copy the `.tflite` file to the device.
3) On Raspberry Pi install `tensorflow` or `tflite-runtime` (lighter):

```powershell
# on Raspberry Pi (Linux shell example):
python3 -m pip install tflite-runtime
```

4) Use the Python TFLite Interpreter to run inference on camera frames. Example skeleton (on-device):

- Initialize interpreter with `tf.lite.Interpreter(model_path='model.tflite')` and allocate tensors.
- Preprocess camera frames to match training preprocessing (resize to 96x96, scale to [0,1] or uint8 depending on model), then call `interpreter.set_tensor(...)` and `interpreter.invoke()`.
- Post-process outputs (argmax, softmax thresholds, etc.) and integrate into your application logic.

Edge AI benefits (short)

- Low latency: inference on-device avoids roundtrips to cloud.
- Privacy: raw images never leave the device.
- Offline operation: works without network connectivity.
- Bandwidth & cost savings: only minimal metadata or compressed results are transmitted.

Next steps / suggestions

- Replace CIFAR-10 with a curated recyclable-items dataset (photo-realistic images, multiple orientations, backgrounds). Use `tf.keras.utils.image_dataset_from_directory` to load labeled folders.
- Add data augmentation and class-balance techniques.
- If Pi performance is insufficient, try a smaller backbone (e.g., MobileNetV2 with reduced alpha, or a custom small CNN) or use Coral Edge TPU with a compiled TFLite model.
- Run experiments logging with TensorBoard and sweep learning rate / augmentations.

Notes

- The provided scripts are intentionally simple and intended for demonstration and Colab experimentation. You can run them locally if you have TensorFlow installed, but training on a CPU may be slow; use GPU or Colab.
