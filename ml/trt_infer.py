"""
trt_infer.py

Shared TensorRT inference helper for all classifier models.
Loads .trt engines and runs inference via pycuda.

Usage:
    from trt_infer import TRTClassifier
    clf = TRTClassifier("models/glass_check.trt", ["guinness_tulip","not_glass",...])
    label, confidence = clf.predict(crop_rgb)
"""

import numpy as np
import sys
from pathlib import Path

def _log(*args):
    print(*args, file=sys.stderr, flush=True)


class TRTClassifier:
    """
    Loads a TensorRT engine and runs softmax classification.
    Input: RGB crop → resized to 224x224, normalised ImageNet stats.
    Output: (class_label, confidence)
    """

    IMAGE_SIZE = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, engine_path, classes):
        self.classes      = classes
        self.engine_path  = str(engine_path)
        self._engine      = None
        self._context     = None
        self._inputs      = None
        self._outputs     = None
        self._bindings    = None
        self._stream      = None

    def _load(self):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        _log(f"[trt] Loading {self.engine_path}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(self.engine_path, "rb") as f:
            runtime       = trt.Runtime(TRT_LOGGER)
            self._engine  = runtime.deserialize_cuda_engine(f.read())

        self._context  = self._engine.create_execution_context()
        self._stream   = cuda.Stream()

        # Allocate buffers
        self._inputs   = []
        self._outputs  = []
        self._bindings = []

        for binding in self._engine:
            size  = trt.volume(self._engine.get_binding_shape(binding))
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))

            if self._engine.binding_is_input(binding):
                self._inputs.append({"host": host_mem, "device": device_mem})
            else:
                self._outputs.append({"host": host_mem, "device": device_mem})

        _log(f"[trt] Loaded {Path(self.engine_path).name} — {len(self.classes)} classes")

    def _preprocess(self, crop_rgb):
        import cv2
        img = cv2.resize(crop_rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        img = img.transpose(2, 0, 1)        # HWC → CHW
        return np.ascontiguousarray(img[np.newaxis], dtype=np.float32)

    def predict(self, crop_rgb):
        """
        Args:
            crop_rgb: numpy HxWx3 RGB array

        Returns:
            (label: str, confidence: float)
        """
        import pycuda.driver as cuda

        if self._engine is None:
            self._load()

        tensor = self._preprocess(crop_rgb).ravel()
        np.copyto(self._inputs[0]["host"], tensor)

        # H2D
        cuda.memcpy_htod_async(
            self._inputs[0]["device"],
            self._inputs[0]["host"],
            self._stream
        )

        # Inference
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle
        )

        # D2H
        cuda.memcpy_dtoh_async(
            self._outputs[0]["host"],
            self._outputs[0]["device"],
            self._stream
        )
        self._stream.synchronize()

        logits = self._outputs[0]["host"].copy()

        # Softmax
        exp    = np.exp(logits - logits.max())
        probs  = exp / exp.sum()

        best_idx    = int(np.argmax(probs))
        confidence  = float(probs[best_idx])
        label       = self.classes[best_idx] if best_idx < len(self.classes) else str(best_idx)

        return label, confidence


# ── ONNX Runtime fallback ─────────────────────────────────────

class ONNXClassifier:
    """
    Fallback classifier using ONNX Runtime when TRT engine is not available.
    Same interface as TRTClassifier.
    """

    IMAGE_SIZE = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, onnx_path, classes):
        self.classes   = classes
        self.onnx_path = str(onnx_path)
        self._session  = None

    def _load(self):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._session = ort.InferenceSession(self.onnx_path, providers=providers)
        _log(f"[onnx] Loaded {Path(self.onnx_path).name} on {self._session.get_providers()[0]}")

    def _preprocess(self, crop_rgb):
        import cv2
        img = cv2.resize(crop_rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        img = img.transpose(2, 0, 1)[np.newaxis]
        return np.ascontiguousarray(img, dtype=np.float32)

    def predict(self, crop_rgb):
        if self._session is None:
            self._load()

        tensor  = self._preprocess(crop_rgb)
        logits  = self._session.run(None, {"image": tensor})[0][0]
        exp     = np.exp(logits - logits.max())
        probs   = exp / exp.sum()

        best_idx   = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        label      = self.classes[best_idx] if best_idx < len(self.classes) else str(best_idx)

        return label, confidence


def load_classifier(trt_path, onnx_path, classes):
    """
    Load TRT engine if available, fall back to ONNX Runtime.
    """
    if Path(trt_path).exists():
        return TRTClassifier(trt_path, classes)
    elif Path(onnx_path).exists():
        _log(f"[trt] {Path(trt_path).name} not found — using ONNX fallback")
        return ONNXClassifier(onnx_path, classes)
    else:
        raise FileNotFoundError(
            f"Neither {trt_path} nor {onnx_path} found"
        )