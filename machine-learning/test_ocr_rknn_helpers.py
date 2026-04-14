import sys
import types

import numpy as np
from PIL import Image

onnxruntime = types.ModuleType("onnxruntime")
onnxruntime.InferenceSession = object
onnxruntime.SessionOptions = object
onnxruntime.ExecutionMode = types.SimpleNamespace(ORT_PARALLEL="ORT_PARALLEL")
onnxruntime.get_available_providers = lambda: []
sys.modules.setdefault("onnxruntime", onnxruntime)

rknn_pool = types.ModuleType("immich_ml.sessions.rknn.native.rknn_pool")


class DummyNativeRKNNExecutor:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def infer(self, inputs):
        return list(inputs)

    def get_io_info(self):
        return {"inputs": [], "outputs": []}


rknn_pool.NativeRKNNExecutor = DummyNativeRKNNExecutor
sys.modules.setdefault("immich_ml.sessions.rknn.native.rknn_pool", rknn_pool)

from immich_ml.schemas import ModelFormat
from immich_ml.sessions.rknn import ocr as rknn_ocr


def test_resolve_ocr_model_format_defaults_to_onnx_without_rknn(monkeypatch) -> None:
    monkeypatch.setattr(rknn_ocr, "is_rknn_available", lambda: False)

    assert rknn_ocr.resolve_ocr_model_format(None, component="OCR detection") == ModelFormat.ONNX


def test_resolve_ocr_model_format_defaults_to_rknn_when_available(monkeypatch) -> None:
    monkeypatch.setattr(rknn_ocr, "is_rknn_available", lambda: True)

    assert rknn_ocr.resolve_ocr_model_format(None, component="OCR detection") == ModelFormat.RKNN


def test_resolve_ocr_model_format_rejects_unsupported_backend() -> None:
    assert rknn_ocr.resolve_ocr_model_format(ModelFormat.ARMNN, component="OCR recognition") == ModelFormat.ONNX


def test_detection_transform_and_crop_follow_landscape_canvas() -> None:
    image = Image.new("RGB", (800, 400), color=(12, 34, 56))

    tensor, transform_info = rknn_ocr.transform_detection_input(
        image,
        max_resolution=736,
        mean=np.zeros(3, dtype=np.float32),
        std_inv=np.ones(3, dtype=np.float32),
    )

    assert tensor.shape == (1, 3, 736, 1280)
    assert transform_info == (736, 1280, 640, 1280)

    output = np.zeros((1, 1, 184, 320), dtype=np.float32)
    cropped = rknn_ocr.crop_detection_output(output, transform_info)

    assert cropped.shape == (1, 1, 160, 320)
