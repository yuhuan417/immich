# Model pool over the native RKNN executor (sessions/rknn/native/rknn_pool.cpp):
# one model load in shared memory, worker contexts cloned from it via rknn_dup_context,
# so RAM no longer multiplies with the thread count.

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from numpy.typing import NDArray

from immich_ml.config import log
from immich_ml.models.constants import RKNN_SUPPORTED_SOCS
from immich_ml.schemas import ModelTensor

try:
    from .native import rknn_pool as _native
except ImportError:
    _native = None


def get_soc(device_tree_path: Path | str) -> str | None:
    try:
        with Path(device_tree_path).open() as f:
            device_compatible_str = f.read()
            for soc in RKNN_SUPPORTED_SOCS:
                if soc in device_compatible_str:
                    return soc
            log.warning("Device is not supported for RKNN")
    except OSError as e:
        log.warning(f"Could not read {device_tree_path}. Reason: %s", e)
    return None


soc_name = get_soc("/proc/device-tree/compatible") if _native is not None else None
is_available = _native is not None and soc_name is not None


class RknnNode(NamedTuple):
    name: str
    shape: tuple[int, ...]


class RknnPoolExecutor:
    def __init__(self, model_path: str, tpes: int) -> None:
        if _native is None:
            raise RuntimeError("rknn is not available!")
        self.tpes = tpes
        # want_float=False keeps the outputs in the precision the NPU computed them in
        self.native = _native.NativeRKNNExecutor(model_path, tpes, False)
        info: dict[str, Any] = self.native.get_io_info()
        self.inputs = [RknnNode(node["name"], tuple(node["dims"])) for node in info["inputs"]]
        self.outputs = [RknnNode(node["name"], tuple(node["dims"])) for node in info["outputs"]]
        self.custom_string = self.native.get_custom_string()

    def run(self, inputs: list[ModelTensor], data_format: str | None = None) -> list[NDArray[Any]]:
        # data_format is accepted for the old pool interface; the extension transposes
        # and casts every input to what the binary declares, so no layout hint is needed
        return self.native.infer(inputs)

    def release(self) -> None:
        self.native = None  # dropping the handle destroys the cloned contexts

    def __del__(self) -> None:
        self.release()
