from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
from rapidocr.utils.typings import EngineType, LangDet, LangRec, OCRVersion, TaskType
from rapidocr.utils.typings import ModelType as RapidModelType

from immich_ml.config import log
from immich_ml.schemas import ModelFormat, ModelSession

_SQUARE_CANVAS = (736, 736)
_LANDSCAPE_CANVAS = (736, 1280)
_PORTRAIT_CANVAS = (1280, 736)


def is_rknn_available() -> bool:
    from . import immich_session

    return immich_session.is_available


def resolve_ocr_model_format(requested_model_format: ModelFormat | None, *, component: str) -> ModelFormat:
    if requested_model_format == ModelFormat.RKNN:
        return ModelFormat.RKNN
    if requested_model_format in (None, ModelFormat.ONNX):
        if requested_model_format is None and is_rknn_available():
            return ModelFormat.RKNN
        return ModelFormat.ONNX

    log.warning("%s is not supported for %s; using ONNX instead.", str(requested_model_format).upper(), component)
    return ModelFormat.ONNX


def download_detection_model(
    *,
    model_format: ModelFormat,
    model_name: str,
    save_path: Path,
    download_rknn: Callable[[], None],
) -> ModelFormat:
    return _download_ocr_model(
        model_format=model_format,
        model_name=model_name,
        save_path=save_path,
        component="OCR detection",
        task_type=TaskType.DET,
        language=LangDet.CH,
        download_rknn=download_rknn,
    )


def download_recognition_model(
    *,
    model_format: ModelFormat,
    model_name: str,
    language: LangRec,
    save_path: Path,
    download_rknn: Callable[[], None],
) -> ModelFormat:
    return _download_ocr_model(
        model_format=model_format,
        model_name=model_name,
        save_path=save_path,
        component="OCR recognition",
        task_type=TaskType.REC,
        language=language,
        download_rknn=download_rknn,
    )


def load_rknn_session(
    make_session: Callable[[Path], ModelSession],
    model_path: Path,
    *,
    input_name: str,
) -> tuple[ModelSession, str]:
    session = make_session(model_path)
    inputs = session.get_inputs()
    if inputs:
        input_name = inputs[0].name or input_name
    return session, input_name


def transform_detection_input(
    img: Image.Image,
    *,
    max_resolution: int,
    mean: NDArray[np.float32],
    std_inv: NDArray[np.float32],
) -> tuple[NDArray[np.float32], tuple[int, int, int, int]]:
    canvas_h, canvas_w = _get_canvas_shape(img)
    ratio = min(
        float(max_resolution) / min(img.height, img.width),
        float(canvas_h) / img.height,
        float(canvas_w) / img.width,
    )

    resize_h = _round_to_stride(img.height * ratio, canvas_h)
    resize_w = _round_to_stride(img.width * ratio, canvas_w)
    resized_img = img.resize((int(resize_w), int(resize_h)), resample=Image.Resampling.LANCZOS)

    resized_np: NDArray[np.float32] = cv2.cvtColor(
        np.array(resized_img, dtype=np.float32),
        cv2.COLOR_RGB2BGR,
    )  # type: ignore
    img_np = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    img_np[:resize_h, :resize_w, :] = resized_np
    img_np -= mean
    img_np *= std_inv
    img_np = np.transpose(img_np, (2, 0, 1))
    return np.expand_dims(img_np, axis=0), (canvas_h, canvas_w, resize_h, resize_w)


def crop_detection_output(output: NDArray[np.float32], transform_info: tuple[int, int, int, int]) -> NDArray[np.float32]:
    canvas_h, canvas_w, resize_h, resize_w = transform_info
    output_h, output_w = output.shape[-2:]
    effective_h = max(1, min(output_h, int(round(output_h * resize_h / canvas_h))))
    effective_w = max(1, min(output_w, int(round(output_w * resize_w / canvas_w))))
    return output[..., :effective_h, :effective_w]


def _get_canvas_shape(img: Image.Image) -> tuple[int, int]:
    ratio = img.width / float(img.height)
    if 0.8 <= ratio <= 1.25:
        return _SQUARE_CANVAS
    if ratio > 1:
        return _LANDSCAPE_CANVAS
    return _PORTRAIT_CANVAS


def _round_to_stride(value: float, limit: int, stride: int = 32) -> int:
    rounded = max(stride, int(round(value / stride) * stride))
    return min(limit, rounded)


def _download_ocr_model(
    *,
    model_format: ModelFormat,
    model_name: str,
    save_path: Path,
    component: str,
    task_type: TaskType,
    language: LangDet | LangRec,
    download_rknn: Callable[[], None],
) -> ModelFormat:
    if model_format == ModelFormat.RKNN:
        try:
            download_rknn()
            return ModelFormat.RKNN
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Failed to download %s model '%s' for %s; falling back to ONNX if available.",
                component,
                model_name,
                model_format.upper(),
                exc_info=exc,
            )

    model_info = InferSession.get_model_url(
        FileInfo(
            engine_type=EngineType.ONNXRUNTIME,
            ocr_version=OCRVersion.PPOCRV5,
            task_type=task_type,
            lang_type=language,
            model_type=_get_rapid_model_type(model_name),
        )
    )
    DownloadFile.run(
        DownloadFileInput(
            file_url=model_info["model_dir"],
            sha256=model_info["SHA256"],
            save_path=save_path,
            logger=log,
        )
    )
    return ModelFormat.ONNX


class RknnTextRecognitionRunner:
    def __init__(self, *, model_dir: Path, model_name: str, language: LangRec, batch_size: int) -> None:
        self.model_dir = model_dir
        self.model_name = model_name
        self.language = language
        self.batch_size = max(1, batch_size)
        self.decoder = CTCLabelDecode(character_path=self._get_dict_path())
        self._bucket_widths = (320, 640, 960)
        self._image_height = 48
        self._base_width = 320

    def predict(
        self,
        *,
        session: ModelSession,
        input_name: str,
        images: list[NDArray[np.uint8]],
    ) -> tuple[list[str], NDArray[np.float32]]:
        if not images:
            return [], np.empty(0, dtype=np.float32)

        width_list = [img.shape[1] / float(img.shape[0]) for img in images]
        indices = np.argsort(np.asarray(width_list, dtype=np.float32)).tolist()
        rec_res: list[tuple[str, float]] = [("", 0.0)] * len(images)

        for batch_indices, target_width in self._iter_batches(indices, width_list):
            norm_img_batch = np.concatenate(
                [self._resize_norm_img(images[index], target_width)[np.newaxis, :] for index in batch_indices]
            ).astype(np.float32)

            preds = session.run(None, {input_name: norm_img_batch})[0]
            line_results, _ = self.decoder(
                preds,
                False,
                wh_ratio_list=tuple(width_list[index] for index in batch_indices),
                max_wh_ratio=target_width / self._image_height,
            )

            for offset, (text, score) in enumerate(line_results):
                rec_res[batch_indices[offset]] = (text, float(score))

        texts, scores = zip(*rec_res)
        return list(texts), np.asarray(scores, dtype=np.float32)

    def _get_dict_path(self) -> Path:
        dict_url = InferSession.get_dict_key_url(
            FileInfo(
                engine_type=EngineType.PADDLE,
                ocr_version=OCRVersion.PPOCRV5,
                task_type=TaskType.REC,
                lang_type=self.language,
                model_type=_get_rapid_model_type(self.model_name),
            )
        )
        local_dict_path = InferSession.DEFAULT_MODEL_PATH / Path(dict_url).name
        if local_dict_path.is_file():
            return local_dict_path

        dict_path = self.model_dir / Path(dict_url).name
        if dict_path.is_file():
            return dict_path

        self.model_dir.mkdir(parents=True, exist_ok=True)
        DownloadFile.run(
            DownloadFileInput(
                file_url=dict_url,
                sha256=None,
                save_path=dict_path,
                logger=log,
            )
        )
        return dict_path

    def _iter_batches(self, indices: list[int], width_list: list[float]) -> list[tuple[list[int], int]]:
        batches: list[tuple[list[int], int]] = []
        current = 0

        while current < len(indices):
            batch_indices = [indices[current]]
            target_width = self._get_batch_width(width_list[indices[current]])
            current += 1

            while current < len(indices) and len(batch_indices) < self.batch_size:
                next_index = indices[current]
                next_width = self._get_batch_width(width_list[next_index])
                if next_width != target_width:
                    break
                batch_indices.append(next_index)
                current += 1

            batches.append((batch_indices, target_width))

        return batches

    def _get_batch_width(self, max_wh_ratio: float) -> int:
        required_width = max(self._base_width, int(math.ceil(self._image_height * max_wh_ratio)))
        for bucket_width in self._bucket_widths:
            if required_width <= bucket_width:
                return bucket_width

        log.debug(
            "OCR recognition width %s exceeds the largest RKNN bucket %s; clamping to the largest bucket.",
            required_width,
            self._bucket_widths[-1],
        )
        return self._bucket_widths[-1]

    def _resize_norm_img(self, img: NDArray[np.uint8], target_width: int) -> NDArray[np.float32]:
        ratio = img.shape[1] / float(img.shape[0])
        resized_w = min(target_width, int(math.ceil(self._image_height * ratio)))

        normalized = cv2.resize(img, (resized_w, self._image_height)).astype(np.float32)
        normalized = np.transpose(normalized, (2, 0, 1)) / 255.0
        normalized -= 0.5
        normalized /= 0.5

        padded = np.zeros((img.shape[2], self._image_height, target_width), dtype=np.float32)
        padded[:, :, :resized_w] = normalized
        return padded


def _get_rapid_model_type(model_name: str) -> RapidModelType:
    return RapidModelType.MOBILE if "mobile" in model_name.lower() else RapidModelType.SERVER
