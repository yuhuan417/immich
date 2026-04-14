import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from rapidocr.ch_ppocr_rec import TextRecInput
from rapidocr.ch_ppocr_rec import TextRecognizer as RapidTextRecognizer
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
from rapidocr.utils.typings import EngineType, LangRec, OCRVersion, TaskType
from rapidocr.utils.typings import ModelType as RapidModelType
from rapidocr.utils.vis_res import VisRes

from immich_ml.config import log, settings
from immich_ml.models.base import InferenceModel
from immich_ml.models.transforms import pil_to_cv2
from immich_ml.schemas import ModelFormat, ModelSession, ModelTask, ModelType
from immich_ml.sessions import rknn
from immich_ml.sessions.ort import OrtSession

from .schemas import OcrOptions, TextDetectionOutput, TextRecognitionOutput


class TextRecognizer(InferenceModel):
    depends = [(ModelType.DETECTION, ModelTask.OCR)]
    identity = (ModelType.RECOGNITION, ModelTask.OCR)

    def __init__(self, model_name: str, min_score: float = 0.9, **model_kwargs: Any) -> None:
        self.language = LangRec[model_name.split("__")[0]] if "__" in model_name else LangRec.CH
        self.min_score = model_kwargs.get("minScore", min_score)
        max_batch_size = settings.max_batch_size and settings.max_batch_size.ocr
        self.batch_size = max_batch_size if max_batch_size else 6
        self.input_name = "x"
        self.decoder: CTCLabelDecode | None = None
        self._bucket_widths = (320, 640, 960)
        self._image_height = 48
        self._base_width = 320
        self._empty: TextRecognitionOutput = {
            "box": np.empty(0, dtype=np.float32),
            "boxScore": np.empty(0, dtype=np.float32),
            "text": [],
            "textScore": np.empty(0, dtype=np.float32),
        }
        VisRes.__init__ = lambda self, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]

        model_format = model_kwargs.pop(
            "model_format",
            ModelFormat.RKNN if rknn.is_available else ModelFormat.ONNX,
        )
        super().__init__(model_name, **model_kwargs, model_format=model_format)

    def _download(self) -> None:
        if self.model_format == ModelFormat.RKNN:
            try:
                return super()._download()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to download OCR recognition model '%s' for %s; falling back to ONNX if available.",
                    self.model_name,
                    self.model_format.upper(),
                    exc_info=exc,
                )
                self.model_format = ModelFormat.ONNX

        model_info = InferSession.get_model_url(
            FileInfo(
                engine_type=EngineType.ONNXRUNTIME,
                ocr_version=OCRVersion.PPOCRV5,
                task_type=TaskType.REC,
                lang_type=self.language,
                model_type=RapidModelType.MOBILE if "mobile" in self.model_name else RapidModelType.SERVER,
            )
        )
        download_params = DownloadFileInput(
            file_url=model_info["model_dir"],
            sha256=model_info["SHA256"],
            save_path=self.model_path_for_format(ModelFormat.ONNX),
            logger=log,
        )
        DownloadFile.run(download_params)

    def _load(self) -> ModelSession:
        if self.model_format == ModelFormat.RKNN:
            session = self._make_session(self.model_path_for_format(self.model_format))
            inputs = session.get_inputs()
            if inputs:
                self.input_name = inputs[0].name or self.input_name
            self.decoder = CTCLabelDecode(character_path=self._get_dict_path())
            return session

        # Keep the mainline behavior for non-RKNN backends.
        session = OrtSession(self.model_path_for_format(ModelFormat.ONNX))
        self.model = RapidTextRecognizer(
            OcrOptions(
                session=session.session,
                rec_batch_num=self.batch_size,
                rec_img_shape=(3, 48, 320),
                lang_type=self.language,
            )
        )
        return session

    def _get_dict_path(self) -> Path:
        dict_url = InferSession.get_dict_key_url(
            FileInfo(
                engine_type=EngineType.PADDLE,
                ocr_version=OCRVersion.PPOCRV5,
                task_type=TaskType.REC,
                lang_type=self.language,
                model_type=RapidModelType.MOBILE if "mobile" in self.model_name else RapidModelType.SERVER,
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

    def _predict(self, img: Image.Image, texts: TextDetectionOutput) -> TextRecognitionOutput:
        boxes, box_scores = texts["boxes"], texts["scores"]
        if boxes.shape[0] == 0:
            return self._empty

        normalized_boxes = boxes.astype(np.float32, copy=True)
        normalized_boxes[:, :, 0] /= img.width
        normalized_boxes[:, :, 1] /= img.height

        if self.model_format == ModelFormat.RKNN:
            rec_texts, rec_scores = self._predict_batch(self.get_crop_img_list(img, boxes))
            valid_score_idx = rec_scores > self.min_score
            valid_score_idx_list = valid_score_idx.tolist()

            return {
                "box": normalized_boxes.reshape(-1, 8)[valid_score_idx].reshape(-1),
                "text": [rec_texts[i] for i in range(len(rec_texts)) if valid_score_idx_list[i]],
                "boxScore": box_scores[valid_score_idx],
                "textScore": rec_scores[valid_score_idx],
            }

        rec = self.model(TextRecInput(img=self.get_crop_img_list(img, boxes)))
        if rec.txts is None:
            return self._empty

        text_scores = np.array(rec.scores, dtype=np.float32)
        valid_text_score_idx = text_scores > self.min_score
        valid_score_idx_list = valid_text_score_idx.tolist()
        return {
            "box": normalized_boxes.reshape(-1, 8)[valid_text_score_idx].reshape(-1),
            "text": [rec.txts[i] for i in range(len(rec.txts)) if valid_score_idx_list[i]],
            "boxScore": box_scores[valid_text_score_idx],
            "textScore": text_scores[valid_text_score_idx],
        }

    def _predict_batch(self, images: list[NDArray[np.uint8]]) -> tuple[list[str], NDArray[np.float32]]:
        if not images:
            return [], np.empty(0, dtype=np.float32)
        if self.decoder is None:
            raise RuntimeError("OCR decoder is not initialized")

        width_list = [img.shape[1] / float(img.shape[0]) for img in images]
        indices = np.argsort(np.asarray(width_list, dtype=np.float32)).tolist()
        rec_res: list[tuple[str, float]] = [("", 0.0)] * len(images)

        for batch_indices, target_width in self._iter_batches(indices, width_list):
            norm_img_batch = np.concatenate(
                [self._resize_norm_img(images[index], target_width)[np.newaxis, :] for index in batch_indices]
            ).astype(np.float32)

            preds = self.session.run(None, {self.input_name: norm_img_batch})[0]
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

    def _iter_batches(self, indices: list[int], width_list: list[float]) -> list[tuple[list[int], int]]:
        batches: list[tuple[list[int], int]] = []
        current = 0
        batch_size = self.batch_size or 1

        while current < len(indices):
            batch_indices = [indices[current]]
            target_width = self._get_batch_width(width_list[indices[current]])
            current += 1

            while current < len(indices) and len(batch_indices) < batch_size:
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

    def get_crop_img_list(self, img: Image.Image, boxes: NDArray[np.float32]) -> list[NDArray[np.uint8]]:
        img_crop_width = np.maximum(
            np.linalg.norm(boxes[:, 1] - boxes[:, 0], axis=1), np.linalg.norm(boxes[:, 2] - boxes[:, 3], axis=1)
        ).astype(np.int32)
        img_crop_height = np.maximum(
            np.linalg.norm(boxes[:, 0] - boxes[:, 3], axis=1), np.linalg.norm(boxes[:, 1] - boxes[:, 2], axis=1)
        ).astype(np.int32)
        pts_std = np.zeros((img_crop_width.shape[0], 4, 2), dtype=np.float32)
        pts_std[:, 1:3, 0] = img_crop_width[:, None]
        pts_std[:, 2:4, 1] = img_crop_height[:, None]

        img_crop_sizes = np.stack([img_crop_width, img_crop_height], axis=1)
        all_coeffs = self._get_perspective_transform(pts_std, boxes)
        imgs: list[NDArray[np.uint8]] = []
        for coeffs, dst_size in zip(all_coeffs, img_crop_sizes):
            dst_img = img.transform(
                size=tuple(dst_size),
                method=Image.Transform.PERSPECTIVE,
                data=tuple(coeffs),
                resample=Image.Resampling.BICUBIC,
            )

            dst_width, dst_height = dst_img.size
            if dst_height * 1.0 / dst_width >= 1.5:
                dst_img = dst_img.rotate(90, expand=True)
            imgs.append(pil_to_cv2(dst_img))

        return imgs

    def _get_perspective_transform(self, src: NDArray[np.float32], dst: NDArray[np.float32]) -> NDArray[np.float32]:
        N = src.shape[0]
        x, y = src[:, :, 0], src[:, :, 1]
        u, v = dst[:, :, 0], dst[:, :, 1]
        A = np.zeros((N, 8, 9), dtype=np.float32)

        # Fill even rows (0, 2, 4, 6): [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        A[:, ::2, 0] = x
        A[:, ::2, 1] = y
        A[:, ::2, 2] = 1
        A[:, ::2, 6] = -u * x
        A[:, ::2, 7] = -u * y
        A[:, ::2, 8] = -u

        # Fill odd rows (1, 3, 5, 7): [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
        A[:, 1::2, 3] = x
        A[:, 1::2, 4] = y
        A[:, 1::2, 5] = 1
        A[:, 1::2, 6] = -v * x
        A[:, 1::2, 7] = -v * y
        A[:, 1::2, 8] = -v

        # Solve using SVD for all matrices at once
        _, _, Vt = np.linalg.svd(A)
        H = Vt[:, -1, :].reshape(N, 3, 3)
        H = H / H[:, 2:3, 2:3]

        # Extract the 8 coefficients for each transformation
        return np.column_stack(
            [H[:, 0, 0], H[:, 0, 1], H[:, 0, 2], H[:, 1, 0], H[:, 1, 1], H[:, 1, 2], H[:, 2, 0], H[:, 2, 1]]
        )  # pyright: ignore[reportReturnType]

    def configure(self, **kwargs: Any) -> None:
        self.min_score = kwargs.get("minScore", self.min_score)
