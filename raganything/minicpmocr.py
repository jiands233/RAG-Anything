"""
Optional MiniCPM-OCR parser.

Layout detection runs in-process with the SDK's PP-DocLayoutV3 provider.
Each crop is recognized by an external MiniCPM-OCR 1.0 server (vLLM, SGLang,
llama.cpp, Ollama, or any OpenAI-compatible endpoint) using the prompt for
that block type. The 1.3B model is not bundled.

Nothing in this module imports ``minicpmvocr`` until a parse or an
installation check asks for it.
"""

from __future__ import annotations

import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from raganything.parser import HTML_FILE_EXTENSIONS, Parser, html_requires_docling_error

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "minicpm-ocr"
DEFAULT_BACKEND = "vllm"
DEFAULT_MAX_MODEL_LEN = 40960
DEFAULT_STARTUP_TIMEOUT = 60
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_LAYOUT_DIR = "~/.cache/minicpm-ocr/layout"
SUPPORTED_BACKENDS = ("vllm", "sglang", "openai", "llamacpp", "ollama")

# Same labels the SDK's markdown assembly drops: repeated page chrome.
_LAYOUT_ARTIFACT_LABELS = frozenset(
    {
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
    }
)
# Crops the VLM is often told to skip. Keep them as images so the multimodal
# pipeline can still describe figures, charts, and seals.
_VISUAL_LABELS = frozenset({"image", "header_image", "footer_image", "chart", "seal"})
_HEADING_LEVELS = {"doc_title": 1, "paragraph_title": 2}
_OTSL_MARKERS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>")

_INSTALL_HINT = (
    "MiniCPM-OCR parser requires the optional `minicpm-ocr` package. "
    "Install with `pip install -e '.[minicpmocr]'` or "
    "`uv sync --extra minicpmocr`. "
    "Layout detection also needs `paddlepaddle` (CPU and GPU builds differ "
    "by platform); install it from "
    "https://www.paddlepaddle.org.cn/install/quick. "
    "Recognition calls an external MiniCPM-OCR server: set MINICPM_OCR_BASE_URL."
)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def normalize_server_url(url: str) -> str:
    """The SDK expects the server root, without a trailing ``/v1``."""
    url = (url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def load_minicpm_settings() -> Dict[str, Any]:
    """Read MiniCPM-OCR settings from the environment.

    Raises:
        ValueError: If the base URL is missing or the backend name is unknown.
    """
    base_url = normalize_server_url(_env("MINICPM_OCR_BASE_URL"))
    if not base_url:
        raise ValueError(
            "MINICPM_OCR_BASE_URL is required and must point at a MiniCPM-OCR "
            "inference server root, for example http://127.0.0.1:8000 "
            "(a trailing /v1 is removed)."
        )
    backend = _env("MINICPM_OCR_BACKEND", DEFAULT_BACKEND).lower()
    if backend not in SUPPORTED_BACKENDS:
        allowed = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(
            f"MINICPM_OCR_BACKEND must be one of {allowed}; got {backend!r}."
        )
    layout_device = _env("MINICPM_OCR_LAYOUT_DEVICE", "cpu") or "cpu"
    return {
        "base_url": base_url,
        "model": _env("MINICPM_OCR_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        "backend": backend,
        "api_key": _env("MINICPM_OCR_API_KEY"),
        "max_model_len": _env_int("MINICPM_OCR_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN),
        "startup_timeout_sec": _env_int(
            "MINICPM_OCR_STARTUP_TIMEOUT", DEFAULT_STARTUP_TIMEOUT
        ),
        "request_timeout_sec": _env_int(
            "MINICPM_OCR_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT
        ),
        "layout_device": layout_device,
        "layout_dir": str(
            Path(_env("MINICPM_OCR_LAYOUT_DIR", DEFAULT_LAYOUT_DIR)).expanduser()
        ),
    }


def apply_minicpm_settings(
    cfg: Dict[str, Any], settings: Dict[str, Any]
) -> Dict[str, Any]:
    """Write ``settings`` into a MiniCPM-OCR SDK config mapping."""
    cfg.setdefault("engine", {}).update(
        {
            "type": settings["backend"],
            "model_name": settings["model"],
            "server_url": settings["base_url"],
            "api_key": settings["api_key"],
            "max_model_len": settings["max_model_len"],
            "startup_timeout_sec": settings["startup_timeout_sec"],
            "request_timeout_sec": settings["request_timeout_sec"],
            # The transcription lives in message content. Leaving thinking on
            # can park it in reasoning_content, which the SDK client ignores.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    )
    cfg.setdefault("layout", {}).update(
        {
            "provider": "ppdoclayout3_paddlex",
            "model_dir": "",
            "device": settings["layout_device"],
        }
    )
    return cfg


def _valid_bbox(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _resize_to_min_side(image: Any, target_min_side: Optional[int]) -> Any:
    if not target_min_side:
        return image
    width, height = image.size
    if min(width, height) >= int(target_min_side):
        return image
    scale = int(target_min_side) / min(width, height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    try:
        from PIL import Image

        return image.resize(new_size, Image.Resampling.LANCZOS)
    except ImportError:
        return image.resize(new_size)


def _looks_like_otsl(text: str) -> bool:
    return any(marker in text for marker in _OTSL_MARKERS)


class MiniCPMOCRParser(Parser):
    """MiniCPM-OCR document parser.

    ``language`` is accepted on every entry point and ignored: one
    MiniCPM-OCR 1.0 model covers Chinese and English. Settings are read from
    the environment on the first parse (see :func:`load_minicpm_settings`).

    Layout artifacts (page headers, footers, page numbers, footnotes, and
    aside text) are omitted unless ``include_layout_blocks=True``, matching
    MinerU's default.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sdk: Any = None
        self._settings: Optional[Dict[str, Any]] = None
        self._cfg: Optional[Dict[str, Any]] = None
        self._prompt_map: Any = None
        self._layout_detector: Any = None
        self._engine: Any = None
        self._image_seq = 0
        self._init_lock = threading.Lock()
        # Paddle's predictor and the SDK client are single-threaded.
        self._infer_lock = threading.Lock()

    def _require_sdk(self) -> Any:
        try:
            from minicpmvocr import (
                DEFAULT_CONFIG,
                create_backend,
                create_layout_detector,
                get_element_cfg,
                load_prompt_map,
            )
            from minicpmvocr.layouts.download import ensure_layout_model
            from minicpmvocr.postprocess.markdown import postprocess_content
            from minicpmvocr.postprocess.otsl import convert_otsl_to_html
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        return _Sdk(
            DEFAULT_CONFIG=DEFAULT_CONFIG,
            create_backend=create_backend,
            create_layout_detector=create_layout_detector,
            get_element_cfg=get_element_cfg,
            load_prompt_map=load_prompt_map,
            ensure_layout_model=ensure_layout_model,
            postprocess_content=postprocess_content,
            convert_otsl_to_html=convert_otsl_to_html,
        )

    def _runtime_ready(self) -> bool:
        return (
            self._engine is not None
            and self._layout_detector is not None
            and self._prompt_map is not None
            and self._sdk is not None
        )

    def _ensure_runtime(self) -> None:
        """Start layout detection and the recognition client once.

        A failed engine start leaves ``_engine`` unset so the next parse
        retries the connection without reloading the layout model.
        """
        if self._runtime_ready():
            return
        with self._init_lock:
            if self._runtime_ready():
                return
            if self._sdk is None:
                self._sdk = self._require_sdk()
            if self._settings is None:
                self._settings = load_minicpm_settings()
            if self._cfg is None:
                self._cfg = apply_minicpm_settings(
                    deepcopy(self._sdk.DEFAULT_CONFIG), self._settings
                )
            if self._prompt_map is None:
                self._prompt_map = self._sdk.load_prompt_map(self._cfg)
            if self._layout_detector is None:
                layout_dir = Path(self._settings["layout_dir"])
                self._cfg["layout"]["model_dir"] = str(
                    self._sdk.ensure_layout_model(layout_dir)
                )
                self._layout_detector = self._sdk.create_layout_detector(
                    self._cfg["layout"]
                )
            if self._engine is None:
                engine = self._sdk.create_backend(self._cfg)
                engine.start()
                self._engine = engine

    def _load_rgb_image(self, image_path: Path) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-OCR image parsing requires Pillow, which is installed "
                "with the minicpmocr extra: `pip install -e '.[minicpmocr]'`."
            ) from exc
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            rgb.load()
            return rgb.copy()

    def _render_pdf_pages(self, pdf_path: Path) -> Iterator[Tuple[int, Any]]:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ImportError(
                "PDF parsing with parser='minicpmocr' requires `pypdfium2`. "
                "Install with `pip install -e '.[minicpmocr]'` or "
                "`uv sync --extra minicpmocr`."
            ) from exc

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                try:
                    rendered = page.render(scale=2.0)
                    if not hasattr(rendered, "to_pil"):
                        raise RuntimeError(
                            "Unsupported rendered page format from pypdfium2."
                        )
                    image = rendered.to_pil().convert("RGB")
                    image.load()
                    # Copy before the page bitmap is released.
                    yield page_idx, image.copy()
                finally:
                    if hasattr(page, "close"):
                        page.close()
        finally:
            if hasattr(pdf, "close"):
                pdf.close()

    def _artifact_dir(self, source: Path, output_dir: Optional[str]) -> Path:
        if output_dir:
            return Path(output_dir)
        return source.parent / f"{source.stem}_minicpmocr"

    def _save_image(self, image: Any, output_dir: Path, page_idx: int) -> str:
        self._image_seq += 1
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"minicpmocr_p{page_idx}_{self._image_seq}.png"
        image.save(path, format="PNG")
        return str(path.resolve())

    def _block_image(self, page: Any, block: Dict[str, Any]) -> Optional[Any]:
        bbox = _valid_bbox(block.get("bbox"))
        if bbox is None:
            logger.warning(
                "Skipping %s block with invalid bbox: %r",
                block.get("label", ""),
                block.get("bbox"),
            )
            return None
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        crop = block.get("image")
        image = self._coerce_image(crop)
        if image is None and x2 > x1 and y2 > y1 and hasattr(page, "crop"):
            image = self._coerce_image(page.crop((x1, y1, x2, y2)))
        return image

    def _coerce_image(self, crop: Any) -> Optional[Any]:
        if crop is None:
            return None
        if getattr(crop, "size", None) == 0:
            return None
        if hasattr(crop, "convert"):
            image = crop.convert("RGB")
        else:
            try:
                from PIL import Image

                image = Image.fromarray(crop).convert("RGB")
            except Exception:
                logger.warning("Skipping a layout crop that is not an image")
                return None
        width, height = getattr(image, "size", (0, 0))
        if width <= 0 or height <= 0:
            return None
        return image

    def _postprocess(self, text: str, output_format: str, label: str) -> str:
        fn = getattr(self._sdk, "postprocess_content", None)
        if fn is None:
            return (text or "").strip()
        try:
            return fn(text, output_format, label=label) or ""
        except Exception:
            logger.warning(
                "MiniCPM-OCR postprocess failed for %s; keeping raw text",
                label,
                exc_info=True,
            )
            return (text or "").strip()

    def _table_body(self, text: str, label: str) -> str:
        if label not in {"table", "chart"} or not _looks_like_otsl(text):
            return text
        converter = getattr(self._sdk, "convert_otsl_to_html", None)
        if converter is None:
            return text
        try:
            html = converter(text)
        except Exception:
            logger.warning(
                "OTSL to HTML conversion failed; keeping raw table text",
                exc_info=True,
            )
            return text
        return html or text

    def _content_item(
        self,
        label: str,
        text: str,
        output_format: str,
        page_idx: int,
    ) -> Optional[Dict[str, Any]]:
        cleaned = self._postprocess(text, output_format, label).strip()
        if not cleaned:
            return None
        if output_format == "latex" or label in {"display_formula", "inline_formula"}:
            return {
                "type": "equation",
                "latex": cleaned,
                "text": cleaned,
                "text_format": "latex",
                "page_idx": page_idx,
            }
        if output_format == "html" or label in {"table", "chart"}:
            return {
                "type": "table",
                "table_body": self._table_body(cleaned, label),
                "table_caption": [],
                "table_footnote": [],
                "page_idx": page_idx,
            }
        item: Dict[str, Any] = {"type": "text", "text": cleaned, "page_idx": page_idx}
        level = _HEADING_LEVELS.get(label)
        if level is not None:
            item["text_level"] = level
        return item

    def _image_item(
        self, image: Any, output_dir: Path, page_idx: int
    ) -> Dict[str, Any]:
        return {
            "type": "image",
            "img_path": self._save_image(image, output_dir, page_idx),
            "image_caption": [],
            "image_footnote": [],
            "page_idx": page_idx,
        }

    def _recognize_page(
        self,
        page: Any,
        page_idx: int,
        output_dir: Path,
        include_layout_blocks: bool,
    ) -> List[Dict[str, Any]]:
        self._ensure_runtime()
        with self._infer_lock:
            detected = self._layout_detector.detect([page]) or []
            detections = detected[0] if detected else []
            blocks = self._layout_detector.crop(page, detections)
            content_list: List[Dict[str, Any]] = []
            for block in blocks:
                item = self._block_to_content(
                    page,
                    block,
                    page_idx=page_idx,
                    output_dir=output_dir,
                    include_layout_blocks=include_layout_blocks,
                )
                if item is not None:
                    content_list.append(item)
            return content_list

    def _block_to_content(
        self,
        page: Any,
        block: Dict[str, Any],
        page_idx: int,
        output_dir: Path,
        include_layout_blocks: bool,
    ) -> Optional[Dict[str, Any]]:
        label = str(block.get("label") or "")
        if label in _LAYOUT_ARTIFACT_LABELS and not include_layout_blocks:
            return None

        elem_cfg = self._sdk.get_element_cfg(self._prompt_map, label)
        skip_vlm = bool(block.get("skip_vlm", False)) or elem_cfg is None
        if skip_vlm:
            if label not in _VISUAL_LABELS:
                return None
            image = self._block_image(page, block)
            if image is None:
                return None
            return self._image_item(image, output_dir, page_idx)

        image = self._block_image(page, block)
        if image is None:
            return None
        model_image = _resize_to_min_side(image, elem_cfg.get("resize_min_side"))
        prediction = self._engine.recognize(
            model_image,
            prompt=elem_cfg["prompt"],
            max_new_tokens=elem_cfg["max_new_tokens"],
            slice_nums=elem_cfg.get("slice_nums"),
        )
        if not isinstance(prediction, dict):
            raise RuntimeError(
                f"MiniCPM-OCR backend returned {type(prediction).__name__}, expected a dict."
            )
        finish_reason = prediction.get("finish_reason")
        if finish_reason not in (None, "", "stop"):
            logger.warning(
                "Skipping truncated %s block: finish_reason=%s", label, finish_reason
            )
            return None
        return self._content_item(
            label,
            str(prediction.get("text") or ""),
            str(elem_cfg.get("output_format") or "text"),
            page_idx,
        )

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        del method, lang
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        include_layout_blocks = bool(kwargs.get("include_layout_blocks", False))
        artifact_dir = self._artifact_dir(pdf_path, output_dir)
        content_list: List[Dict[str, Any]] = []
        pages = self._render_pdf_pages(pdf_path)
        try:
            for page_idx, image in pages:
                content_list.extend(
                    self._recognize_page(
                        image,
                        page_idx=int(page_idx),
                        output_dir=artifact_dir,
                        include_layout_blocks=include_layout_blocks,
                    )
                )
        finally:
            close = getattr(pages, "close", None)
            if callable(close):
                close()
        return content_list

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        del lang
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        ext = image_path.suffix.lower()
        if ext not in self.IMAGE_FORMATS:
            supported = ", ".join(sorted(self.IMAGE_FORMATS))
            raise ValueError(
                f"Unsupported image format: {ext}. Supported formats: {supported}"
            )

        page_idx = int(kwargs.get("page_idx", 0))
        include_layout_blocks = bool(kwargs.get("include_layout_blocks", False))
        image = self._load_rgb_image(image_path)
        return self._recognize_page(
            image,
            page_idx=page_idx,
            output_dir=self._artifact_dir(image_path, output_dir),
            include_layout_blocks=include_layout_blocks,
        )

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        pdf_path = self.convert_office_to_pdf(doc_path, output_dir)
        return self.parse_pdf(
            pdf_path=pdf_path, output_dir=output_dir, lang=lang, **kwargs
        )

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self.parse_pdf(
                file_path, output_dir, method=method, lang=lang, **kwargs
            )
        if ext in self.IMAGE_FORMATS:
            return self.parse_image(file_path, output_dir, lang=lang, **kwargs)
        if ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(file_path, output_dir, lang=lang, **kwargs)
        if ext in self.TEXT_FORMATS:
            return self.parse_text_file(file_path, output_dir, lang=lang, **kwargs)
        if ext in HTML_FILE_EXTENSIONS:
            raise html_requires_docling_error(file_path)

        raise ValueError(
            f"Unsupported file format: {ext}. "
            "MiniCPM-OCR parser supports PDF, image, office, and text formats."
        )

    def check_installation(self) -> bool:
        """Return True when ``minicpmvocr`` imports.

        This does not contact the inference server and does not download the
        layout model. A missing ``MINICPM_OCR_BASE_URL`` surfaces on the first
        parse.
        """
        try:
            self._require_sdk()
        except ImportError as exc:
            self.installation_error = str(exc)
            self.logger.debug(self.installation_error)
            return False
        self.installation_error = None
        return True


class _Sdk:
    """Callables pulled from ``minicpmvocr`` (or a test double)."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
