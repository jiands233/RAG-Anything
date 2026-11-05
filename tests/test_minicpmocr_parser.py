"""MiniCPM-OCR parser tests. Inference and layout models are never loaded."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from raganything.minicpmocr import (
    MiniCPMOCRParser,
    apply_minicpm_settings,
    load_minicpm_settings,
    normalize_server_url,
)
from raganything.parser import SUPPORTED_PARSERS, get_parser


PROMPT_MAP = {
    "default": {
        "prompt": "Text Recognition:",
        "max_new_tokens": 8192,
        "output_format": "text",
        "resize_min_side": None,
        "slice_nums": 9,
    },
    "element_prompts": {
        "text": {
            "prompt": "Text Recognition:",
            "max_new_tokens": 4096,
            "output_format": "text",
        },
        "table": {
            "prompt": "Table Recognition:",
            "max_new_tokens": 8192,
            "output_format": "html",
        },
        "display_formula": {
            "prompt": "Formula Recognition:",
            "max_new_tokens": 2048,
            "output_format": "latex",
        },
        "doc_title": {
            "prompt": "Text Recognition:",
            "max_new_tokens": 512,
            "output_format": "text",
        },
        "paragraph_title": {
            "prompt": "Text Recognition:",
            "max_new_tokens": 512,
            "output_format": "text",
        },
        "chart": {
            "prompt": "Chart to Table:",
            "max_new_tokens": 4096,
            "output_format": "html",
        },
        "seal": {
            "prompt": "Seal Recognition:",
            "max_new_tokens": 256,
            "output_format": "text",
        },
    },
}


def element_cfg(prompt_map, label):
    """Same lookup the SDK's get_element_cfg uses, without importing it."""
    default = prompt_map.get("default", {})
    elem = prompt_map.get("element_prompts", {}).get(label, {})
    if elem.get("skip", False):
        return None
    return {
        "prompt": elem.get("prompt", default.get("prompt", "Text Recognition:")),
        "max_new_tokens": elem.get(
            "max_new_tokens", default.get("max_new_tokens", 4096)
        ),
        "output_format": elem.get(
            "output_format", default.get("output_format", "text")
        ),
        "resize_min_side": elem.get("resize_min_side", default.get("resize_min_side")),
        "slice_nums": elem.get("slice_nums", default.get("slice_nums", 9)),
    }


class FakeImage:
    def __init__(self, size=(8, 8)):
        self.size = size

    def convert(self, mode):
        return self

    def crop(self, box):
        return FakeImage(size=(box[2] - box[0], box[3] - box[1]))

    def resize(self, size, resample=None):
        return FakeImage(size=size)

    def save(self, path, format=None):
        Path(path).write_bytes(b"png")

    def load(self):
        return None

    def copy(self):
        return FakeImage(self.size)


class Layout:
    def __init__(self, blocks):
        self._blocks = blocks
        self.detect_calls = 0

    def detect(self, images):
        self.detect_calls += 1
        return [[] for _ in images]

    def crop(self, image, detections):
        return self._blocks


class Engine:
    def __init__(self, texts, finish_reason="stop"):
        self._texts = list(texts)
        self.finish_reason = finish_reason
        self.calls = []

    def recognize(self, image, prompt, max_new_tokens, slice_nums=None):
        self.calls.append(
            {
                "image": image,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "slice_nums": slice_nums,
            }
        )
        text = self._texts.pop(0) if self._texts else ""
        result = {"text": text}
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        return result


def _block(
    label="text",
    bbox=None,
    image=None,
    skip_vlm=False,
):
    return {
        "label": label,
        "bbox": [0, 0, 4, 4] if bbox is None else bbox,
        "image": FakeImage() if image is None else image,
        "skip_vlm": skip_vlm,
    }


def _attach(parser, blocks, engine, convert_otsl_to_html=None):
    parser._layout_detector = Layout(blocks)
    parser._engine = engine
    parser._prompt_map = PROMPT_MAP
    parser._sdk = SimpleNamespace(
        get_element_cfg=element_cfg,
        postprocess_content=lambda text, output_format, label="": (text or "").strip(),
        convert_otsl_to_html=convert_otsl_to_html,
    )
    parser._load_rgb_image = lambda path: FakeImage()
    return parser


def _image(tmp_path, name="sample.png"):
    path = tmp_path / name
    path.write_bytes(b"image-bytes")
    return path


def test_supported_parsers_include_minicpmocr():
    assert "minicpmocr" in SUPPORTED_PARSERS


def test_get_parser_returns_minicpmocr_parser():
    parser = get_parser("MiniCPMOCR")
    assert isinstance(parser, MiniCPMOCRParser)


def test_parser_module_import_does_not_import_sdk():
    sys.modules.pop("minicpmvocr", None)
    import raganything.parser as parser_module

    parser_module.get_parser("minicpmocr")
    assert "minicpmvocr" not in sys.modules


def test_check_installation_false_when_dependency_missing():
    parser = MiniCPMOCRParser()
    assert parser.check_installation() is False
    assert "minicpm-ocr" in parser.installation_error


def test_check_installation_true_when_sdk_imports(monkeypatch):
    parser = MiniCPMOCRParser()
    monkeypatch.setattr(parser, "_require_sdk", lambda: object())
    assert parser.check_installation() is True
    assert parser.installation_error is None


def test_check_installation_does_not_require_base_url(monkeypatch):
    monkeypatch.delenv("MINICPM_OCR_BASE_URL", raising=False)
    parser = MiniCPMOCRParser()
    monkeypatch.setattr(parser, "_require_sdk", lambda: object())
    assert parser.check_installation() is True


def test_parse_image_raises_import_error_with_install_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://127.0.0.1:8000")
    parser = MiniCPMOCRParser()
    parser._load_rgb_image = lambda path: FakeImage()
    with pytest.raises(ImportError, match="minicpm-ocr"):
        parser.parse_image(_image(tmp_path))


def test_parse_requires_base_url(monkeypatch, tmp_path):
    monkeypatch.delenv("MINICPM_OCR_BASE_URL", raising=False)
    parser = MiniCPMOCRParser()
    monkeypatch.setattr(parser, "_require_sdk", lambda: object())
    parser._load_rgb_image = lambda path: FakeImage()
    with pytest.raises(ValueError, match="MINICPM_OCR_BASE_URL"):
        parser.parse_image(_image(tmp_path))


def test_normalize_server_url_strips_v1_and_slash():
    assert normalize_server_url("http://localhost:8000/v1/") == "http://localhost:8000"
    assert normalize_server_url("http://localhost:8000/v1") == "http://localhost:8000"
    assert normalize_server_url("http://localhost:8000/") == "http://localhost:8000"


def test_load_settings_reads_env(monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://localhost:8000/v1/")
    monkeypatch.setenv("MINICPM_OCR_BACKEND", "sglang")
    monkeypatch.setenv("MINICPM_OCR_MODEL", "custom-model")
    monkeypatch.setenv("MINICPM_OCR_API_KEY", "secret")
    monkeypatch.setenv("MINICPM_OCR_MAX_MODEL_LEN", "8192")
    monkeypatch.setenv("MINICPM_OCR_STARTUP_TIMEOUT", "15")
    monkeypatch.setenv("MINICPM_OCR_REQUEST_TIMEOUT", "30")
    monkeypatch.setenv("MINICPM_OCR_LAYOUT_DEVICE", "cuda:0")
    monkeypatch.setenv("MINICPM_OCR_LAYOUT_DIR", "~/layouts")

    settings = load_minicpm_settings()

    assert settings["base_url"] == "http://localhost:8000"
    assert settings["backend"] == "sglang"
    assert settings["model"] == "custom-model"
    assert settings["api_key"] == "secret"
    assert settings["max_model_len"] == 8192
    assert settings["startup_timeout_sec"] == 15
    assert settings["request_timeout_sec"] == 30
    assert settings["layout_device"] == "cuda:0"
    assert settings["layout_dir"] == str(Path("~/layouts").expanduser())


def test_load_settings_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("MINICPM_OCR_BACKEND", "nope")
    with pytest.raises(ValueError, match="MINICPM_OCR_BACKEND"):
        load_minicpm_settings()


def test_load_settings_falls_back_on_invalid_int(monkeypatch, caplog):
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("MINICPM_OCR_MAX_MODEL_LEN", "nope")
    settings = load_minicpm_settings()
    assert settings["max_model_len"] == 40960
    assert "Invalid MINICPM_OCR_MAX_MODEL_LEN" in caplog.text


def test_apply_settings_disables_thinking_and_selects_layout():
    settings = {
        "base_url": "http://localhost:8000",
        "model": "minicpm-ocr",
        "backend": "openai",
        "api_key": "secret",
        "max_model_len": 40960,
        "startup_timeout_sec": 60,
        "request_timeout_sec": 300,
        "layout_device": "cpu",
        "layout_dir": "/tmp/layout",
    }
    cfg = apply_minicpm_settings({"engine": {}, "layout": {"device": "cuda"}}, settings)
    assert cfg["engine"]["type"] == "openai"
    assert cfg["engine"]["server_url"] == "http://localhost:8000"
    assert cfg["engine"]["api_key"] == "secret"
    assert cfg["engine"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert cfg["layout"]["provider"] == "ppdoclayout3_paddlex"
    assert cfg["layout"]["device"] == "cpu"


def test_engine_start_failure_is_retried(monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://localhost:8000/v1/")
    parser = MiniCPMOCRParser()
    calls = {"start": 0, "layout": 0}

    class RetryEngine:
        def start(self):
            calls["start"] += 1
            if calls["start"] == 1:
                raise RuntimeError("inference server down")

    class Sdk:
        DEFAULT_CONFIG = {"engine": {}, "layout": {}, "prompt": {}}

        def load_prompt_map(self, cfg):
            return PROMPT_MAP

        def create_layout_detector(self, cfg):
            calls["layout"] += 1
            return Layout([])

        def create_backend(self, cfg):
            return RetryEngine()

        def ensure_layout_model(self, directory):
            return directory

    monkeypatch.setattr(parser, "_require_sdk", lambda: Sdk())

    with pytest.raises(RuntimeError, match="inference server down"):
        parser._ensure_runtime()
    assert parser._engine is None

    parser._ensure_runtime()
    assert parser._engine is not None
    assert calls["start"] == 2
    assert calls["layout"] == 1
    assert parser._cfg["engine"]["server_url"] == "http://localhost:8000"
    assert parser._cfg["engine"]["type"] == "vllm"
    assert parser._cfg["engine"]["model_name"] == "minicpm-ocr"


def test_parse_image_maps_blocks_to_content_list(monkeypatch, tmp_path):
    del monkeypatch
    engine = Engine(["Hello", "Same", "Same"])
    blocks = [_block(), _block(label="text"), _block(label="text")]
    parser = _attach(MiniCPMOCRParser(), blocks, engine)

    content_list = parser.parse_image(_image(tmp_path), page_idx=7)

    assert content_list == [
        {"type": "text", "text": "Hello", "page_idx": 7},
        {"type": "text", "text": "Same", "page_idx": 7},
        {"type": "text", "text": "Same", "page_idx": 7},
    ]
    assert engine.calls[0]["prompt"] == "Text Recognition:"
    assert engine.calls[0]["max_new_tokens"] == 4096
    assert engine.calls[0]["slice_nums"] == 9


def test_parse_image_uses_table_and_formula_prompts(tmp_path):
    engine = Engine(
        [
            "<table><tr><td>A</td></tr></table>",
            "E = mc^2",
            "Paper title",
            "stamped",
        ]
    )
    blocks = [
        _block(label="table"),
        _block(label="display_formula"),
        _block(label="doc_title"),
        _block(label="seal"),
    ]
    parser = _attach(MiniCPMOCRParser(), blocks, engine)

    content_list = parser.parse_image(_image(tmp_path), lang="zh")

    assert [item["type"] for item in content_list] == [
        "table",
        "equation",
        "text",
        "text",
    ]
    assert content_list[0]["table_body"] == "<table><tr><td>A</td></tr></table>"
    assert content_list[0]["table_caption"] == []
    assert content_list[0]["table_footnote"] == []
    assert content_list[1] == {
        "type": "equation",
        "latex": "E = mc^2",
        "text": "E = mc^2",
        "text_format": "latex",
        "page_idx": 0,
    }
    assert content_list[2]["text_level"] == 1
    assert content_list[2]["text"] == "Paper title"
    assert [call["prompt"] for call in engine.calls] == [
        "Table Recognition:",
        "Formula Recognition:",
        "Text Recognition:",
        "Seal Recognition:",
    ]
    assert engine.calls[1]["max_new_tokens"] == 2048


def test_chart_recognition_becomes_a_table(tmp_path):
    engine = Engine(["<table><tr><td>1</td></tr></table>"])
    parser = _attach(MiniCPMOCRParser(), [_block(label="chart")], engine)

    content_list = parser.parse_image(_image(tmp_path))

    assert content_list[0]["type"] == "table"
    assert engine.calls[0]["prompt"] == "Chart to Table:"


def test_unknown_label_uses_default_prompt(tmp_path):
    engine = Engine(["fallback"])
    parser = _attach(MiniCPMOCRParser(), [_block(label="unknown_label")], engine)

    content_list = parser.parse_image(_image(tmp_path))

    assert content_list == [{"type": "text", "text": "fallback", "page_idx": 0}]
    assert engine.calls[0]["prompt"] == "Text Recognition:"


def test_layout_artifacts_are_dropped_unless_requested(tmp_path):
    engine = Engine(["kept", "header text"])
    blocks = [_block(label="text"), _block(label="header")]
    parser = _attach(MiniCPMOCRParser(), blocks, engine)

    dropped = parser.parse_image(_image(tmp_path))
    assert [item["text"] for item in dropped] == ["kept"]
    assert len(engine.calls) == 1

    engine.calls.clear()
    engine._texts = ["kept", "header text"]
    kept = parser.parse_image(_image(tmp_path), include_layout_blocks=True)
    assert [item["text"] for item in kept] == ["kept", "header text"]
    assert len(engine.calls) == 2


def test_skip_vlm_saves_figure_and_skips_chrome(tmp_path):
    engine = Engine(["unused"])
    blocks = [
        _block(label="image", skip_vlm=True),
        _block(label="header_image", skip_vlm=True),
        _block(label="chart", skip_vlm=True),
    ]
    parser = _attach(MiniCPMOCRParser(), blocks, engine)

    content_list = parser.parse_image(
        _image(tmp_path), output_dir=str(tmp_path / "out")
    )

    assert engine.calls == []
    assert [item["type"] for item in content_list] == ["image", "image"]
    for item in content_list:
        assert Path(item["img_path"]).is_absolute()
        assert Path(item["img_path"]).is_file()
        assert item["image_caption"] == []
        assert item["image_footnote"] == []
        assert item["page_idx"] == 0


def test_invalid_bbox_is_skipped(tmp_path, caplog):
    engine = Engine(["never used"])
    blocks = [_block(bbox=[10, 10, 5, 5]), _block(bbox=[0, 0, 10])]
    parser = _attach(MiniCPMOCRParser(), blocks, engine)

    assert parser.parse_image(_image(tmp_path)) == []
    assert engine.calls == []
    assert "invalid bbox" in caplog.text


def test_empty_and_truncated_blocks_are_skipped(tmp_path, caplog):
    engine = Engine(["", "partial text"], finish_reason="length")
    parser = _attach(MiniCPMOCRParser(), [_block(), _block()], engine)

    assert parser.parse_image(_image(tmp_path)) == []
    assert "truncated" in caplog.text


def test_missing_finish_reason_is_kept(tmp_path):
    engine = Engine(["Hello"], finish_reason=None)
    parser = _attach(MiniCPMOCRParser(), [_block()], engine)

    assert parser.parse_image(_image(tmp_path)) == [
        {"type": "text", "text": "Hello", "page_idx": 0}
    ]


def test_page_crop_fallback_when_block_has_no_image(tmp_path):
    engine = Engine(["from page crop"])
    block = _block(image=False)
    block["image"] = None
    parser = _attach(MiniCPMOCRParser(), [block], engine)

    content_list = parser.parse_image(_image(tmp_path))

    assert content_list[0]["text"] == "from page crop"
    assert engine.calls[0]["image"].size == (4, 4)


def test_otsl_table_is_converted_when_converter_is_available(tmp_path):
    engine = Engine(["<fcel>A<fcel>B<nl>"])
    parser = _attach(
        MiniCPMOCRParser(),
        [_block(label="table")],
        engine,
        convert_otsl_to_html=lambda text: (
            "<table><tr><td>A</td><td>B</td></tr></table>"
        ),
    )

    content_list = parser.parse_image(_image(tmp_path))

    assert content_list[0]["table_body"] == (
        "<table><tr><td>A</td><td>B</td></tr></table>"
    )


def test_backend_failure_propagates(tmp_path):
    class FailingEngine(Engine):
        def recognize(self, *args, **kwargs):
            raise RuntimeError("backend down")

    parser = _attach(MiniCPMOCRParser(), [_block()], FailingEngine([]))
    with pytest.raises(RuntimeError, match="backend down"):
        parser.parse_image(_image(tmp_path))


def test_non_dict_prediction_raises(tmp_path):
    class BadEngine(Engine):
        def recognize(self, *args, **kwargs):
            return "not-a-dict"

    parser = _attach(MiniCPMOCRParser(), [_block()], BadEngine([]))
    with pytest.raises(RuntimeError, match="expected a dict"):
        parser.parse_image(_image(tmp_path))


def test_parse_pdf_assigns_page_index(monkeypatch, tmp_path):
    parser = MiniCPMOCRParser()
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        parser,
        "_render_pdf_pages",
        lambda pdf_path: iter([(0, FakeImage()), (1, FakeImage())]),
    )
    seen = []

    def recognize(image, page_idx, output_dir, include_layout_blocks):
        seen.append(page_idx)
        return [{"type": "text", "text": f"page-{page_idx}", "page_idx": page_idx}]

    monkeypatch.setattr(parser, "_recognize_page", recognize)

    assert parser.parse_pdf(pdf) == [
        {"type": "text", "text": "page-0", "page_idx": 0},
        {"type": "text", "text": "page-1", "page_idx": 1},
    ]
    assert seen == [0, 1]


def test_parse_pdf_raises_import_error_when_renderer_missing(monkeypatch, tmp_path):
    parser = MiniCPMOCRParser()
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pypdfium2":
            raise ImportError("missing pypdfium2")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pypdfium2"):
        parser.parse_pdf(pdf)


def test_parse_office_doc_converts_then_parses_pdf(monkeypatch, tmp_path):
    parser = MiniCPMOCRParser()
    doc = tmp_path / "notes.docx"
    doc.write_bytes(b"doc")
    pdf = tmp_path / "notes.pdf"
    seen = {}

    def fake_convert(doc_path, output_dir=None):
        seen["doc"] = Path(doc_path)
        return pdf

    def fake_pdf(pdf_path, output_dir=None, method="auto", lang=None, **kwargs):
        seen["pdf"] = Path(pdf_path)
        return [{"type": "text", "text": "from pdf", "page_idx": 0}]

    monkeypatch.setattr(parser, "convert_office_to_pdf", fake_convert)
    monkeypatch.setattr(parser, "parse_pdf", fake_pdf)

    assert parser.parse_document(doc) == [
        {"type": "text", "text": "from pdf", "page_idx": 0}
    ]
    assert seen["doc"] == doc
    assert seen["pdf"] == pdf


def test_parse_image_rejects_unknown_extension(tmp_path):
    parser = MiniCPMOCRParser()
    path = tmp_path / "notes.svg"
    path.write_text("<svg/>")
    with pytest.raises(ValueError, match="Unsupported image format"):
        parser.parse_image(path)


def test_parse_document_rejects_unknown_extension(tmp_path):
    parser = MiniCPMOCRParser()
    path = tmp_path / "notes.csv"
    path.write_text("a,b\n")
    with pytest.raises(ValueError, match="Unsupported file format"):
        parser.parse_document(path)
