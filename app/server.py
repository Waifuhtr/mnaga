"""
Web service for translating manga pages into Turkish.

Pipeline per page (all of it upstream manga-image-translator, unmodified):

    detection -> OCR -> mask refinement -> inpainting -> translation -> rendering

Only the translation step is ours: it goes to a local llama-server running
Hy-MT2-7B through upstream's `custom_openai` translator, which needs no patching
because Hy-MT2 already honours the `<|1|>`-style numbered batch protocol that
upstream uses.

Everything is loaded from files baked into the Docker image. This module never
downloads a model.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("mnaga")

# --------------------------------------------------------------------------- #
# Paths and configuration
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
MIT_ROOT = Path(os.environ.get("MIT_ROOT", "/opt/manga-image-translator"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/data/work"))
GPT_CONFIG = Path(os.environ.get("GPT_CONFIG_PATH", REPO_DIR / "config" / "gpt_config.yaml"))

# --------------------------------------------------------------------------- #
# Build provenance
# --------------------------------------------------------------------------- #
# The Dockerfile pulls https://api.github.com/repos/.../commits/<branch> into
# this file to bust the layer cache when the branch moves. It is also the only
# record inside the container of WHICH commit was cloned, since the clone's
# .git directory is deleted to keep the image small.
#
# Reading it back turns "is the Space actually running my latest push?" into a
# glance at /health instead of a diagnostic session. A whole round of debugging
# went into a build that predated the code being looked for, which is exactly
# what this prevents.
BUILD_INFO_PATH = Path(os.environ.get("BUILD_INFO_PATH", "/tmp/app-commit.json"))


def _read_build_info() -> Dict[str, Any]:
    try:
        data = json.loads(BUILD_INFO_PATH.read_text())
        commit = data.get("commit") or {}
        message = (commit.get("message") or "").splitlines()
        return {
            "commit": (data.get("sha") or "")[:7] or None,
            "committed_at": (commit.get("author") or {}).get("date"),
            "subject": message[0][:120] if message else None,
        }
    except Exception as exc:
        # Not fatal: an image built some other way just reports nothing.
        log.warning("build info unavailable (%s): %s", BUILD_INFO_PATH, exc)
        return {"commit": None, "committed_at": None, "subject": None}


BUILD_INFO = _read_build_info()


def _asset_version() -> str:
    """
    Cache-busting token for /static assets.

    StaticFiles serves ETag/Last-Modified, so browsers hold on to app.js and
    style.css across rebuilds while index.html - served by our own route, with
    no validators - always comes back fresh. That combination shipped a NEW
    page driven by an OLD script: the font picker sat on its "Yükleniyor…"
    placeholder forever because the cached script had no code to fill it, and
    with the picker dead the detector could not be chosen either. Two tabs of
    the same Space behaved differently for a whole test round because of it.

    Tying the asset URLs to the build makes a rebuild invalidate them.
    """
    if BUILD_INFO.get("commit"):
        return BUILD_INFO["commit"]
    # No build info (local run): fall back to the newest static file's mtime.
    try:
        return str(max(int(f.stat().st_mtime) for f in STATIC_DIR.glob("*") if f.is_file()))
    except (ValueError, OSError):
        return "dev"


log.info("build: commit=%s committed_at=%s %s",
         BUILD_INFO["commit"] or "<unknown>",
         BUILD_INFO["committed_at"] or "<unknown>",
         BUILD_INFO["subject"] or "")

LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8081"))

# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
# assets/fonts/ in this repo is the drop-in folder: every font file in it shows
# up in the UI's font picker. Adding a font is a commit, not a code change -
# nothing here lists font names, they are discovered by scanning the folder.
# The folder rides into the image with the plain `git clone` of this repo
# (Dockerfile stage 9), so no Dockerfile change is needed either.
#
# One font renders the whole page. Upstream calls text_render.set_font() once
# per page, so a per-region font (dialogue vs SFX) would need a different
# mechanism than this - deliberately out of scope here.
FONT_DIR_LOCAL = REPO_DIR / "assets" / "fonts"
FONT_DIR_UPSTREAM = MIT_ROOT / "fonts"
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc")

# Upstream's fonts/ also holds Arial-Unicode / msyh / msgothic, which are the
# fallback faces text_render reaches for when a glyph is missing - they are not
# lettering faces, so they are not offered as a choice. These three are.
UPSTREAM_FONTS = ("comic shanns 2.ttf", "anime_ace.ttf", "anime_ace_3.ttf")

# Always present (bundled upstream) and has full Turkish coverage, so it is a
# safe last resort when a requested font is gone.
FALLBACK_FONT_FILE = "comic shanns 2.ttf"

# Picked by filename stem, so it keeps working if the file is replaced by a
# patched version, and can be overridden per deployment.
DEFAULT_FONT_KEY = os.environ.get("RENDER_FONT_KEY", "ccwildwords")

# The letters Turkish needs beyond ASCII. A face missing some of these still
# renders - text_render falls through to FALLBACK_FONTS (Arial-Unicode first),
# so nothing comes out as tofu, those particular letters just arrive in another
# face, which reads as a style mismatch mid-word. The UI says which fonts do
# that instead of leaving it to be discovered on a finished page.
TURKISH_GLYPHS = "çÇğĞıİöÖşŞüÜ"

_font_cache: Dict[str, Any] = {"sig": None, "fonts": {}}


def font_key(path: Path) -> str:
    """Stable, URL-safe id for a font file, derived from its name."""
    key = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    return key or "font"


def _missing_turkish_glyphs(path: Path) -> Optional[str]:
    """
    Which Turkish letters this face lacks, or None if it could not be checked.

    freetype-py is what upstream renders with, so it is always installed
    alongside us; the guard is only so a font the library chokes on degrades to
    "unknown coverage" instead of taking the whole font list down.
    """
    try:
        import freetype  # noqa: PLC0415 - optional at import time, present at runtime

        face = freetype.Face(str(path))
        return "".join(ch for ch in TURKISH_GLYPHS if face.get_char_index(ch) == 0)
    except Exception as exc:  # pragma: no cover - depends on the dropped-in file
        log.warning("could not read glyph coverage of %s: %s", path.name, exc)
        return None


def _scan_font_dirs() -> Dict[str, Dict[str, Any]]:
    fonts: Dict[str, Dict[str, Any]] = {}

    # Ours first, so dropping in a patched "anime_ace_3.ttf" (or any upstream
    # name) shadows upstream's copy instead of appearing twice.
    candidates: List[Path] = []
    if FONT_DIR_LOCAL.is_dir():
        candidates += sorted(
            p for p in FONT_DIR_LOCAL.iterdir()
            if p.is_file() and p.suffix.lower() in FONT_EXTENSIONS
        )
    candidates += [FONT_DIR_UPSTREAM / name for name in UPSTREAM_FONTS]

    for path in candidates:
        if not path.is_file():
            continue
        key = font_key(path)
        if key in fonts:
            continue
        missing = _missing_turkish_glyphs(path)
        fonts[key] = {
            "key": key,
            "file": path.name,
            "path": str(path),
            # The filename is the label: whatever the user names the file is
            # what they see in the picker.
            "label": path.stem,
            "source": "repo" if path.parent == FONT_DIR_LOCAL else "upstream",
            "turkish": "unknown" if missing is None else ("full" if not missing else "partial"),
            "missing_glyphs": missing or "",
        }
    return fonts


def list_fonts() -> Dict[str, Dict[str, Any]]:
    """
    Every selectable font, keyed by font_key(). Re-scanned when the drop-in
    folder changes so a font added to a running container is picked up without
    a restart; otherwise served from cache, since the glyph check opens files.
    """
    try:
        sig = (
            FONT_DIR_LOCAL.stat().st_mtime_ns if FONT_DIR_LOCAL.is_dir() else None,
            FONT_DIR_UPSTREAM.stat().st_mtime_ns if FONT_DIR_UPSTREAM.is_dir() else None,
        )
    except OSError:
        sig = None

    if _font_cache["sig"] != sig or not _font_cache["fonts"]:
        _font_cache["fonts"] = _scan_font_dirs()
        _font_cache["sig"] = sig
        log.info("fonts available: %s", ", ".join(_font_cache["fonts"]) or "<none>")
    return _font_cache["fonts"]


def default_font_key() -> str:
    """The configured default if it exists, else whatever is first in the list."""
    fonts = list_fonts()
    if DEFAULT_FONT_KEY in fonts:
        return DEFAULT_FONT_KEY
    fallback = font_key(Path(FALLBACK_FONT_FILE))
    if fallback in fonts:
        return fallback
    return next(iter(fonts), "")


def resolve_font(key: Optional[str]) -> Optional[Path]:
    """
    Map a font key from the UI to a file on disk. Never returns a path that does
    not exist: an unknown key falls back to the default, then to any font at all.
    """
    fonts = list_fonts()
    entry = fonts.get(key or "") or fonts.get(default_font_key())
    if entry:
        return Path(entry["path"])
    log.error("no usable font in %s or %s", FONT_DIR_LOCAL, FONT_DIR_UPSTREAM)
    return None


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
# Measured on a real 1280x1816 page against the actual DBNet model: at
# upstream's defaults (text 0.5 / box 0.7) a small two-word bubble produced a
# box covering only the first word, which then fell out of the pipeline and
# left the source text un-erased. At 0.4 / 0.6 the box covers the whole phrase,
# and the page's total region count only moved 57 -> 58 - so the looser
# threshold did not flood the page with spurious boxes.
#
# "paddle" selects upstream's Rust PP-OCR-family detector. It costs nothing
# extra in the image (rusty-manga-image-translator is already a requirement),
# but it is UNVERIFIED on a real page - it could not be exercised in the dev
# sandbox, whose TLS-intercepting proxy the Rust HTTP client rejects. It is
# offered as a choice, not made the default, until it has actually been run.
DEFAULT_DETECTOR = os.environ.get("DETECTOR", "default")   # "default" | "paddle"
DEFAULT_TEXT_THRESHOLD = float(os.environ.get("DETECT_TEXT_THRESHOLD", "0.4"))
DEFAULT_BOX_THRESHOLD = float(os.environ.get("DETECT_BOX_THRESHOLD", "0.6"))

# --------------------------------------------------------------------------- #
# Text erasure (the "stain" left behind after inpainting)
# --------------------------------------------------------------------------- #
# How far the erase mask is grown past the detected glyphs before inpainting
# runs. Undersized masks leave a rim of the original ink behind, which reads as
# a smudge around an otherwise cleaned bubble; oversized ones start eating the
# bubble outline and the art around it.
#
# Both values are upstream's own defaults and are NOT changed here - there is
# no measurement yet saying a different number is better, and guessing at one
# would just move the problem. They are exposed per job instead, so the whole
# range can be compared on one build rather than one value per rebuild.
#
# The two are read from different places upstream (manga_translator.py):
#   mask_dilation_offset -> config.mask_dilation_offset, per call
#   kernel_size          -> self.kernel_size, set on the instance in __init__
# so kernel_size has to be assigned onto the translator per job, like the font.
DEFAULT_MASK_DILATION = int(os.environ.get("ERASE_MASK_DILATION", "20"))
DEFAULT_ERASE_KERNEL = int(os.environ.get("ERASE_KERNEL_SIZE", "3"))

# --------------------------------------------------------------------------- #
# Rendered text size
# --------------------------------------------------------------------------- #
# Upstream picks a font size per region from that region's own geometry, then
# floors it at font_size_minimum. At -1 that floor is auto: (height+width)/200,
# which on a 1280x1816 page is about 15px - small enough that a cramped region
# renders as unreadable mush with words broken mid-way.
#
# Whether a region IS cramped depends on the detector. Measured on the same
# page: DBNet produced 13 regions, paddle 16. Paddle finds more text (it is the
# one that catches the small "Her ass" bubble DBNet drops) but its regions are
# tighter, so more of them hit the floor. Raising the floor is what makes
# paddle's output readable.
#
# -1 keeps upstream's auto behaviour and stays the default; the UI offers
# explicit floors so the two detectors can be compared fairly.
DEFAULT_FONT_SIZE_MIN = int(os.environ.get("RENDER_FONT_SIZE_MIN", "-1"))

# Inpainting is by far the most VRAM-hungry stage, and it scales with the
# square of the working resolution. Measured on a T4 (15 GiB, ~5.5 GiB of it
# already held by llama.cpp): lama_large at 1280x1816 tried to allocate
# 13.58 GiB and died with CUDA OOM, while 704x504 was fine. Upstream's default
# of 2048 therefore means "no downscaling" for a normal manga page and blows up.
# 1024 on the long side lands around 4 GiB, which fits with room to spare.
DEFAULT_INPAINT_SIZE = int(os.environ.get("INPAINT_SIZE", "1024"))
# Tried in order when the GPU still runs out of memory.
INPAINT_FALLBACKS = [1024, 768, 512]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_UPLOAD_FILES = int(os.environ.get("MAX_UPLOAD_FILES", "300"))

WORK_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Import manga-image-translator
# --------------------------------------------------------------------------- #
if str(MIT_ROOT) not in sys.path:
    sys.path.insert(0, str(MIT_ROOT))

MIT_IMPORT_ERROR: Optional[str] = None
try:
    from PIL import Image

    from manga_translator import Config, MangaTranslator  # type: ignore
    from manga_translator.config import (  # type: ignore
        Detector,
        DetectorConfig,
        InpaintPrecision,
        Inpainter,
        InpainterConfig,
        Ocr,
        OcrConfig,
        RenderConfig,
        Translator,
        TranslatorConfig,
    )
except Exception as exc:  # pragma: no cover - only hit on a broken image
    MIT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    log.error("failed to import manga-image-translator: %s", MIT_IMPORT_ERROR)
    log.error(traceback.format_exc())


# --------------------------------------------------------------------------- #
# Ordering helpers
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple:
    """
    Sort key that orders embedded numbers numerically: 1, 2, 3, ..., 10, 11
    instead of the lexicographic 1, 10, 11, 2.

    Manga pages are typically named "1.jpg", "2.jpg", ... so reading order has
    to follow the number, not the string.
    """
    stem = Path(name).stem
    parts = _NUM_RE.split(stem)
    key: List[Any] = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part), ""))
        elif part:
            key.append((0, 0, part.lower()))
    return tuple(key) or ((0, 0, stem.lower()),)


def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_SUFFIXES


def safe_member_name(name: str) -> Optional[str]:
    """
    Flatten a zip entry to a bare filename and reject anything that tries to
    escape the extraction directory (absolute paths, '..', hidden macOS junk).
    """
    name = name.replace("\\", "/")
    if name.endswith("/"):
        return None
    base = os.path.basename(name)
    if not base or base.startswith(".") or "__MACOSX" in name:
        return None
    if os.path.isabs(name) or ".." in Path(name).parts:
        return None
    return base


# --------------------------------------------------------------------------- #
# Job state
# --------------------------------------------------------------------------- #
@dataclass
class PageResult:
    name: str
    status: str = "pending"          # pending | running | done | error
    error: Optional[str] = None
    warning: Optional[str] = None
    regions: int = 0
    seconds: float = 0.0


@dataclass
class Job:
    id: str
    created: float = field(default_factory=time.time)
    status: str = "queued"           # queued | running | done | error | cancelled
    stage: str = ""
    message: str = ""
    error: Optional[str] = None
    pages: List[PageResult] = field(default_factory=list)
    done_count: int = 0
    debug: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return WORK_DIR / self.id

    @property
    def in_dir(self) -> Path:
        return self.dir / "input"

    @property
    def out_dir(self) -> Path:
        return self.dir / "output"

    def to_dict(self) -> Dict[str, Any]:
        total = len(self.pages)
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "total": total,
            "done": self.done_count,
            "percent": int(self.done_count * 100 / total) if total else 0,
            "debug": self.debug,
            # Reported back so a test run can be checked against what was
            # actually used, rather than what was requested.
            "font": self.options.get("font_resolved") or self.options.get("font"),
            "detector": self.options.get("detector"),
            "erase": "%s/%s" % (self.options.get("mask_dilation_offset"),
                                self.options.get("erase_kernel_size")),
            "font_min": self.options.get("font_size_minimum"),
            "pages": [
                {
                    "index": i,
                    "name": p.name,
                    "status": p.status,
                    "error": p.error,
                    "warning": p.warning,
                    "regions": p.regions,
                    "seconds": round(p.seconds, 1),
                }
                for i, p in enumerate(self.pages)
            ],
        }


JOBS: Dict[str, Job] = {}
JOB_LOCK = asyncio.Lock()   # one translation at a time: the GPU is shared


# --------------------------------------------------------------------------- #
# Translator
# --------------------------------------------------------------------------- #
_translator: Optional["MangaTranslator"] = None
_translator_lock = asyncio.Lock()


def _using_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


async def get_translator() -> "MangaTranslator":
    """
    Build the MangaTranslator once and reuse it, so detection/OCR/inpainting
    weights stay resident instead of being reloaded per page.
    """
    global _translator
    if _translator is not None:
        return _translator
    async with _translator_lock:
        if _translator is not None:
            return _translator
        if MIT_IMPORT_ERROR:
            raise RuntimeError(f"manga-image-translator is unavailable: {MIT_IMPORT_ERROR}")

        use_gpu = _using_gpu()
        font = resolve_font(default_font_key())
        params = {
            # kernel_size is read with int(params.get('kernel_size')) upstream,
            # with no default, so it must always be supplied.
            "kernel_size": 3,
            "use_gpu": use_gpu,
            "verbose": False,
            # Deliberately False. With True, upstream turns a failed inpainting
            # into a silently un-erased page instead of an error, which is how a
            # GPU OOM ended up looking like a successful but unreadable result.
            "ignore_errors": False,
            "font_path": str(font) if font else None,
            "models_ttl": 0,
            "batch_size": 1,
        }
        log.info("initialising MangaTranslator (gpu=%s, font=%s)", use_gpu, params["font_path"])
        _translator = MangaTranslator(params)
        return _translator


def _clear_glyph_cache() -> None:
    """
    Drop upstream's memoised glyph bitmaps so the next page renders with the
    font that is actually selected. Guarded: if upstream ever renames or drops
    the cache, rendering must keep working rather than the job dying here.
    """
    try:
        from manga_translator.rendering import text_render  # noqa: PLC0415

        text_render.get_char_glyph.cache_clear()
    except Exception as exc:  # pragma: no cover - depends on upstream internals
        log.warning("could not clear the glyph cache, a font switch may not "
                    "take effect on this page: %s", exc)


def _odd(value: Any, default: int, low: int, high: int) -> int:
    """
    Clamp to [low, high] and force odd.

    cv2.getStructuringElement takes even sizes but grows the mask asymmetrically
    for them, which would shift the erased area by half a pixel per dilation
    instead of widening it evenly.
    """
    try:
        n = min(high, max(low, int(value)))
    except (TypeError, ValueError):
        return default
    if n % 2 == 0:
        # Step up to the next odd, or down if that would leave the range.
        n = n - 1 if n + 1 > high else n + 1
    return max(n, 1)


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return min(high, max(low, int(value)))
    except (TypeError, ValueError):
        return default


def _threshold(value: Any, default: float) -> float:
    """
    A detection threshold is a probability; anything outside (0, 1) silently
    turns detection into all-or-nothing, so a bad form value is clamped rather
    than passed through to produce a mysteriously empty or noisy page.
    """
    try:
        return min(0.95, max(0.05, float(value)))
    except (TypeError, ValueError):
        return default


def build_config(options: Dict[str, Any], inpainting_size: Optional[int] = None) -> "Config":
    """Translate UI options into an upstream Config object."""
    ocr_choice = Ocr.mocr if options.get("ocr") == "mocr" else Ocr.ocr48px
    inpainter = Inpainter.lama_large if options.get("inpainter", "lama_large") == "lama_large" else Inpainter.none

    render = RenderConfig()
    # Manga is right-to-left, but Turkish output reads left-to-right, so region
    # ordering should not be flipped.
    render.rtl = False
    if options.get("font_size_offset"):
        render.font_size_offset = int(options["font_size_offset"])
    # -1 means "upstream decides"; anything else is an explicit pixel floor.
    font_min = options.get("font_size_minimum", DEFAULT_FONT_SIZE_MIN)
    render.font_size_minimum = (
        -1 if int(font_min) < 0 else _clamp_int(font_min, DEFAULT_FONT_SIZE_MIN, 8, 80)
    )

    translator_cfg = TranslatorConfig(
        translator=Translator.custom_openai,
        target_lang=options.get("target_lang", "TRK"),
        gpt_config=str(GPT_CONFIG) if GPT_CONFIG.exists() else None,
    )
    if options.get("source_lang") and options["source_lang"] != "auto":
        translator_cfg.skip_lang = None

    return Config(
        translator=translator_cfg,
        ocr=OcrConfig(ocr=ocr_choice),
        detector=DetectorConfig(
            detector=Detector.paddle if options.get("detector") == "paddle" else Detector.default,
            detection_size=int(options.get("detection_size", 2048)),
            # Both detectors take these through the same interface
            # (detection/__init__.py: dispatch -> detector.detect(...)).
            text_threshold=_threshold(options.get("text_threshold"), DEFAULT_TEXT_THRESHOLD),
            box_threshold=_threshold(options.get("box_threshold"), DEFAULT_BOX_THRESHOLD),
        ),
        inpainter=InpainterConfig(
            inpainter=inpainter,
            inpainting_size=int(inpainting_size or options.get("inpainting_size") or DEFAULT_INPAINT_SIZE),
            # T4 is Turing and has no native bf16; fp16 is both supported and
            # half the activation memory of fp32.
            inpainting_precision=InpaintPrecision.fp16,
        ),
        render=render,
        # Upstream reads the mask dilation from the config...
        mask_dilation_offset=_clamp_int(
            options.get("mask_dilation_offset"), DEFAULT_MASK_DILATION, 0, 80),
        # ...but NOT the kernel size, which it takes from the instance. Set here
        # anyway so the two never disagree if upstream fixes that.
        kernel_size=_odd(options.get("erase_kernel_size"), DEFAULT_ERASE_KERNEL, 1, 15),
    )


# --------------------------------------------------------------------------- #
# Job execution
# --------------------------------------------------------------------------- #
def _is_oom(exc: BaseException) -> bool:
    """CUDA OOM arrives as torch.OutOfMemoryError, or a RuntimeError saying so."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def _free_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


async def translate_page(translator: "MangaTranslator", image, page: PageResult,
                         options: Dict[str, Any]):
    """
    Run one page, stepping the inpainting resolution down if the GPU runs out
    of memory.

    This exists because of how upstream reacts to an inpainting failure: with
    ignore_errors set it does `ctx.img_inpainted = ctx.img_rgb`, i.e. it falls
    back to the ORIGINAL image and then renders the translation on top of text
    that was never erased. The job reports success while the page is actually
    unreadable. So errors are raised here and handled explicitly instead.

    Inpainting memory grows with the square of the working resolution, so a
    smaller size is a real fix rather than a coin flip.
    """
    requested = int(options.get("inpainting_size") or DEFAULT_INPAINT_SIZE)
    sizes = [requested] + [s for s in INPAINT_FALLBACKS if s < requested]
    last: Optional[BaseException] = None

    for attempt, size in enumerate(sizes):
        cfg = build_config(options, inpainting_size=size)
        try:
            ctx = await translator.translate(image.copy(), cfg, image_name=page.name)
            if attempt:
                page.warning = (
                    f"GPU belleği yetmediği icin metin silme {size}px'e dusuruldu."
                )
            return ctx
        except Exception as exc:
            last = exc
            if not _is_oom(exc):
                raise
            log.warning("page %s: CUDA OOM at inpainting_size=%s, retrying smaller",
                        page.name, size)
            _free_vram()

    raise RuntimeError(
        "GPU bellegi yetmedi: metin silme en dusuk cozunurlukte bile basarisiz oldu."
    ) from last


async def run_job(job: Job) -> None:
    async with JOB_LOCK:
        job.status = "running"
        job.stage = "loading"
        job.message = "Modeller hazırlanıyor…"
        try:
            translator = await get_translator()

            # Font is picked per job. MangaTranslator reads self.font_path at
            # render time (manga_translator.py: _run_text_rendering hands it
            # straight to dispatch_rendering), so assigning it here is enough -
            # the translator does not need rebuilding. JOB_LOCK serialises jobs,
            # so nothing else can be mid-render while this changes.
            font = resolve_font(job.options.get("font"))
            if font:
                translator.font_path = str(font)
            job.options["font_resolved"] = font.name if font else None

            # Upstream memoises rendered glyphs with
            #   @functools.lru_cache
            #   def get_char_glyph(cdpt, font_size, direction)
            # and resolves the face from a module-level FONT_SELECTION that
            # set_font() swaps. The font is NOT part of that cache key, so once
            # a character has been drawn at a given size the cached bitmap is
            # reused no matter which font is selected afterwards - a page
            # rendered after a font switch comes out in the PREVIOUS font.
            #
            # Reproduced directly against the real faces: 'Ü' at 30px returned
            # byte-identical bitmaps for anime_ace and CCWildWords while the
            # cache was warm, and two different bitmaps once it was cleared.
            #
            # Per-job font selection is ours, so clearing this is ours too.
            _clear_glyph_cache()

            # Mask refinement reads self.kernel_size, not the config
            # (manga_translator.py: _run_mask_refinement), so the per-job value
            # has to be assigned here the same way the font is.
            kernel = _odd(job.options.get("erase_kernel_size"), DEFAULT_ERASE_KERNEL, 1, 15)
            translator.kernel_size = kernel

            log.info("job %s: font=%s detector=%s erase=%s/%s font_min=%s", job.id,
                     font.name if font else "<none>", job.options.get("detector"),
                     job.options.get("mask_dilation_offset"), kernel,
                     job.options.get("font_size_minimum"))

            hook_state = {"stage": ""}

            async def progress_hook(state: str, finished: bool) -> None:
                hook_state["stage"] = state
                job.stage = state

            translator.add_progress_hook(progress_hook)

            job.out_dir.mkdir(parents=True, exist_ok=True)

            for index, page in enumerate(job.pages):
                page.status = "running"
                job.message = f"{page.name} çevriliyor ({index + 1}/{len(job.pages)})"
                started = time.time()
                try:
                    src = job.in_dir / page.name
                    image = Image.open(src)
                    image.load()
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")

                    ctx = await translate_page(translator, image, page, job.options)

                    result = getattr(ctx, "result", None)
                    if result is None:
                        raise RuntimeError("çeviri sonucu boş döndü")

                    # Keep the original filename exactly as uploaded, so a
                    # gallery upload of 1.jpg/2.jpg comes back as 1.png/2.png in
                    # the same reading order.
                    out_path = job.out_dir / f"{Path(page.name).stem}.png"
                    if result.mode not in ("RGB", "RGBA"):
                        result = result.convert("RGB")
                    result.save(out_path, format="PNG")

                    page.regions = len(getattr(ctx, "text_regions", []) or [])
                    page.status = "done"
                except Exception as exc:
                    page.status = "error"
                    page.error = f"{type(exc).__name__}: {exc}"
                    log.error("page %s failed: %s", page.name, page.error)
                    if job.debug:
                        log.error(traceback.format_exc())
                finally:
                    page.seconds = time.time() - started
                    job.done_count = index + 1
                    # Release whatever the vision models held before the next page.
                    _free_vram()

            ok = sum(1 for p in job.pages if p.status == "done")
            failed = sum(1 for p in job.pages if p.status == "error")
            job.status = "done"
            job.stage = "finished"
            if failed and not ok:
                job.status = "error"
                job.error = "Hiçbir sayfa çevrilemedi. Ayrıntılar için sayfa listesine bakın."
                job.message = job.error
            elif failed:
                job.message = f"{ok} sayfa çevrildi, {failed} sayfa başarısız."
            else:
                job.message = f"{ok} sayfa başarıyla çevrildi."
        except Exception as exc:
            job.status = "error"
            job.stage = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
            log.error("job %s failed: %s", job.id, job.error)
            log.error(traceback.format_exc())


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Manga Çeviri — Hy-MT2", version="1.0.0")
STATIC_DIR = APP_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>UI dosyası bulunamadı</h1>", status_code=500)

    version = _asset_version()
    html = page.read_text(encoding="utf-8")
    for asset in ("app.js", "style.css"):
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={version}")
    # The page carries the asset URLs, so it must never itself be a stale copy.
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


async def llama_status() -> Dict[str, Any]:
    """
    Is the translation backend up yet?

    The web app now starts in parallel with llama-server's model load
    (scripts/entrypoint.sh), so for the first minutes of a cold start the UI is
    reachable while the backend is not. Everything that needs the backend asks
    here rather than assuming it is there.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health")
            return {"reachable": resp.status_code == 200, "status_code": resp.status_code}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe: reports the app, the model file and the llama backend."""
    model_path = Path(os.environ.get("HY_MT2_MODEL_PATH", "/opt/models/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf"))
    llama = await llama_status()

    body = {
        "status": "ok" if (llama.get("reachable") and not MIT_IMPORT_ERROR) else "degraded",
        "manga_image_translator": "ok" if not MIT_IMPORT_ERROR else MIT_IMPORT_ERROR,
        "model_file": {"path": str(model_path), "present": model_path.exists()},
        "llama_server": llama,
        "gpu": _using_gpu(),
        "jobs": len(JOBS),
        # Which commit this image was built from - see BUILD_INFO above.
        "build": BUILD_INFO,
    }
    return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)


@app.get("/api/fonts")
async def fonts() -> JSONResponse:
    """
    The font picker's contents. Driven entirely by what is in assets/fonts/,
    so the UI never needs editing when a font is added to the repo.
    """
    current = default_font_key()
    return JSONResponse({
        "default": current,
        "fonts": [
            {
                "key": f["key"],
                "label": f["label"],
                "file": f["file"],
                "source": f["source"],
                "turkish": f["turkish"],
                "missing_glyphs": f["missing_glyphs"],
            }
            for f in list_fonts().values()
        ],
    })


@app.get("/api/fonts/preview/{key}")
async def font_preview(key: str) -> Response:
    """
    Render a Turkish sample with one font, as a PNG.

    A font can carry a glyph for every Turkish letter, have each of those
    glyphs be unique, and still draw the wrong shapes - a patched face in
    assets/fonts/ turned out to draw ü as b, ç as 3 and Ö as U while passing
    every presence and uniqueness check thrown at it. Nothing short of looking
    at the output catches that, so the UI shows the output.

    Costs no GPU and no translation: a bad font is visible before a page is
    ever run through the pipeline.
    """
    import io as _io

    import freetype  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    path = resolve_font(key)
    if path is None:
        raise HTTPException(404, "Yazı tipi bulunamadı.")

    sample = "ÖZÜR DİLERİM · kaç saç ğüşıöç · ÇĞİÖŞÜ"
    size, pad = 34, 8
    try:
        face = freetype.Face(path.open("rb"))
        face.set_pixel_sizes(0, size)

        width = pad * 2
        for ch in sample:
            face.load_char(ch)
            width += face.glyph.advance.x >> 6
        height = int(size * 1.9)
        canvas = np.zeros((height, max(width, 1)), np.uint8)

        x, baseline = pad, int(size * 1.35)
        for ch in sample:
            face.load_char(ch)
            g = face.glyph
            bm = g.bitmap
            if bm.rows and bm.width:
                y0, x0 = baseline - g.bitmap_top, x + g.bitmap_left
                y1, x1 = y0 + bm.rows, x0 + bm.width
                if 0 <= y0 and 0 <= x0 and y1 <= height and x1 <= canvas.shape[1]:
                    patch = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.width)
                    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], patch)
            x += g.advance.x >> 6

        buf = _io.BytesIO()
        Image.fromarray(255 - canvas).save(buf, format="PNG")
    except Exception as exc:
        log.warning("font preview failed for %s: %s", path.name, exc)
        raise HTTPException(500, f"Önizleme oluşturulamadı: {exc}")

    return Response(buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/jobs")
async def create_job(
    files: List[UploadFile] = File(default=[]),
    target_lang: str = Form("TRK"),
    font: str = Form(""),
    ocr: str = Form("48px"),
    detector: str = Form(DEFAULT_DETECTOR),
    inpainter: str = Form("lama_large"),
    detection_size: int = Form(2048),
    inpainting_size: int = Form(DEFAULT_INPAINT_SIZE),
    text_threshold: float = Form(DEFAULT_TEXT_THRESHOLD),
    box_threshold: float = Form(DEFAULT_BOX_THRESHOLD),
    mask_dilation_offset: int = Form(DEFAULT_MASK_DILATION),
    erase_kernel_size: int = Form(DEFAULT_ERASE_KERNEL),
    font_size_offset: int = Form(0),
    font_size_minimum: int = Form(DEFAULT_FONT_SIZE_MIN),
    debug: bool = Form(False),
) -> JSONResponse:
    if MIT_IMPORT_ERROR:
        raise HTTPException(500, f"Çeviri motoru yüklenemedi: {MIT_IMPORT_ERROR}")
    if not files:
        raise HTTPException(400, "Hiç dosya yüklenmedi.")

    # The UI is up before llama-server finishes loading its model, so a job
    # submitted in that window would fail deep inside the pipeline with a
    # connection error on a half-processed page. Refuse it up front instead.
    llama = await llama_status()
    if not llama.get("reachable"):
        raise HTTPException(
            503,
            "Çeviri modeli hâlâ yükleniyor. Sayfanın üstündeki durum "
            "göstergesi yeşile döndüğünde tekrar deneyin.",
        )

    job = Job(id=uuid.uuid4().hex[:12])
    job.debug = debug
    job.options = {
        "target_lang": target_lang,
        "font": font or default_font_key(),
        "ocr": ocr,
        "detector": detector,
        "inpainter": inpainter,
        "detection_size": detection_size,
        "inpainting_size": inpainting_size,
        "text_threshold": text_threshold,
        "box_threshold": box_threshold,
        "mask_dilation_offset": mask_dilation_offset,
        "erase_kernel_size": erase_kernel_size,
        "font_size_offset": font_size_offset,
        "font_size_minimum": font_size_minimum,
    }
    job.in_dir.mkdir(parents=True, exist_ok=True)

    names: List[str] = []
    try:
        for upload in files:
            raw = await upload.read()
            filename = os.path.basename(upload.filename or "")
            if not filename:
                continue

            if filename.lower().endswith(".zip"):
                # ZIP upload: take every image inside, keeping its own filename.
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        for info in zf.infolist():
                            member = safe_member_name(info.filename)
                            if not member or not is_image(member):
                                continue
                            target = job.in_dir / member
                            if target.exists():
                                target = job.in_dir / f"{target.stem}_{len(names)}{target.suffix}"
                            with zf.open(info) as srcf, open(target, "wb") as dstf:
                                shutil.copyfileobj(srcf, dstf)
                            names.append(target.name)
                except zipfile.BadZipFile:
                    raise HTTPException(400, f"Bozuk ZIP dosyası: {filename}")
            elif is_image(filename):
                target = job.in_dir / filename
                if target.exists():
                    target = job.in_dir / f"{target.stem}_{len(names)}{target.suffix}"
                target.write_bytes(raw)
                names.append(target.name)

            if len(names) > MAX_UPLOAD_FILES:
                raise HTTPException(400, f"Çok fazla görsel (en fazla {MAX_UPLOAD_FILES}).")
    except HTTPException:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise

    if not names:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(400, "Yüklenen dosyalarda çevrilebilir görsel bulunamadı.")

    # Reading order follows the page number, not the string.
    names.sort(key=natural_key)
    job.pages = [PageResult(name=n) for n in names]

    JOBS[job.id] = job
    asyncio.create_task(run_job(job))
    log.info("job %s queued with %d page(s)", job.id, len(job.pages))
    return JSONResponse(job.to_dict(), status_code=202)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "İş bulunamadı.")
    return JSONResponse(job.to_dict())


@app.get("/api/jobs/{job_id}/page/{index}")
async def job_page(job_id: str, index: int) -> Response:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "İş bulunamadı.")
    if index < 0 or index >= len(job.pages):
        raise HTTPException(404, "Sayfa bulunamadı.")
    page = job.pages[index]
    path = job.out_dir / f"{Path(page.name).stem}.png"
    if not path.exists():
        raise HTTPException(404, "Bu sayfanın sonucu henüz hazır değil.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/original/{index}")
async def job_original(job_id: str, index: int) -> Response:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "İş bulunamadı.")
    if index < 0 or index >= len(job.pages):
        raise HTTPException(404, "Sayfa bulunamadı.")
    path = job.in_dir / job.pages[index].name
    if not path.exists():
        raise HTTPException(404, "Kaynak görsel bulunamadı.")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/download")
async def job_download(job_id: str) -> Response:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "İş bulunamadı.")
    done = [p for p in job.pages if p.status == "done"]
    if not done:
        raise HTTPException(409, "İndirilecek çevrilmiş sayfa yok.")

    archive = job.dir / f"manga-ceviri-{job.id}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for page in sorted(done, key=lambda p: natural_key(p.name)):
            src = job.out_dir / f"{Path(page.name).stem}.png"
            if src.exists():
                # Original stem is preserved, so page order survives the round trip.
                zf.write(src, arcname=f"{Path(page.name).stem}.png")
    return FileResponse(archive, media_type="application/zip", filename=archive.name)


@app.delete("/api/jobs/{job_id}")
async def job_delete(job_id: str) -> JSONResponse:
    job = JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(404, "İş bulunamadı.")
    shutil.rmtree(job.dir, ignore_errors=True)
    return JSONResponse({"deleted": job_id})


@app.get("/api/debug/llama")
async def debug_llama() -> JSONResponse:
    """Debug view: last lines of the llama-server log plus its reported props."""
    import httpx

    out: Dict[str, Any] = {}
    logfile = Path("/tmp/llama-server.log")
    if logfile.exists():
        lines = logfile.read_text(errors="replace").splitlines()
        out["log_tail"] = lines[-60:]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            out["props"] = (await client.get(f"http://{LLAMA_HOST}:{LLAMA_PORT}/props")).json()
    except Exception as exc:
        out["props_error"] = str(exc)
    return JSONResponse(out)
