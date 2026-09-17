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

LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8081"))

# anime_ace*.ttf, upstream's default manga fonts, have no glyphs for the Turkish
# letters g-breve, dotless i, dotted I or s-cedilla; Turkish text rendered with
# them comes out visibly broken. "comic shanns 2" keeps the comic look and does
# cover them. Verified with fontTools against the actual font files.
DEFAULT_FONT = os.environ.get("RENDER_FONT", "comic shanns 2.ttf")

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
            "pages": [
                {
                    "index": i,
                    "name": p.name,
                    "status": p.status,
                    "error": p.error,
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
        font = MIT_ROOT / "fonts" / DEFAULT_FONT
        params = {
            # kernel_size is read with int(params.get('kernel_size')) upstream,
            # with no default, so it must always be supplied.
            "kernel_size": 3,
            "use_gpu": use_gpu,
            "verbose": False,
            "ignore_errors": True,
            "font_path": str(font) if font.exists() else None,
            "models_ttl": 0,
            "batch_size": 1,
        }
        log.info("initialising MangaTranslator (gpu=%s, font=%s)", use_gpu, params["font_path"])
        _translator = MangaTranslator(params)
        return _translator


def build_config(options: Dict[str, Any]) -> "Config":
    """Translate UI options into an upstream Config object."""
    ocr_choice = Ocr.mocr if options.get("ocr") == "mocr" else Ocr.ocr48px
    inpainter = Inpainter.lama_large if options.get("inpainter", "lama_large") == "lama_large" else Inpainter.none

    render = RenderConfig()
    # Manga is right-to-left, but Turkish output reads left-to-right, so region
    # ordering should not be flipped.
    render.rtl = False
    if options.get("font_size_offset"):
        render.font_size_offset = int(options["font_size_offset"])

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
            detector=Detector.default,
            detection_size=int(options.get("detection_size", 2048)),
        ),
        inpainter=InpainterConfig(inpainter=inpainter),
        render=render,
        kernel_size=3,
    )


# --------------------------------------------------------------------------- #
# Job execution
# --------------------------------------------------------------------------- #
async def run_job(job: Job) -> None:
    async with JOB_LOCK:
        job.status = "running"
        job.stage = "loading"
        job.message = "Modeller hazırlanıyor…"
        try:
            translator = await get_translator()
            config = build_config(job.options)

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

                    ctx = await translator.translate(image, config, image_name=page.name)

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
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe: reports the app, the model file and the llama backend."""
    import httpx

    model_path = Path(os.environ.get("HY_MT2_MODEL_PATH", "/opt/models/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf"))
    llama: Dict[str, Any] = {"reachable": False}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health")
            llama = {"reachable": resp.status_code == 200, "status_code": resp.status_code}
    except Exception as exc:
        llama = {"reachable": False, "error": str(exc)}

    body = {
        "status": "ok" if (llama.get("reachable") and not MIT_IMPORT_ERROR) else "degraded",
        "manga_image_translator": "ok" if not MIT_IMPORT_ERROR else MIT_IMPORT_ERROR,
        "model_file": {"path": str(model_path), "present": model_path.exists()},
        "llama_server": llama,
        "gpu": _using_gpu(),
        "jobs": len(JOBS),
    }
    return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)


@app.post("/api/jobs")
async def create_job(
    files: List[UploadFile] = File(default=[]),
    target_lang: str = Form("TRK"),
    ocr: str = Form("48px"),
    inpainter: str = Form("lama_large"),
    detection_size: int = Form(2048),
    font_size_offset: int = Form(0),
    debug: bool = Form(False),
) -> JSONResponse:
    if MIT_IMPORT_ERROR:
        raise HTTPException(500, f"Çeviri motoru yüklenemedi: {MIT_IMPORT_ERROR}")
    if not files:
        raise HTTPException(400, "Hiç dosya yüklenmedi.")

    job = Job(id=uuid.uuid4().hex[:12])
    job.debug = debug
    job.options = {
        "target_lang": target_lang,
        "ocr": ocr,
        "inpainter": inpainter,
        "detection_size": detection_size,
        "font_size_offset": font_size_offset,
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
