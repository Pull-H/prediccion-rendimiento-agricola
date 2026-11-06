"""Authenticated access to the frozen article and its declared research assets."""
import hashlib
import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from .models import User
from .security import current_user

PUBLICATION_DIR = Path(os.getenv("PUBLICATION_DIR", str(Path(__file__).resolve().parents[2] / "docs" / "manuscrito")))
router = APIRouter(prefix="/api/publication", tags=["Artículo y figuras"])
MIME = {".png": "image/png", ".tiff": "image/tiff", ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv", ".md": "text/plain", ".json": "application/json",
        ".ris": "application/x-research-info-systems", ".zip": "application/zip"}


def catalogue():
    try:
        payload = json.loads((PUBLICATION_DIR / "catalogo_publicacion.json").read_text(encoding="utf-8"))
        if len(payload["figures"]) != 6 or len(payload["tables"]) != 6 or not isinstance(payload["files"], dict):
            raise ValueError("Expected six figures and six tables")
        return payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "Los materiales del artículo todavía no están disponibles") from exc


@router.get("")
def publication(user: User = Depends(current_user)):
    payload = catalogue()
    return {**payload, "files": {name: {key: value for key, value in entry.items() if key != "path"}
                                 for name, entry in payload["files"].items()}}


@router.get("/files/{filename}")
def publication_file(filename: str, download: bool = False, user: User = Depends(current_user)):
    payload = catalogue()
    entry = payload["files"].get(filename)
    if entry is None:
        raise HTTPException(404, "Archivo del artículo no encontrado")
    root = PUBLICATION_DIR.resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix not in MIME:
        raise HTTPException(404, "Archivo del artículo no disponible")
    if path.stat().st_size != entry["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise HTTPException(409, "El archivo cambió desde la generación del catálogo; vuelva a generar los materiales")
    return FileResponse(path, media_type=MIME[path.suffix], filename=filename,
                        content_disposition_type="attachment" if download else "inline")
