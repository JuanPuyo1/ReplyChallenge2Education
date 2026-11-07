from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from agents.llm_client import (
    collection_vector_size,
    embed_text,
    ensure_collections,
    get_qdrant_client,
)

LATEX_PATTERN = re.compile(
    r"(\$[^$]+\$|\\(?:alpha|beta|gamma|omega|varepsilon|varphi|theta|lambda)\b|"
    r"\b[A-Z]\s*=\s*[A-Z]\s*[-+]\s*[A-Z]\b)"
)


def extract_pages(pdf_path: str | Path) -> list[str]:
    path = str(pdf_path)
    try:
        import pymupdf4llm

        import fitz

        page_count = fitz.open(path).page_count
        return [
            pymupdf4llm.to_markdown(path, pages=[page_number]).strip()
            for page_number in range(page_count)
        ]
    except Exception:
        import fitz

        with fitz.open(path) as document:
            return [page.get_text("text").strip() for page in document]


def _payload(
    *,
    text: str,
    page_number: int,
    lecture_id: int,
    course_code: str,
    topic: str,
    kind: str,
) -> dict[str, Any]:
    symbols = [match.strip("$") for match in LATEX_PATTERN.findall(text)]
    payload = {
        "text": text,
        "lecture_id": lecture_id,
        "course_code": course_code,
        "slide_number": page_number,
        "topic": topic or f"Slide {page_number}",
        "latex_symbols": symbols,
    }
    if kind == "past_exams":
        payload.update(
            {
                "question": text[:800],
                "point_weight": 8,
                "year": 2025,
                "question_number": page_number,
            }
        )
    return payload


def index_pdf(
    pdf_path: str | Path,
    *,
    collection: str,
    lecture_id: int,
    course_code: str,
    topic: str = "",
) -> list[dict[str, Any]]:
    if collection not in {"course_slides", "past_exams"}:
        raise ValueError("collection must be course_slides or past_exams")
    pages = extract_pages(pdf_path)
    ensure_collections()
    client = get_qdrant_client()
    indexed: list[dict[str, Any]] = []
    if not client:
        return indexed

    from qdrant_client.models import PointStruct

    vector_size = collection_vector_size(collection)
    points: list[PointStruct] = []
    for page_number, text in enumerate(pages, start=1):
        if not text:
            continue
        payload = _payload(
            text=text,
            page_number=page_number,
            lecture_id=lecture_id,
            course_code=course_code,
            topic=topic,
            kind=collection,
        )
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"polito:{collection}:{course_code}:{lecture_id}:{page_number}",
            )
        )
        points.append(
            PointStruct(id=point_id, vector=embed_text(text, vector_size), payload=payload)
        )
        indexed.append({"point_id": point_id, **payload})
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)
    return indexed
