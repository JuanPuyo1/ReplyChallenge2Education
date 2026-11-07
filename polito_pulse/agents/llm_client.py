from __future__ import annotations

import hashlib
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_VECTOR_SIZE = 1536
COLLECTIONS = ("course_slides", "past_exams", "manim_templates")
_qdrant_unavailable_until = 0.0
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"
DEFAULT_GROQ_STT_MODEL = "whisper-large-v3-turbo"
DEFAULT_GROQ_EMBEDDING_MODEL = "nomic-embed-text-v1.5"


def is_claude_configured() -> bool:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return key.startswith("sk-ant-") and "PENDING" not in key and len(key) > 12


def is_groq_configured() -> bool:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key.startswith("gsk_") and len(key) > 20


def is_openai_configured() -> bool:
    """Back-compat: live LLM path is configured when Claude chat is available."""
    return is_claude_configured()


@lru_cache(maxsize=1)
def get_anthropic_client():
    if not is_claude_configured():
        return None
    from anthropic import Anthropic

    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@lru_cache(maxsize=1)
def get_groq_client():
    if not is_groq_configured():
        return None
    from openai import OpenAI

    return OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"])


@lru_cache(maxsize=1)
def get_qdrant_client():
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url:
        return None
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=url,
        api_key=api_key or None,
        timeout=30,
        check_compatibility=False,
    )


def deterministic_embedding(text: str, size: int = DEFAULT_VECTOR_SIZE) -> list[float]:
    """Stable normalized fallback vector suitable for cosine search."""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < size:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        values.extend(((byte / 127.5) - 1.0) for byte in block)
        counter += 1
    values = values[:size]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def coerce_vector_dimension(vector: list[float], size: int) -> list[float]:
    if len(vector) < size:
        vector = vector + deterministic_embedding("dimension-padding", size - len(vector))
    vector = vector[:size]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_text(text: str, size: int = DEFAULT_VECTOR_SIZE) -> list[float]:
    client = get_groq_client()
    if client:
        try:
            response = client.embeddings.create(
                model=os.getenv("GROQ_EMBEDDING_MODEL", DEFAULT_GROQ_EMBEDDING_MODEL),
                input=text,
            )
            return coerce_vector_dimension(response.data[0].embedding, size)
        except Exception:
            pass
    return deterministic_embedding(text, size)


def _vector_size_from_config(config: Any) -> int:
    vectors = config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        return int(first.size if hasattr(first, "size") else first["size"])
    return DEFAULT_VECTOR_SIZE


def collection_vector_size(name: str) -> int:
    client = get_qdrant_client()
    if not client:
        return DEFAULT_VECTOR_SIZE
    try:
        return _vector_size_from_config(client.get_collection(name).config)
    except Exception:
        return DEFAULT_VECTOR_SIZE


def ensure_collections() -> dict[str, int]:
    client = get_qdrant_client()
    if not client:
        return {name: DEFAULT_VECTOR_SIZE for name in COLLECTIONS}

    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    existing = {item.name for item in client.get_collections().collections}
    sizes: dict[str, int] = {}
    for name in COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=DEFAULT_VECTOR_SIZE, distance=Distance.COSINE),
            )
            sizes[name] = DEFAULT_VECTOR_SIZE
        else:
            sizes[name] = collection_vector_size(name)

        index_types = {
            "lecture_id": PayloadSchemaType.INTEGER,
            "course_code": PayloadSchemaType.KEYWORD,
            "slide_number": PayloadSchemaType.INTEGER,
            "topic": PayloadSchemaType.KEYWORD,
        }
        for field, schema in index_types.items():
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            except Exception:
                # Index may already exist, or a legacy collection may reject the field.
                pass
    return sizes


def qdrant_search(
    collection: str,
    query: str,
    *,
    limit: int = 3,
    query_filter: Any = None,
) -> list[dict[str, Any]]:
    global _qdrant_unavailable_until
    if time.monotonic() < _qdrant_unavailable_until:
        return []
    client = get_qdrant_client()
    if not client:
        return []
    result_queue: Queue = Queue(maxsize=1)

    def remote_query() -> None:
        try:
            size = _vector_size_from_config(client.get_collection(collection).config)
            vector = embed_text(query, size)
            response = client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            result_queue.put(response)
        except Exception as exc:
            result_queue.put(exc)

    worker = Thread(target=remote_query, daemon=True, name=f"qdrant-{collection}")
    worker.start()
    worker.join(timeout=5)
    if worker.is_alive():
        _qdrant_unavailable_until = time.monotonic() + 60
        return []
    try:
        response = result_queue.get_nowait()
    except Empty:
        return []
    if isinstance(response, Exception):
        _qdrant_unavailable_until = time.monotonic() + 60
        return []
    return [
        {"score": float(point.score or 0), **dict(point.payload or {})}
        for point in response.points
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def chat_json(
    *,
    system: str,
    user: str,
    fallback: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    client = get_anthropic_client()
    if not client:
        return fallback
    try:
        response = client.messages.create(
            model=model or os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
            max_tokens=4096,
            temperature=0.2,
            system=f"{system}\nReturn a single JSON object only. No markdown.",
            messages=[{"role": "user", "content": user}],
        )
        content = "".join(block.text for block in response.content if getattr(block, "text", None))
        return _extract_json_object(content or "{}")
    except Exception:
        return fallback


def fallback_listener_response(raw_text: str) -> dict[str, Any]:
    lower = raw_text.lower()
    if "bode" in lower or "phase" in lower:
        return {
            "cleaned_text": (
                "At the gain crossover frequency $\\omega_c$, the phase margin is "
                "$\\varphi_m = 180^\\circ + \\angle G(j\\omega_c)H(j\\omega_c)$."
            ),
            "matched_slide_number": 24,
            "similarity_score": 0.92,
        }
    return {
        "cleaned_text": (
            "The Nyquist contour detours around poles on the imaginary axis. "
            "Count encirclements using $N = Z - P$ as $\\varepsilon \\to 0^+$."
        ),
        "matched_slide_number": 18,
        "similarity_score": 0.94,
    }
