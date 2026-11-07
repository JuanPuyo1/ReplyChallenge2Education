from __future__ import annotations

import io
import os
from itertools import cycle

from agents.llm_client import (
    DEFAULT_GROQ_STT_MODEL,
    get_groq_client,
    is_groq_configured,
)

DEMO_SEGMENTS = cycle(
    [
        "The Nyquist contour must detour around poles on the imaginary axis and epsilon tends to zero.",
        "For G of s H of s, count clockwise encirclements of minus one and use N equals Z minus P.",
        "At the gain crossover omega c, phase margin measures relative stability and damping.",
    ]
)


def transcribe_audio(blob: bytes, filename: str = "lecture.webm") -> str:
    if is_groq_configured():
        client = get_groq_client()
        audio = io.BytesIO(blob)
        audio.name = filename
        try:
            response = client.audio.transcriptions.create(
                model=os.getenv("GROQ_STT_MODEL", DEFAULT_GROQ_STT_MODEL),
                file=audio,
                response_format="text",
            )
            return str(response).strip()
        except Exception:
            pass
    return next(DEMO_SEGMENTS)


def process_audio_chunk(
    *,
    lecture_id: int,
    blob: bytes,
    seq: int,
    start_ms: int,
    duration_ms: int,
) -> dict:
    from agents.listener_agent import process_transcript_chunk

    raw_text = transcribe_audio(blob)
    return process_transcript_chunk(
        lecture_id=lecture_id,
        raw_text=raw_text,
        seq=seq,
        start_ms=start_ms,
        duration_ms=duration_ms,
    )
