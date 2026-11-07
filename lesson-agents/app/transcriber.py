"""Agent 1, step 1: one standalone audio file in, clean text out (Groq Whisper)."""
from __future__ import annotations

import re

from . import config
from .llm import client

# Whisper invents these on silence / noise, especially at chunk edges.
_HALLUCINATIONS = re.compile(
    r"(thanks? (you )?for watching|subscribe to|sottotitoli (creati|a cura) |"
    r"amara\.org|grazie per la visione|www\.\S+\.(com|it))", re.I)


def build_prompt(title: str, topics: list[dict], tail: str) -> str:
    """Whisper's `prompt` (max ~224 tokens) steers spelling of jargon and gives
    continuity across chunk boundaries. Vocabulary first, then the previous words."""
    vocab = f"{title}. " + ", ".join(t["name"] for t in topics) + ". "
    return (vocab[:300] + tail)[-700:]


def _keep(seg: dict) -> bool:
    if seg.get("no_speech_prob", 0) > 0.6 and seg.get("avg_logprob", 0) < -1.0:
        return False                        # mostly silence
    if seg.get("compression_ratio", 0) > 2.4:
        return False                        # looping / repeated text
    return not _HALLUCINATIONS.search(seg.get("text", ""))


def transcribe(audio: bytes, filename: str, prompt: str = "", language: str | None = None) -> str:
    kwargs = dict(file=(filename, audio), model=config.STT_MODEL,
                  response_format="verbose_json", temperature=0.0)
    if prompt:
        kwargs["prompt"] = prompt
    lang = language or config.DEFAULT_LANGUAGE
    if lang:
        kwargs["language"] = lang
    r = client().audio.transcriptions.create(**kwargs)

    data = r.model_dump() if hasattr(r, "model_dump") else dict(r)
    segments = data.get("segments") or []
    if not segments:
        text = (data.get("text") or "").strip()
        return "" if _HALLUCINATIONS.search(text) else text
    return " ".join(s["text"].strip() for s in segments if _keep(s)).strip()
