"""Agent 1: live transcript alignment, math repair, and slide telemetry.

This preserves the original Listener's literal extraction contract, window gating,
silence handling, and invented-ID validation while enriching it with Qdrant slide matches.
"""
from __future__ import annotations

import json
from typing import Any

from django.db.models import F
from pydantic import BaseModel, Field

from agents.llm_client import chat_json, fallback_listener_response

LISTENER_WINDOW_SECONDS = 30


class ListenerOutput(BaseModel):
    topic_id: int | None = None
    off_plan_topic: str | None = None
    summary: str
    key_concepts: list[str] = Field(default_factory=list)
    examples_given: list[str] = Field(default_factory=list)
    student_questions: list[str] = Field(default_factory=list)
    clarity_signals: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    cleaned_text: str = ""
    matched_slide_number: int | None = None
    similarity_score: float = Field(default=0.0, ge=0, le=1)
    time_spent_seconds: int = 0
    quick_questions: list[str] = Field(default_factory=list)


SYSTEM = """You follow a live university lecture transcript and align it to course slides.
Speech recognition may mishear mathematical phonetics. Be literal: only report concepts,
examples, questions, and clarity problems present in the new window. Repair spoken math
into inline LaTeX using the exact notation in SLIDE MATCHES. Never invent a slide number.
Write in the transcript language and return one JSON object with these keys:
- cleaned_text: the window with spoken math repaired into inline LaTeX
- summary: one sentence on what was explained
- matched_slide_number: slide_number from SLIDE MATCHES, or null
- similarity_score: 0-1 confidence in that match
- key_concepts: concepts explicitly explained
- examples_given: worked examples or concrete cases the professor actually walked through
- student_questions: questions students asked, verbatim when possible
- clarity_signals: moments that may confuse students (skipped steps, rushing, undefined symbols)
- quick_questions: up to 3 short check-in questions about this window for students
Use empty lists when the window contains none of an item."""

EXAMPLE_MARKERS = ("for example", "for instance", "example", "let's take", "consider")
RUSH_MARKERS = ("quickly", "let's move on", "skip", "obviously", "trivially")


def _detect_signals(raw_text: str) -> dict[str, list[str]]:
    """Keyword heuristics so the no-API demo still records teaching signals."""
    sentences = [part.strip() for part in raw_text.replace("?", "?.").split(".") if part.strip()]
    lower = raw_text.lower()
    return {
        "examples_given": [s for s in sentences if any(m in s.lower() for m in EXAMPLE_MARKERS)],
        "student_questions": [
            s.split(":", 1)[-1].strip() for s in sentences if s.endswith("?")
        ],
        "clarity_signals": (
            ["Professor moved on quickly; the step may need a recap."]
            if any(m in lower for m in RUSH_MARKERS)
            else []
        ),
    }


FALLBACK_QUESTIONS = {
    "nyquist": "Why must the Nyquist contour indent around a pole on the imaginary axis?",
    "bode": "How does the phase margin $\\varphi_m$ relate to closed-loop damping?",
}


def _fallback_question(topic: str) -> str:
    lower = topic.lower()
    if "bode" in lower or "phase" in lower:
        return FALLBACK_QUESTIONS["bode"]
    if "nyquist" in lower or "contour" in lower:
        return FALLBACK_QUESTIONS["nyquist"]
    return f"Could you explain the key idea of {topic} to a classmate?"


def should_run_listener(
    total_duration_ms: int, *, final: bool = False, window_seconds: int = LISTENER_WINDOW_SECONDS
) -> bool:
    """Preserved window rule: run at threshold, or flush any pending content on final."""
    return total_duration_ms >= window_seconds * 1000 or (final and total_duration_ms > 0)


def _fallback(raw_text: str, slide_matches: list[dict[str, Any]], duration_ms: int) -> dict:
    base = fallback_listener_response(raw_text)
    best = slide_matches[0] if slide_matches else {}
    slide = int(best.get("slide_number") or base["matched_slide_number"])
    score = float(best.get("score") or base["similarity_score"])
    cleaned = base["cleaned_text"]
    topic = best.get("topic") or (
        "Bode Phase Margin" if slide == 24 else "Nyquist Contour Indentation"
    )
    return {
        "topic_id": slide,
        "off_plan_topic": None,
        "summary": cleaned,
        "key_concepts": [topic],
        **_detect_signals(raw_text),
        "confidence": score,
        "cleaned_text": cleaned,
        "matched_slide_number": slide,
        "similarity_score": score,
        "time_spent_seconds": max(1, round(duration_ms / 1000)),
        "quick_questions": [_fallback_question(topic)],
    }


def run_listener(
    *,
    lecture_id: int,
    raw_text: str,
    duration_ms: int,
) -> ListenerOutput:
    from core.models import Lecture
    from mcp_server.server import search_course_slides

    lecture = Lecture.objects.select_related("course").get(pk=lecture_id)
    if not raw_text.strip():
        return ListenerOutput(
            topic_id=None,
            summary="(silence)",
            confidence=0.0,
            time_spent_seconds=max(1, round(duration_ms / 1000)),
        )

    slide_matches = search_course_slides(raw_text, lecture_id, 3)
    fallback = _fallback(raw_text, slide_matches, duration_ms)
    user = (
        f"LECTURE: {lecture.title}\n"
        f"COURSE: {lecture.course.code} — {lecture.course.title}\n"
        f"SLIDE MATCHES:\n{json.dumps(slide_matches, ensure_ascii=False)}\n\n"
        f"NEW TRANSCRIPT WINDOW:\n\"\"\"\n{raw_text}\n\"\"\""
    )
    data = chat_json(system=SYSTEM, user=user, fallback=fallback)
    merged = {**fallback, **data}

    valid_slides = set(lecture.slides.values_list("slide_number", flat=True))
    matched = merged.get("matched_slide_number")
    if matched not in valid_slides:
        matched = fallback["matched_slide_number"] if fallback["matched_slide_number"] in valid_slides else None
        merged["matched_slide_number"] = matched
    if merged.get("topic_id") not in valid_slides:
        merged["topic_id"] = matched
    merged["time_spent_seconds"] = max(1, round(duration_ms / 1000))
    return ListenerOutput.model_validate(merged)


def process_transcript_chunk(
    *,
    lecture_id: int,
    raw_text: str,
    seq: int,
    start_ms: int,
    duration_ms: int,
) -> dict:
    from core.models import Lecture, SlideChunk, TranscriptSegment

    lecture = Lecture.objects.get(pk=lecture_id)
    result = run_listener(
        lecture_id=lecture_id,
        raw_text=raw_text,
        duration_ms=duration_ms,
    )
    slide = (
        SlideChunk.objects.filter(lecture=lecture, slide_number=result.matched_slide_number).first()
        if result.matched_slide_number is not None
        else None
    )
    topic = slide.topic if slide else (result.off_plan_topic or "")
    segment, _ = TranscriptSegment.objects.update_or_create(
        lecture=lecture,
        seq=seq,
        defaults={
            "start_ms": start_ms,
            "duration_ms": duration_ms,
            "raw_text": raw_text,
            "cleaned_text": result.cleaned_text or result.summary,
            "matched_slide_number": result.matched_slide_number,
            "similarity_score": result.similarity_score,
            "signals": {
                "topic": topic,
                "key_concepts": result.key_concepts,
                "examples_given": result.examples_given,
                "student_questions": result.student_questions,
                "clarity_signals": result.clarity_signals,
            },
        },
    )
    if slide:
        SlideChunk.objects.filter(pk=slide.pk).update(
            time_spent_seconds=F("time_spent_seconds") + result.time_spent_seconds,
            best_similarity=result.similarity_score,
        )
    if result.quick_questions and topic:
        update_pulse_questions(lecture, topic, result.quick_questions[0])
    return {"segment_id": segment.id, **result.model_dump()}


def update_pulse_questions(lecture, topic: str, question: str) -> None:
    """Keep one check-in question per recent topic, so the pulse spans the lecture."""
    others = [
        {"text": item["text"], "topic": item["topic"]}
        for item in lecture.pulse_questions()
        if item["topic"] != topic
    ]
    lecture.checkin_questions = (others + [{"text": question, "topic": topic}])[-3:]
    lecture.save(update_fields=["checkin_questions"])
