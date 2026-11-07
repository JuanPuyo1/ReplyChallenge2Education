"""Agent 1, step 2: the Listener / Alignment agent.

Every ~30-60 s of transcript it answers: which planned topic is the teacher on right now,
and what does the teaching look like (concepts, examples, student questions, clarity
problems)? Its output is what lets Agent 2 connect "students didn't get topic X" to
"topic X got 3 minutes and no examples"."""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from . import config, db
from .llm import call_json


class ListenerOutput(BaseModel):
    topic_id: int | None = Field(
        description="id of the planned topic being taught in this window, or null if "
                    "none (admin talk, break, off-plan content)")
    off_plan_topic: str | None = Field(
        None, description="short name of what is being taught if it is not in the plan")
    summary: str = Field(description="1-2 sentences on what was taught in this window")
    key_concepts: list[str] = Field(default_factory=list, description="terms/ideas introduced")
    examples_given: list[str] = Field(
        default_factory=list, description="concrete examples, analogies, worked exercises")
    student_questions: list[str] = Field(
        default_factory=list, description="questions or confusion voiced by students")
    clarity_signals: list[str] = Field(
        default_factory=list,
        description="possible clarity problems: undefined jargon, rushing, skipped steps, "
                    "teacher saying 'this is hard'/'we'll skip this', contradictions")
    confidence: float = Field(ge=0, le=1, description="confidence in topic_id")


SYSTEM = """You follow a live lesson transcript and align it to the lesson plan.
The transcript comes from speech recognition: expect missing punctuation, misheard words,
and mostly the teacher's voice. Be literal: only report examples, questions and clarity
problems that are actually in the new transcript window. Empty lists are fine.
Write in the same language as the transcript."""


def run(lesson: dict, window_text: str) -> ListenerOutput:
    plan = [{"id": t["id"], "name": t["name"], "description": t.get("description") or ""}
            for t in lesson["topics"]]
    recent = db.recent_annotations(lesson["id"])
    user = (
        f"LESSON: {lesson['title']}\n"
        f"PLAN (in teaching order):\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"PREVIOUS WINDOWS (most recent last):\n{json.dumps(recent, ensure_ascii=False, default=str)}\n\n"
        f"NEW TRANSCRIPT WINDOW:\n\"\"\"\n{window_text}\n\"\"\""
    )
    out = call_json(config.LISTENER_MODEL, SYSTEM, user, ListenerOutput, reasoning_effort="low")
    if out.topic_id is not None and out.topic_id not in {t["id"] for t in lesson["topics"]}:
        out.topic_id = None  # model invented an id
    return out


def maybe_run(lesson: dict, final: bool = False) -> dict | None:
    """Run the Listener if enough un-annotated transcript has accumulated (or the
    lesson just ended). Returns the stored annotation, or None."""
    pending = db.pending_segments(lesson["id"])
    if not pending:
        return None
    total_ms = sum(s["duration_ms"] for s in pending)
    if total_ms < config.LISTENER_WINDOW_S * 1000 and not final:
        return None
    text = " ".join(s["text"] for s in pending if s["text"]).strip()
    start_seq, end_seq = pending[0]["seq"], pending[-1]["seq"]
    if not text:
        out = {"topic_id": None, "summary": "(silence)", "confidence": 0.0}
    else:
        out = run(lesson, text).model_dump()
    return db.insert_annotation(lesson["id"], start_seq, end_seq,
                                pending[0]["start_ms"], total_ms, out)
