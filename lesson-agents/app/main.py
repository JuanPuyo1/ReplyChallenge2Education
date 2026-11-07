from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import analyst_agent, db, listener_agent, transcriber

STATIC = Path(__file__).resolve().parent.parent / "static"
# Chunks of one lesson are processed one at a time so Whisper's prompt and the Listener
# always see the previous text. Different lessons run in parallel.
_locks: dict[int, threading.Lock] = defaultdict(threading.Lock)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_schema()
    yield
    db.close()


app = FastAPI(title="Lesson Agents", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


class TopicIn(BaseModel):
    name: str
    description: str | None = None


class LessonIn(BaseModel):
    title: str
    course: str | None = None
    language: str | None = Field(None, description="'it', 'en', ... or null for auto")
    topics: list[TopicIn] = Field(min_length=1)


def _lesson_or_404(lesson_id: int) -> dict:
    lesson = db.get_lesson(lesson_id)
    if lesson is None:
        raise HTTPException(404, "lesson not found")
    return lesson


@app.post("/api/lessons")
def create_lesson(body: LessonIn):
    return db.create_lesson(body.title, body.course, body.language,
                            [t.model_dump() for t in body.topics])


@app.get("/api/lessons/{lesson_id}")
def get_lesson(lesson_id: int):
    return _lesson_or_404(lesson_id)


@app.post("/api/lessons/{lesson_id}/chunks")
def upload_chunk(lesson_id: int,
                 audio: UploadFile = File(...),
                 seq: int = Form(...),
                 start_ms: int = Form(...),
                 duration_ms: int = Form(...),
                 final: bool = Form(False)):
    """Agent 1: transcribe one chunk, then let the Listener annotate if a window is full."""
    lesson = _lesson_or_404(lesson_id)
    data = audio.file.read()
    with _locks[lesson_id]:
        text = ""
        if len(data) > 2000:  # an empty/near-empty recorder blob is just container headers
            prompt = transcriber.build_prompt(lesson["title"], lesson["topics"],
                                              db.transcript_tail(lesson_id, seq))
            text = transcriber.transcribe(data, audio.filename or f"chunk-{seq}.webm",
                                          prompt, lesson.get("language"))
        db.upsert_segment(lesson_id, seq, start_ms, duration_ms, text)
        annotation = listener_agent.maybe_run(lesson, final=final)
        if final:
            db.end_lesson(lesson_id)
    return {"seq": seq, "text": text, "annotation": annotation}


@app.get("/api/lessons/{lesson_id}/transcript")
def transcript(lesson_id: int):
    _lesson_or_404(lesson_id)
    return {"segments": db.full_transcript(lesson_id),
            "annotations": db.all_annotations(lesson_id)}


@app.post("/api/lessons/{lesson_id}/analyze")
def analyze(lesson_id: int):
    """Agent 2: combine transcript annotations with student feedback into a report."""
    _lesson_or_404(lesson_id)
    try:
        return analyst_agent.run(lesson_id)
    except analyst_agent.NoFeedbackYet:
        raise HTTPException(409, "no student feedback for this lesson yet")


@app.get("/api/lessons/{lesson_id}/report")
def report(lesson_id: int):
    r = db.latest_report(lesson_id)
    if r is None:
        raise HTTPException(404, "no report yet; POST /analyze first")
    return r


@app.get("/api/lessons/{lesson_id}/report.md", response_class=PlainTextResponse)
def report_md(lesson_id: int):
    r = db.latest_report(lesson_id)
    if r is None:
        raise HTTPException(404, "no report yet")
    return analyst_agent.to_markdown(r["report"])
