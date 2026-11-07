# Lesson Agents

Two agents for lessons:

1. **Lesson Listener**: turns live lesson audio from the browser into a transcript. Every
   ~30 s it also notes which planned topic is being taught and how: concepts, examples,
   student questions, and clarity problems.
2. **Lesson Analyst**: joins those notes with the student form results in Postgres. It
   reports how good the lesson was, which topics need more attention, and the likely
   reasons.

```
Browser mic ──30 s standalone audio files──▶ POST /chunks
                                               │
                     Groq whisper-large-v3-turbo  (prompt = topic names + previous words)
                                               │
                              transcript_segments (Postgres)
                                               │  every LISTENER_WINDOW_S seconds
                          Listener agent: gpt-oss-20b ─▶ listener_annotations
                                                                │
 student form ─▶ feedback_responses / feedback_topic_ratings ───┤
                                                                ▼
                    POST /analyze: SQL stats ─▶ Analyst agent: gpt-oss-120b ─▶ lesson_reports
```

## Run

```bash
docker compose up -d                       # Postgres on :5432 (or use your own DATABASE_URL)
cp .env.example .env                       # add GROQ_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000, enter the topics, press **Start recording**, then **Stop**.
The schema is created on startup. To try the Analyst without real students:

```bash
python -m scripts.seed_feedback <lesson_id> 25
```

Then click **Run Analyst**, or `curl -X POST localhost:8000/api/lessons/<id>/analyze`.
A Markdown version is at `/api/lessons/<id>/report.md`.

`getUserMedia` needs `localhost` or HTTPS. If you use another device on the LAN, put it
behind HTTPS (e.g. `cloudflared tunnel` or `mkcert`).

## Design choices (and how they simplify your original plan)

**Steps 1 and 2 share one loop.** The browser uploads a standalone file every 30 s
over plain HTTP. The server transcribes it and runs the Listener when a window is full.
You don't need WebSockets, a separate scheduler or a message queue. Chunk length and
Listener window are set separately (`LISTENER_WINDOW_S`). For example, 15 s chunks give
faster captions while the Listener still runs every 30 s.

**The recorder rotates.** `MediaRecorder.start(timeslice)` only writes the WebM header
into the first blob, so later blobs can't be decoded on their own and Whisper rejects
them. The page therefore starts a new recorder before stopping the old one. Each chunk is
a valid file, and the short overlap means no audio is lost.

**Chunk boundaries.** Whisper gets the last ~600 characters as its `prompt`, plus the
lesson's topic names so it spells the jargon correctly. Segments that look like
silence-hallucinations are dropped. These include "Thanks for watching" and the Italian
"Sottotitoli a cura di…", and are detected via `no_speech_prob` and `compression_ratio`.

**Groq vs Deepgram.** Groq Whisper Turbo costs about $0.04 per audio hour. Deepgram
Nova-3 streaming costs roughly 10–20× more, but gives sub-300 ms live captions. For lesson
analysis a 30 s delay doesn't matter, so Groq is the simpler and cheaper choice. Use
Nova-3, not Nova-2, if you need live captions for students: Nova-2 is now the legacy
model.

**No agent framework.** Each agent is one structured LLM call with a Pydantic schema,
validation, and one automatic retry. This is easy to debug and test. LangGraph or similar
only becomes useful if you add tools or branching.

**The Analyst does no arithmetic.** Averages, the share of students who didn't understand
(rating ≤ `LOW_UNDERSTANDING`), minutes per topic, severity and ranking are all computed
in SQL/Python. The LLM only explains causes, groups the comments into themes and suggests
fixes. Its output is then merged with the computed numbers, so the report is
reproducible and the model can't invent statistics.

Severity rules (`analyst_agent.severity`):

- **high**: ≥ 40 % of students didn't understand, or average < 2.5
- **medium**: ≥ 20 %, or average < 3.3

## Connecting your existing form

Your form results are already in Postgres. Change the two functions marked **ADAPT** in
`app/db.py`, `feedback_overall` and `feedback_per_topic`, plus `feedback_comments`, to
read from your tables. The only requirement is that the form's topics map to
`lesson_topics.id`. The simplest way is to generate the form from the lesson plan. A
"tick what you didn't understand" form fits by treating ticked as 1 and unticked as 5.

## Test

```bash
DATABASE_URL=postgresql://... pytest -q
```

The test runs the whole pipeline on a real Postgres with a fake Groq client: upload
chunks, Listener alignment, hallucination filter, feedback, Analyst, JSON-retry path,
Markdown report.

## Practical notes

- **Audio quality matters more than the model.** A clip-on or USB mic near the teacher
  beats a laptop mic at the back of the room.
- Set the lesson language (`it`/`en`). Auto-detect can flip language on short chunks.
- Groq bills at least 10 s per request, so keep chunks ≥ 10 s.
- Recording students' voices may need consent under GDPR. Store pseudonymous
  `student_ref` values, not names.
