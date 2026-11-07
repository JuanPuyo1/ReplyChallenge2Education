"""All SQL lives here. The two feedback queries marked ADAPT are the only ones that
depend on how your student form stores its results."""
from __future__ import annotations

from pathlib import Path

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from . import config

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=10,
                               kwargs={"row_factory": dict_row}, open=True)
    return _pool


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def init_schema() -> None:
    sql = (Path(__file__).resolve().parent.parent / "schema.sql").read_text()
    with pool().connection() as c:
        c.execute(sql)


# ---------------------------------------------------------------- lessons ---

def create_lesson(title: str, course: str | None, language: str | None,
                  topics: list[dict]) -> dict:
    with pool().connection() as c:
        lesson = c.execute(
            "INSERT INTO lessons (title, course, language) VALUES (%s, %s, %s) RETURNING *",
            (title, course, language or None)).fetchone()
        rows = []
        for i, t in enumerate(topics):
            rows.append(c.execute(
                "INSERT INTO lesson_topics (lesson_id, position, name, description) "
                "VALUES (%s, %s, %s, %s) RETURNING id, position, name, description",
                (lesson["id"], i, t["name"], t.get("description"))).fetchone())
    return {**lesson, "topics": rows}


def get_lesson(lesson_id: int) -> dict | None:
    with pool().connection() as c:
        lesson = c.execute("SELECT * FROM lessons WHERE id = %s", (lesson_id,)).fetchone()
        if not lesson:
            return None
        lesson["topics"] = c.execute(
            "SELECT id, position, name, description FROM lesson_topics "
            "WHERE lesson_id = %s ORDER BY position", (lesson_id,)).fetchall()
    return lesson


def end_lesson(lesson_id: int) -> None:
    with pool().connection() as c:
        c.execute("UPDATE lessons SET ended_at = now() WHERE id = %s", (lesson_id,))


# ------------------------------------------------------------- transcript ---

def upsert_segment(lesson_id: int, seq: int, start_ms: int, duration_ms: int, text: str) -> None:
    with pool().connection() as c:
        c.execute(
            "INSERT INTO transcript_segments (lesson_id, seq, start_ms, duration_ms, text) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (lesson_id, seq) DO UPDATE "
            "SET start_ms = EXCLUDED.start_ms, duration_ms = EXCLUDED.duration_ms, text = EXCLUDED.text",
            (lesson_id, seq, start_ms, duration_ms, text))


def transcript_tail(lesson_id: int, before_seq: int, chars: int = 600) -> str:
    """Last few hundred characters before this chunk, used as Whisper's prompt."""
    with pool().connection() as c:
        rows = c.execute(
            "SELECT text FROM transcript_segments WHERE lesson_id = %s AND seq < %s "
            "ORDER BY seq DESC LIMIT 3", (lesson_id, before_seq)).fetchall()
    return " ".join(r["text"] for r in reversed(rows))[-chars:]


def pending_segments(lesson_id: int) -> list[dict]:
    """Segments after the last Listener window, in order."""
    with pool().connection() as c:
        return c.execute(
            "SELECT seq, start_ms, duration_ms, text FROM transcript_segments s "
            "WHERE lesson_id = %s AND seq > COALESCE("
            "  (SELECT max(end_seq) FROM listener_annotations WHERE lesson_id = %s), -1) "
            "ORDER BY seq", (lesson_id, lesson_id)).fetchall()


def full_transcript(lesson_id: int) -> list[dict]:
    with pool().connection() as c:
        return c.execute(
            "SELECT seq, start_ms, duration_ms, text FROM transcript_segments "
            "WHERE lesson_id = %s ORDER BY seq", (lesson_id,)).fetchall()


# ------------------------------------------------------------ annotations ---

def insert_annotation(lesson_id: int, start_seq: int, end_seq: int, start_ms: int,
                      duration_ms: int, a: dict) -> dict:
    with pool().connection() as c:
        return c.execute(
            "INSERT INTO listener_annotations (lesson_id, start_seq, end_seq, start_ms, duration_ms,"
            " topic_id, off_plan_topic, summary, key_concepts, examples_given, student_questions,"
            " clarity_signals, confidence) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (lesson_id, start_seq) DO UPDATE SET end_seq = EXCLUDED.end_seq,"
            " duration_ms = EXCLUDED.duration_ms, topic_id = EXCLUDED.topic_id,"
            " off_plan_topic = EXCLUDED.off_plan_topic, summary = EXCLUDED.summary,"
            " key_concepts = EXCLUDED.key_concepts, examples_given = EXCLUDED.examples_given,"
            " student_questions = EXCLUDED.student_questions,"
            " clarity_signals = EXCLUDED.clarity_signals, confidence = EXCLUDED.confidence "
            "RETURNING *",
            (lesson_id, start_seq, end_seq, start_ms, duration_ms, a.get("topic_id"),
             a.get("off_plan_topic"), a["summary"], Jsonb(a.get("key_concepts", [])),
             Jsonb(a.get("examples_given", [])), Jsonb(a.get("student_questions", [])),
             Jsonb(a.get("clarity_signals", [])), a.get("confidence"))).fetchone()


def recent_annotations(lesson_id: int, n: int = 3) -> list[dict]:
    with pool().connection() as c:
        rows = c.execute(
            "SELECT topic_id, off_plan_topic, summary FROM listener_annotations "
            "WHERE lesson_id = %s ORDER BY start_seq DESC LIMIT %s", (lesson_id, n)).fetchall()
    return list(reversed(rows))


def all_annotations(lesson_id: int) -> list[dict]:
    with pool().connection() as c:
        return c.execute(
            "SELECT * FROM listener_annotations WHERE lesson_id = %s ORDER BY start_seq",
            (lesson_id,)).fetchall()


# --------------------------------------------------------------- feedback ---

def feedback_overall(lesson_id: int) -> dict:
    # ADAPT: point this at your form's table/columns.
    with pool().connection() as c:
        return c.execute(
            """
            SELECT count(*)                                      AS n_responses,
                   round(avg(overall_rating)::numeric, 2)::float AS avg_overall,
                   round(avg(pace_rating)::numeric, 2)::float    AS avg_pace,
                   count(*) FILTER (WHERE overall_rating <= 2)   AS n_negative,
                   count(*) FILTER (WHERE overall_rating >= 4)   AS n_positive,
                   jsonb_build_object(
                     '1', count(*) FILTER (WHERE overall_rating = 1),
                     '2', count(*) FILTER (WHERE overall_rating = 2),
                     '3', count(*) FILTER (WHERE overall_rating = 3),
                     '4', count(*) FILTER (WHERE overall_rating = 4),
                     '5', count(*) FILTER (WHERE overall_rating = 5)) AS distribution
            FROM feedback_responses WHERE lesson_id = %s
            """, (lesson_id,)).fetchone()


def feedback_per_topic(lesson_id: int, low: int) -> list[dict]:
    # ADAPT: one row per planned topic, with how many students rated it and how many
    # rated understanding <= `low`.
    with pool().connection() as c:
        return c.execute(
            """
            SELECT t.id AS topic_id, t.name, t.position,
                   count(r.understanding)                                 AS n_ratings,
                   round(avg(r.understanding)::numeric, 2)::float         AS avg_understanding,
                   count(*) FILTER (WHERE r.understanding <= %s)          AS n_low
            FROM lesson_topics t
            LEFT JOIN feedback_topic_ratings r ON r.topic_id = t.id
            WHERE t.lesson_id = %s
            GROUP BY t.id, t.name, t.position
            ORDER BY t.position
            """, (low, lesson_id)).fetchall()


def feedback_comments(lesson_id: int, limit: int = 80) -> list[str]:
    with pool().connection() as c:
        rows = c.execute(
            "SELECT comment FROM feedback_responses WHERE lesson_id = %s "
            "AND comment IS NOT NULL AND btrim(comment) <> '' ORDER BY submitted_at LIMIT %s",
            (lesson_id, limit)).fetchall()
    return [r["comment"][:500] for r in rows]


# ---------------------------------------------------------------- reports ---

def save_report(lesson_id: int, stats: dict, report: dict) -> dict:
    with pool().connection() as c:
        return c.execute(
            "INSERT INTO lesson_reports (lesson_id, stats, report) VALUES (%s, %s, %s) RETURNING *",
            (lesson_id, Jsonb(stats), Jsonb(report))).fetchone()


def latest_report(lesson_id: int) -> dict | None:
    with pool().connection() as c:
        return c.execute(
            "SELECT * FROM lesson_reports WHERE lesson_id = %s ORDER BY created_at DESC LIMIT 1",
            (lesson_id,)).fetchone()
