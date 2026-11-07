"""Agent 2: deterministic, evidence-first pedagogical gap analysis."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from django.db.models import Avg

from agents.llm_client import chat_json
from core.models import GapAlert, Lecture
from mcp_server.server import match_past_exams

BRIEF_SYSTEM = """You are the PoliTo Pulse dashboard analyst.
Write a concise pedagogical brief for the professor from the evidence JSON only.
Return one JSON object with:
- summary: two sentences on what the lesson actually covered and how well students followed it
- improvements: two or three short sentences on what to reteach, each tied to audio time, quiz results, or exam weight in the evidence
- animations: two or three objects with title and why, each naming a Manim animation to generate and the gap it closes
Use the lecture language. Keep symbols in inline LaTeX ($...$). Do not invent numbers that are not in the evidence."""


class NoFeedbackYet(Exception):
    pass


def severity(n: int, pct_low: float, avg: float | None) -> str | None:
    """Preserved thresholds from the original Analyst agent."""
    if n == 0 or avg is None:
        return None
    if pct_low >= 0.40 or avg < 2.5:
        return "high"
    if pct_low >= 0.20 or avg < 3.3:
        return "medium"
    return "low"


def evidence_lines(topic: dict[str, Any], threshold: int = 2) -> list[str]:
    lines = [
        f"{topic['n_low']}/{topic['n_ratings']} students "
        f"({topic['pct_low']:.0%}) rated understanding <= {threshold}/5; "
        f"average {topic['avg_understanding']}/5."
    ]
    if not topic["covered"]:
        lines.append("The transcript shows no time spent on this topic.")
    else:
        lines.append(
            f"Taught for about {topic['minutes_taught']} min, "
            f"{topic['n_examples']} example(s), "
            f"{topic['n_student_questions']} student question(s)."
        )
    return lines


def _quiz_stats(lecture: Lecture) -> tuple[int, float, float]:
    checkins = lecture.checkins.all()
    n = checkins.count()
    average = float(checkins.aggregate(value=Avg("score"))["value"] or 0)
    n_low = sum(1 for score in checkins.values_list("score", flat=True) if score < 0.6)
    return n, average, (n_low / n if n else 0.0)


def compute_stats(lecture: Lecture) -> dict[str, Any]:
    n, quiz_average, pct_low = _quiz_stats(lecture)
    topic_times: dict[str, int] = defaultdict(int)
    slides = list(lecture.slides.all())
    for slide in slides:
        topic_times[slide.topic] += slide.time_spent_seconds

    topics: list[dict[str, Any]] = []
    for topic, seconds in topic_times.items():
        avg_five = round(quiz_average * 5, 2) if n else None
        topics.append(
            {
                "name": topic,
                "n_ratings": n,
                "avg_understanding": avg_five,
                "n_low": round(pct_low * n),
                "pct_low": pct_low,
                "severity": severity(n, pct_low, avg_five),
                "minutes_taught": round(seconds / 60, 1),
                "covered": seconds > 0,
                "n_examples": 0,
                "n_student_questions": 0,
            }
        )
    return {
        "lecture_id": lecture.id,
        "title": lecture.title,
        "n_responses": n,
        "quiz_average": quiz_average,
        "topics": topics,
        "slide_coverage": (
            round(sum(1 for slide in slides if slide.time_spent_seconds > 0) / len(slides) * 100)
            if slides
            else 0
        ),
    }


def _severity_for_gap(audio_seconds: int, quiz_score: float, exam_weight: float) -> str:
    if exam_weight >= 0.7 and (audio_seconds < 60 or quiz_score < 0.5):
        return GapAlert.Severity.CRITICAL
    if audio_seconds < 90 or quiz_score < 0.7:
        return GapAlert.Severity.WARNING
    return GapAlert.Severity.OPTIMAL


def _gap_rows(gaps: list[GapAlert]) -> list[dict[str, Any]]:
    return [
        {
            "topic": gap.topic,
            "severity": gap.severity,
            "audio_seconds": gap.audio_time_seconds,
            "quiz_score": gap.quiz_score,
            "exam_weight": gap.exam_weight,
            "exam_frequency": gap.exam_frequency,
            "recommendation": gap.recommendation,
        }
        for gap in gaps
    ]


def _transcript_excerpt(lecture: Lecture, limit: int = 700) -> str:
    parts: list[str] = []
    total = 0
    for text in lecture.transcripts.order_by("seq").values_list("cleaned_text", flat=True):
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        parts.append(cleaned)
        total += len(cleaned)
        if total >= limit:
            break
    return " ".join(parts)[:limit]


def _animation_title(topic: str) -> str:
    lower = topic.lower()
    if "nyquist" in lower:
        return "Nyquist indentation around the imaginary-axis pole"
    if "bode" in lower or "phase" in lower:
        return "Gain crossover traced against phase margin"
    return f"Step-through of {topic}"


def _fallback_brief(
    lecture: Lecture, stats: dict[str, Any], gap_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    topics = [topic["name"] for topic in stats.get("topics", []) if topic.get("name")]
    topic_list = ", ".join(topics[:4]) or lecture.title
    coverage = stats.get("slide_coverage", 0)
    n_responses = stats.get("n_responses") or 0
    if n_responses:
        pulse = (
            f"Student check-ins average {stats.get('quiz_average') or 0:.0%} "
            f"across {n_responses} response{'s' if n_responses != 1 else ''}."
        )
    else:
        pulse = "No student check-ins have arrived yet."
    excerpt = _transcript_excerpt(lecture, 180)
    if excerpt:
        summary = (
            f"{lecture.title} centers on {topic_list}. "
            f"The recording says: {excerpt} {pulse}"
        )
    else:
        summary = (
            f"{lecture.title} centers on {topic_list}. "
            f"Matched audio covers {coverage}% of the slides. {pulse}"
        )

    ranked = [gap for gap in gap_rows if gap["severity"] != GapAlert.Severity.OPTIMAL] or gap_rows
    improvements: list[str] = []
    animations: list[dict[str, str]] = []
    for gap in ranked[:3]:
        improvements.append(
            f"Reteach {gap['topic']}: {gap['audio_seconds']}s of audio, "
            f"quiz score {gap['quiz_score']:.0%}, exam weight {gap['exam_weight']:.0%}"
            + (f" ({gap['exam_frequency']})." if gap["exam_frequency"] else ".")
        )
        animations.append(
            {
                "title": _animation_title(gap["topic"]),
                "why": gap["recommendation"],
            }
        )
    if not improvements:
        for topic in stats.get("topics", [])[:3]:
            improvements.append(
                f"Revisit {topic['name']} "
                f"({topic['minutes_taught']} min matched in the recording)."
            )
            animations.append(
                {
                    "title": _animation_title(topic["name"]),
                    "why": (
                        "Generate a short scene of the defining diagram so the derivation "
                        "stays visible after the lecture."
                    ),
                }
            )
    if not improvements:
        improvements = [
            f"Match more of {lecture.title} to the slides before the next check-in."
        ]
        animations = [
            {
                "title": _animation_title(lecture.title),
                "why": "Generate the core diagram once a slide match is available.",
            }
        ]
    return {
        "summary": " ".join(summary.split()),
        "improvements": improvements[:3],
        "animations": animations[:3],
    }


def _as_text_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    lines = [str(item).strip() for item in value if str(item).strip()]
    return lines[:3] or fallback


def _as_animations(value: Any, fallback: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return fallback
    animations: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            animations.append({"title": item.strip(), "why": ""})
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "").strip()
            why = str(item.get("why") or item.get("rationale") or "").strip()
            if title:
                animations.append({"title": title, "why": why})
        if len(animations) == 3:
            break
    return animations or fallback


def _normalize_brief(raw: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    summary = raw.get("summary") if isinstance(raw, dict) else None
    summary = str(summary).strip() if summary else ""
    return {
        "summary": summary or fallback["summary"],
        "improvements": _as_text_list(
            raw.get("improvements") if isinstance(raw, dict) else None,
            fallback["improvements"],
        ),
        "animations": _as_animations(
            raw.get("animations") if isinstance(raw, dict) else None,
            fallback["animations"],
        ),
    }


def compose_lesson_brief(
    lecture: Lecture,
    stats: dict[str, Any],
    gaps: list[GapAlert],
) -> dict[str, Any]:
    """Concise lesson read: what happened, what to improve, which animations to generate."""
    gap_rows = _gap_rows(gaps)
    fallback = _fallback_brief(lecture, stats, gap_rows)
    evidence = {
        "title": lecture.title,
        "course": lecture.course.title,
        "slide_coverage": stats.get("slide_coverage"),
        "n_responses": stats.get("n_responses"),
        "quiz_average": stats.get("quiz_average"),
        "topics": stats.get("topics"),
        "transcript_excerpt": _transcript_excerpt(lecture),
        "misconceptions": list(
            lecture.checkins.exclude(misconception="").values_list("misconception", flat=True)[:5]
        ),
        "gaps": gap_rows,
    }
    raw = chat_json(
        system=BRIEF_SYSTEM,
        user=json.dumps(evidence, default=str),
        fallback=fallback,
    )
    brief = _normalize_brief(raw if isinstance(raw, dict) else {}, fallback)
    lecture.lesson_brief = brief
    return brief


def display_lesson_brief(lecture: Lecture, stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prefer the brief saved by the audit; otherwise a live evidence fallback."""
    stored = lecture.lesson_brief if isinstance(lecture.lesson_brief, dict) else {}
    if stored.get("summary") and stored.get("improvements") and stored.get("animations"):
        return _normalize_brief(
            stored,
            {
                "summary": str(stored["summary"]),
                "improvements": ["Revisit the rushed topics before the next check-in."],
                "animations": [{"title": lecture.title, "why": ""}],
            },
        )
    stats = stats or compute_stats(lecture)
    return _fallback_brief(lecture, stats, _gap_rows(list(lecture.gaps.all())))


def run_gap_audit(lecture_id: int, *, allow_empty_feedback: bool = True) -> dict[str, Any]:
    lecture = Lecture.objects.select_related("course").get(pk=lecture_id)
    stats = compute_stats(lecture)
    if not stats["n_responses"] and not allow_empty_feedback:
        raise NoFeedbackYet(lecture_id)

    quiz_score = stats["quiz_average"] or 0.62
    grouped: dict[str, list] = defaultdict(list)
    for slide in lecture.slides.all():
        grouped[slide.topic].append(slide)

    created: list[GapAlert] = []
    for topic, slides in grouped.items():
        audio_seconds = sum(slide.time_spent_seconds for slide in slides)
        exam_matches = match_past_exams(topic, 3)
        top_exam = exam_matches[0] if exam_matches else {}
        point_weight = float(top_exam.get("point_weight", 5))
        exam_weight = min(1.0, point_weight / 10)
        level = _severity_for_gap(audio_seconds, quiz_score, exam_weight)
        first_slide = max(
            slides,
            key=lambda slide: (slide.best_similarity, slide.time_spent_seconds, -slide.slide_number),
        ).slide_number
        if "nyquist" in topic.lower():
            recommendation = (
                "Visualize how the infinitesimal semicircle maps around $-1$ before "
                "revisiting $N = Z - P$."
            )
        else:
            recommendation = (
                "Animate the gain crossover and phase trace together to connect "
                "$\\omega_c$, $\\varphi_m$, and damping."
            )
        gap, _ = GapAlert.objects.update_or_create(
            lecture=lecture,
            topic=topic,
            defaults={
                "title": topic,
                "severity": level,
                "audio_time_seconds": audio_seconds,
                "quiz_score": quiz_score,
                "exam_weight": exam_weight,
                "exam_frequency": top_exam.get(
                    "frequency", "Appears in 3/5 Past Appelli"
                ),
                "slide_number": first_slide,
                "recommendation": recommendation,
            },
        )
        created.append(gap)

    brief = compose_lesson_brief(lecture, stats, created)
    lecture.status = Lecture.Status.AUDITED
    lecture.save(update_fields=["lesson_brief", "status"])
    return {"stats": stats, "gap_ids": [gap.id for gap in created], "lesson_brief": brief}
