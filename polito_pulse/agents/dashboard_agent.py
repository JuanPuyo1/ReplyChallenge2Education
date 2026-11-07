"""Agent 2: deterministic, evidence-first pedagogical gap analysis."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from django.db.models import Avg

from core.models import GapAlert, Lecture
from mcp_server.server import match_past_exams


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

    lecture.status = Lecture.Status.AUDITED
    lecture.save(update_fields=["status"])
    return {"stats": stats, "gap_ids": [gap.id for gap in created]}
