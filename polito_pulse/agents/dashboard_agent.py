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
- next_lecture: object with recap (one {topic, action} per topic in next_lecture_draft.recap, where action is one concrete
  teaching move grounded in that topic's evidence: a worked example, a slower derivation, answering the students' questions,
  a check question) and warm_up_question (one question to open the next lecture)
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


LOW_SCORE = 0.5  # a "confused" check-in answer scores 0.25, "recap" 0.55, "clear" 1.0
SIGNAL_KEYS = ("examples_given", "student_questions", "clarity_signals")


def evidence_lines(topic: dict[str, Any]) -> list[str]:
    if topic["n_ratings"]:
        lines = [
            f"{topic['n_low']}/{topic['n_ratings']} students ({topic['pct_low']:.0%}) "
            f"marked it confusing; comprehension {topic['comprehension']:.0%}."
        ]
    else:
        lines = ["No student check-ins on this topic yet."]
    if not topic["covered"]:
        lines.append(f"Slides {topic['slides']} were never matched in the lecture audio.")
    else:
        lines.append(
            f"Taught for about {topic['minutes_taught']} min, "
            f"{topic['n_examples']} worked example(s), "
            f"{topic['n_student_questions']} student question(s)."
        )
    if topic["clarity_signals"]:
        lines.append(f"Pace flag: {topic['clarity_signals'][0]}")
    return lines


def teaching_actions(topic: dict[str, Any], exam_weight: float, exam_frequency: str) -> list[str]:
    """Concrete, evidence-backed moves for the professor's next lecture."""
    if not topic["covered"]:
        return [f"Slides {topic['slides']} were skipped: schedule them in the next lecture."]
    actions: list[str] = []
    if topic["seconds_taught"] < 90:
        actions.append(
            f"Only {topic['seconds_taught']}s on slides {topic['slides']}: "
            "budget a dedicated recap block."
        )
    if topic["n_examples"] == 0:
        actions.append("No worked example was given: walk through one before moving on.")
    if topic["student_questions"]:
        actions.append(
            f"Open by answering the {len(topic['student_questions'])} student question(s) raised."
        )
    if topic["clarity_signals"]:
        actions.append("Slow down the derivation flagged by the pace signal and write every step.")
    if topic["n_ratings"] and topic["pct_low"] >= 0.4:
        actions.append(
            f"{topic['pct_low']:.0%} of students were confused: run a check question after the recap."
        )
    elif topic["n_ratings"] and topic["comprehension"] < 0.7:
        actions.append(
            f"Only {topic['comprehension']:.0%} comprehension after {topic['minutes_taught']} min: "
            "re-explain from a different angle (graphical intuition first, then the formula)."
        )
    if exam_weight >= 0.7 and actions:
        actions.append(f"High exam stakes ({exam_frequency}): prioritize it over new material.")
    return actions[:4]


def _quiz_stats(lecture: Lecture) -> tuple[int, float, float]:
    checkins = lecture.checkins.all()
    n = checkins.count()
    average = float(checkins.aggregate(value=Avg("score"))["value"] or 0)
    n_low = sum(1 for score in checkins.values_list("score", flat=True) if score < 0.6)
    return n, average, (n_low / n if n else 0.0)


def _topic_scores(lecture: Lecture) -> dict[str, list[float]]:
    """Per-topic check-in scores from topic-tagged pulse questions."""
    scores: dict[str, list[float]] = defaultdict(list)
    for topic_scores in lecture.checkins.values_list("topic_scores", flat=True):
        for topic, score in (topic_scores or {}).items():
            scores[topic].append(float(score))
    return scores


def _topic_signals(lecture: Lecture, slide_topics: dict[int, str]) -> dict[str, dict[str, list]]:
    signals: dict[str, dict[str, list]] = defaultdict(lambda: {key: [] for key in SIGNAL_KEYS})
    for segment in lecture.transcripts.all():
        data = segment.signals or {}
        topic = data.get("topic") or slide_topics.get(segment.matched_slide_number)
        if topic:
            for key in SIGNAL_KEYS:
                seen = {item.lower() for item in signals[topic][key]}
                for item in data.get(key) or []:
                    if str(item).lower() not in seen:
                        signals[topic][key].append(str(item))
                        seen.add(str(item).lower())
    return signals


def _slide_range(numbers: list[int]) -> str:
    return f"{min(numbers)}–{max(numbers)}" if len(numbers) > 1 else str(numbers[0])


def compute_stats(lecture: Lecture) -> dict[str, Any]:
    n, quiz_average, pct_low = _quiz_stats(lecture)
    slides = list(lecture.slides.all())
    by_topic: dict[str, list] = defaultdict(list)
    for slide in slides:
        by_topic[slide.topic].append(slide)
    topic_scores = _topic_scores(lecture)
    signals = _topic_signals(lecture, {slide.slide_number: slide.topic for slide in slides})
    has_topic_pulse = bool(topic_scores)

    topics: list[dict[str, Any]] = []
    for topic, topic_slides in by_topic.items():
        seconds = sum(slide.time_spent_seconds for slide in topic_slides)
        if has_topic_pulse:
            scores = topic_scores.get(topic, [])
        else:
            scores = list(lecture.checkins.values_list("score", flat=True))
        comprehension = sum(scores) / len(scores) if scores else None
        n_topic_low = sum(1 for score in scores if score < LOW_SCORE)
        topic_pct_low = n_topic_low / len(scores) if scores else 0.0
        avg_five = round(comprehension * 5, 2) if comprehension is not None else None
        topic_signals = signals[topic]
        topics.append(
            {
                "name": topic,
                "slides": _slide_range([slide.slide_number for slide in topic_slides]),
                "n_ratings": len(scores),
                "comprehension": comprehension,
                "avg_understanding": avg_five,
                "n_low": n_topic_low,
                "pct_low": topic_pct_low,
                "severity": severity(len(scores), topic_pct_low, avg_five),
                "seconds_taught": seconds,
                "minutes_taught": round(seconds / 60, 1),
                "covered": seconds > 0,
                "n_examples": len(topic_signals["examples_given"]),
                "n_student_questions": len(topic_signals["student_questions"]),
                "examples_given": topic_signals["examples_given"][:5],
                "student_questions": topic_signals["student_questions"][:5],
                "clarity_signals": topic_signals["clarity_signals"][:5],
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


def _severity_for_gap(audio_seconds: int, quiz_score: float | None, exam_weight: float) -> str:
    confused = quiz_score is not None and quiz_score < 0.5
    unclear = quiz_score is not None and quiz_score < 0.7
    if exam_weight >= 0.7 and (audio_seconds < 60 or confused):
        return GapAlert.Severity.CRITICAL
    if audio_seconds < 90 or unclear:
        return GapAlert.Severity.WARNING
    return GapAlert.Severity.OPTIMAL


def _visual_hint(topic: str) -> str:
    """What a Manim animation for this topic should show students."""
    lower = topic.lower()
    if "indentation" in lower or "imaginary" in lower:
        return (
            "Visualize how the infinitesimal semicircle maps around $-1$ before "
            "revisiting $N = Z - P$."
        )
    if "gain margin" in lower:
        return "Show the phase crossover $\\omega_{pc}$ and read the gain margin off the Bode magnitude."
    if "bode" in lower or "phase" in lower:
        return (
            "Animate the gain crossover and phase trace together to connect "
            "$\\omega_c$, $\\varphi_m$, and damping."
        )
    if "nyquist" in lower:
        return "Trace $G(j\\omega)$ as $\\omega$ sweeps from $0$ to $\\infty$ and mirror it for negative $\\omega$."
    return f"Step through the defining diagram of {topic} one element at a time."


def _duration(seconds: int) -> str:
    return f"{seconds}s" if seconds < 120 else f"{round(seconds / 60, 1)} min"


def _gap_rows(gaps: list[GapAlert]) -> list[dict[str, Any]]:
    return [
        {
            "topic": gap.topic,
            "covered": (gap.evidence or {}).get("covered", gap.audio_time_seconds > 0),
            "slides": (gap.evidence or {}).get("slides", ""),
            "severity": gap.severity,
            "audio_seconds": gap.audio_time_seconds,
            "quiz_score": gap.quiz_score,
            "exam_weight": gap.exam_weight,
            "exam_frequency": gap.exam_frequency,
            "recommendation": gap.recommendation,
            "evidence": gap.evidence.get("lines", []),
            "teaching_actions": gap.teaching_actions,
        }
        for gap in gaps
    ]


RECAP_MINUTES = {GapAlert.Severity.CRITICAL: 8, GapAlert.Severity.WARNING: 5}


def next_lecture_plan(lecture: Lecture, gaps: list[GapAlert]) -> dict[str, Any]:
    """Deterministic plan: recap the worst covered gaps, carry over skipped slides."""
    from agents.listener_agent import _fallback_question

    ranked = sorted(
        (gap for gap in gaps if gap.severity != GapAlert.Severity.OPTIMAL),
        key=lambda gap: (gap.severity != GapAlert.Severity.CRITICAL, -gap.exam_weight),
    )
    questions_by_topic = {item["topic"]: item["text"] for item in lecture.pulse_questions()}
    recap: list[dict[str, Any]] = []
    carry_over: list[dict[str, str]] = []
    student_questions: list[str] = []
    for gap in ranked:
        evidence = gap.evidence or {}
        if not evidence.get("covered", True):
            carry_over.append({"topic": gap.topic, "slides": evidence.get("slides", "")})
            continue
        if len(recap) == 2:
            continue
        why = [f"{_duration(gap.audio_time_seconds)} taught"]
        if gap.quiz_score is not None:
            why.insert(0, f"{gap.quiz_score:.0%} comprehension")
        if gap.exam_frequency:
            why.append(gap.exam_frequency.lower())
        # The recap block itself answers the "budget more time" action.
        budget_prefix = f"Only {gap.audio_time_seconds}s on "
        actions = [
            action for action in gap.teaching_actions if not action.startswith(budget_prefix)
        ] or [_visual_hint(gap.topic)]
        recap.append(
            {
                "topic": gap.topic,
                "slides": evidence.get("slides", ""),
                "minutes": RECAP_MINUTES.get(gap.severity, 5),
                "severity": gap.severity,
                "why": ", ".join(why),
                "action": actions[0],
                "actions": actions[:3],
            }
        )
        student_questions.extend(evidence.get("student_questions", []))
    warm_up = ""
    if recap:
        top = recap[0]["topic"]
        warm_up = questions_by_topic.get(top) or _fallback_question(top)
    return {
        "recap": recap,
        "carry_over": carry_over,
        "student_questions": student_questions[:3],
        "warm_up_question": warm_up,
        "total_minutes": sum(item["minutes"] for item in recap),
    }


def _merge_plan(raw: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    """Accept the LLM's wording for actions and warm-up; keep the numbers grounded."""
    plan = {**fallback, "recap": [dict(item) for item in fallback["recap"]]}
    if not isinstance(raw, dict):
        return plan
    actions = {
        str(item.get("topic", "")).strip().lower(): str(item.get("action", "")).strip()
        for item in raw.get("recap") or []
        if isinstance(item, dict)
    }
    for item in plan["recap"]:
        action = actions.get(item["topic"].lower())
        if action and action not in item.get("actions", []):
            item["action"] = action
            item["actions"] = [action] + item.get("actions", [])[:2]
    warm_up = str(raw.get("warm_up_question") or "").strip()
    if warm_up and plan["recap"]:
        plan["warm_up_question"] = warm_up
    return plan


def plan_followup(lecture: Lecture, stats: dict[str, Any]) -> dict[str, Any] | None:
    """Did the previous lecture's recap plan move comprehension on those topics?"""
    previous = (
        Lecture.objects.filter(course=lecture.course, number__lt=lecture.number)
        .order_by("-number")
        .first()
    )
    if not previous:
        return None
    plan = (previous.lesson_brief or {}).get("next_lecture") or {}
    now_by_topic = {topic["name"]: topic for topic in stats.get("topics", [])}
    before_by_topic = {gap.topic: gap.quiz_score for gap in previous.gaps.all()}
    rows = []
    for item in plan.get("recap", []):
        topic = now_by_topic.get(item.get("topic"))
        before = before_by_topic.get(item.get("topic"))
        if not topic or topic["comprehension"] is None or before is None:
            continue
        rows.append(
            {
                "topic": topic["name"],
                "before": round(before * 100),
                "after": round(topic["comprehension"] * 100),
                "delta": round((topic["comprehension"] - before) * 100),
                "minutes_taught": topic["minutes_taught"],
            }
        )
    return {"lecture_number": previous.number, "rows": rows} if rows else None


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
    topic_stats = [topic for topic in stats.get("topics", []) if topic.get("name")]
    coverage = stats.get("slide_coverage", 0)
    n_responses = stats.get("n_responses") or 0
    parts = [f"{lecture.title}: {coverage}% of the slides were matched in the lecture audio."]
    scored = [topic for topic in topic_stats if topic.get("comprehension") is not None]
    if len(scored) > 1:
        best = max(scored, key=lambda topic: topic["comprehension"])
        worst = min(scored, key=lambda topic: topic["comprehension"])
        parts.append(
            f"Students followed {best['name']} best ({best['comprehension']:.0%}); "
            f"{worst['name']} was the weakest ({worst['comprehension']:.0%} after "
            f"{_duration(worst['seconds_taught'])})."
        )
    skipped = [topic["name"] for topic in topic_stats if not topic.get("covered")]
    if skipped:
        parts.append(f"Not reached: {', '.join(skipped)}.")
    if n_responses:
        parts.append(
            f"Check-ins average {stats.get('quiz_average') or 0:.0%} "
            f"across {n_responses} response{'s' if n_responses != 1 else ''}."
        )
    else:
        parts.append("No student check-ins have arrived yet.")
    summary = " ".join(parts)

    ranked = [gap for gap in gap_rows if gap["severity"] != GapAlert.Severity.OPTIMAL] or gap_rows
    improvements: list[str] = []
    animations: list[dict[str, str]] = []
    for gap in ranked[:3]:
        exam_part = f" ({gap['exam_frequency']})" if gap["exam_frequency"] else ""
        if not gap["covered"]:
            improvements.append(
                f"Cover {gap['topic']} (slides {gap['slides']}) next lecture: "
                f"it was not reached{exam_part}."
            )
            continue
        pulse_part = (
            f"comprehension {gap['quiz_score']:.0%}, "
            if gap["quiz_score"] is not None
            else "no check-ins yet, "
        )
        improvements.append(
            f"Reteach {gap['topic']}: {_duration(gap['audio_seconds'])} of audio, "
            f"{pulse_part}exam weight {gap['exam_weight']:.0%}{exam_part}."
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
        "next_lecture": _merge_plan(
            raw.get("next_lecture") if isinstance(raw, dict) else None,
            fallback["next_lecture"],
        ),
    }


def compose_lesson_brief(
    lecture: Lecture,
    stats: dict[str, Any],
    gaps: list[GapAlert],
) -> dict[str, Any]:
    """Concise lesson read: what happened, what to improve, the next-lecture plan."""
    gap_rows = _gap_rows(gaps)
    fallback = {
        **_fallback_brief(lecture, stats, gap_rows),
        "next_lecture": next_lecture_plan(lecture, gaps),
    }
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
        "next_lecture_draft": fallback["next_lecture"],
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
    stats = stats or compute_stats(lecture)
    gaps = list(lecture.gaps.all())
    live = {
        **_fallback_brief(lecture, stats, _gap_rows(gaps)),
        "next_lecture": next_lecture_plan(lecture, gaps),
    }
    if stored.get("summary") and stored.get("improvements") and stored.get("animations"):
        brief = _normalize_brief(stored, live)
        if isinstance(stored.get("next_lecture"), dict):
            brief["next_lecture"] = {**live["next_lecture"], **stored["next_lecture"]}
        return brief
    return live


def run_gap_audit(lecture_id: int, *, allow_empty_feedback: bool = True) -> dict[str, Any]:
    lecture = Lecture.objects.select_related("course").get(pk=lecture_id)
    stats = compute_stats(lecture)
    if not stats["n_responses"] and not allow_empty_feedback:
        raise NoFeedbackYet(lecture_id)

    topic_stats = {topic["name"]: topic for topic in stats["topics"]}
    grouped: dict[str, list] = defaultdict(list)
    for slide in lecture.slides.all():
        grouped[slide.topic].append(slide)

    created: list[GapAlert] = []
    for topic, slides in grouped.items():
        evidence = topic_stats[topic]
        audio_seconds = evidence["seconds_taught"]
        quiz_score = evidence["comprehension"]
        exam_matches = match_past_exams(topic, 3)
        top_exam = exam_matches[0] if exam_matches else {}
        point_weight = float(top_exam.get("point_weight", 5))
        exam_weight = min(1.0, point_weight / 10)
        exam_frequency = top_exam.get("frequency", "No match in past appelli")
        level = _severity_for_gap(audio_seconds, quiz_score, exam_weight)
        first_slide = max(
            slides,
            key=lambda slide: (slide.best_similarity, slide.time_spent_seconds, -slide.slide_number),
        ).slide_number
        gap, _ = GapAlert.objects.update_or_create(
            lecture=lecture,
            topic=topic,
            defaults={
                "title": topic,
                "severity": level,
                "audio_time_seconds": audio_seconds,
                "quiz_score": quiz_score,
                "exam_weight": exam_weight,
                "exam_frequency": exam_frequency,
                "slide_number": first_slide,
                "recommendation": _visual_hint(topic),
                "evidence": {
                    "lines": evidence_lines(evidence),
                    "slides": evidence["slides"],
                    "covered": evidence["covered"],
                    "n_ratings": evidence["n_ratings"],
                    "n_examples": evidence["n_examples"],
                    "student_questions": evidence["student_questions"],
                    "clarity_signals": evidence["clarity_signals"],
                    "exam_question": top_exam.get("question", ""),
                },
                "teaching_actions": teaching_actions(evidence, exam_weight, exam_frequency),
            },
        )
        created.append(gap)

    brief = compose_lesson_brief(lecture, stats, created)
    lecture.status = Lecture.Status.AUDITED
    lecture.save(update_fields=["lesson_brief", "status"])
    return {"stats": stats, "gap_ids": [gap.id for gap in created], "lesson_brief": brief}
