"""Agent 2: the Lesson Analyst.

Joins what was taught (Listener annotations) with what students reported (form results in
Postgres) and writes a report: how good the lesson was, and which topics need more
attention and why.

Division of labour: all numbers, rankings and severities are computed here in Python/SQL,
so they are exact and reproducible. The LLM only does what needs language: reading
comments, explaining likely causes, and suggesting fixes."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from . import config, db
from .llm import call_json


class NoFeedbackYet(Exception):
    pass


# ------------------------------------------------------------------ stats ---

def severity(n: int, pct_low: float, avg: float | None) -> str | None:
    if n == 0 or avg is None:
        return None
    if pct_low >= 0.40 or avg < 2.5:
        return "high"
    if pct_low >= 0.20 or avg < 3.3:
        return "medium"
    return "low"


def compute_stats(lesson: dict) -> dict:
    overall = db.feedback_overall(lesson["id"])
    per_topic = db.feedback_per_topic(lesson["id"], config.LOW_UNDERSTANDING)
    annotations = db.all_annotations(lesson["id"])

    taught = defaultdict(lambda: {"ms": 0, "windows": 0, "concepts": [], "examples": [],
                                  "questions": [], "clarity": [], "summaries": []})
    lesson_ms = sum(a["duration_ms"] for a in annotations) or 0
    off_plan = []
    for a in annotations:
        if a["topic_id"] is None:
            if a.get("off_plan_topic"):
                off_plan.append({"topic": a["off_plan_topic"], "minutes": round(a["duration_ms"] / 60000, 1)})
            continue
        t = taught[a["topic_id"]]
        t["ms"] += a["duration_ms"]
        t["windows"] += 1
        t["concepts"] += a["key_concepts"]
        t["examples"] += a["examples_given"]
        t["questions"] += a["student_questions"]
        t["clarity"] += a["clarity_signals"]
        t["summaries"].append(a["summary"])

    topics = []
    for row in per_topic:
        t = taught.get(row["topic_id"], taught.default_factory())
        n = row["n_ratings"]
        pct_low = round(row["n_low"] / n, 3) if n else 0.0
        topics.append({
            "topic_id": row["topic_id"],
            "name": row["name"],
            "n_ratings": n,
            "avg_understanding": row["avg_understanding"],
            "n_low": row["n_low"],
            "pct_low": pct_low,
            "severity": severity(n, pct_low, row["avg_understanding"]),
            "minutes_taught": round(t["ms"] / 60000, 1),
            "share_of_lesson": round(t["ms"] / lesson_ms, 3) if lesson_ms else None,
            "covered": t["ms"] > 0,
            "n_examples": len(t["examples"]),
            "n_student_questions": len(t["questions"]),
            "key_concepts": list(dict.fromkeys(t["concepts"]))[:15],
            "examples": t["examples"][:8],
            "student_questions": t["questions"][:8],
            "clarity_signals": t["clarity"][:8],
            "summaries": t["summaries"][:12],
        })

    rank = {"high": 0, "medium": 1, "low": 2, None: 3}
    attention = sorted((t for t in topics if t["severity"] in ("high", "medium")),
                       key=lambda t: (rank[t["severity"]], -t["pct_low"], t["avg_understanding"]))
    return {
        "lesson_id": lesson["id"],
        "title": lesson["title"],
        "overall": overall,
        "lesson_minutes_annotated": round(lesson_ms / 60000, 1),
        "off_plan": off_plan,
        "topics": topics,
        "attention_order": [t["topic_id"] for t in attention],
        "low_understanding_threshold": config.LOW_UNDERSTANDING,
    }


def evidence_lines(t: dict, threshold: int) -> list[str]:
    lines = [f"{t['n_low']}/{t['n_ratings']} students ({t['pct_low']:.0%}) rated understanding "
             f"<= {threshold}/5; average {t['avg_understanding']}/5."]
    if not t["covered"]:
        lines.append("The transcript shows no time spent on this topic.")
    else:
        share = f" ({t['share_of_lesson']:.0%} of the lesson)" if t["share_of_lesson"] is not None else ""
        lines.append(f"Taught for about {t['minutes_taught']} min{share}, "
                     f"{t['n_examples']} example(s), {t['n_student_questions']} student question(s).")
    return lines


# -------------------------------------------------------------------- LLM ---

class TopicDiagnosis(BaseModel):
    topic_id: int
    likely_causes: list[str] = Field(description="why students struggled, grounded in the data")
    suggestions: list[str] = Field(description="concrete actions for the next lesson")


class AnalystLLMOutput(BaseModel):
    overall_verdict: Literal["good", "mixed", "poor"]
    headline: str = Field(description="one sentence verdict for the teacher")
    strengths: list[str]
    weaknesses: list[str]
    diagnoses: list[TopicDiagnosis] = Field(
        description="exactly one entry per topic id in TOPICS_TO_DIAGNOSE")
    comment_themes: list[str] = Field(description="recurring themes in student comments")
    next_lesson_plan: list[str] = Field(description="3-6 ordered steps for the next lesson")


SYSTEM = """You are an instructional analyst helping a university teacher improve a lesson.
You receive exact statistics from the student feedback form and per-topic notes from an
agent that listened to the lesson. Rules:
- Never invent or recompute numbers; quote the ones given.
- Ground every cause in evidence: little time, no examples, unanswered questions,
  clarity signals, topic not covered, or what the comments say.
- Distinguish "not taught enough" from "taught but explained poorly".
- Be direct and specific; this is for the teacher, not the students.
- Verdict guide: good = avg overall >= 4 and no high-severity topics; poor = avg overall
  < 3 or most topics high severity; otherwise mixed.
Write in the language of the lesson transcript notes."""


def run(lesson_id: int) -> dict:
    lesson = db.get_lesson(lesson_id)
    if lesson is None:
        raise KeyError(lesson_id)
    stats = compute_stats(lesson)
    if not stats["overall"]["n_responses"]:
        raise NoFeedbackYet(lesson_id)

    by_id = {t["topic_id"]: t for t in stats["topics"]}
    to_diagnose = stats["attention_order"]
    user = (
        f"LESSON: {lesson['title']}\n\n"
        f"OVERALL FEEDBACK:\n{json.dumps(stats['overall'], default=str)}\n"
        f"(pace: 1 = too slow, 3 = right, 5 = too fast)\n\n"
        f"OFF-PLAN CONTENT: {json.dumps(stats['off_plan'], ensure_ascii=False)}\n\n"
        f"TOPICS:\n{json.dumps(stats['topics'], ensure_ascii=False, default=str)}\n\n"
        f"TOPICS_TO_DIAGNOSE (worst first): {to_diagnose}\n\n"
        f"STUDENT COMMENTS:\n" + "\n".join(f"- {c}" for c in db.feedback_comments(lesson_id))
    )
    out = call_json(config.ANALYST_MODEL, SYSTEM, user, AnalystLLMOutput, reasoning_effort="medium")

    # Merge: severity/evidence/order come from code, causes/suggestions from the model.
    diag = {d.topic_id: d for d in out.diagnoses if d.topic_id in by_id}
    attention = []
    for tid in to_diagnose:
        t, d = by_id[tid], diag.get(tid)
        attention.append({
            "topic_id": tid,
            "topic": t["name"],
            "severity": t["severity"],
            "evidence": evidence_lines(t, config.LOW_UNDERSTANDING),
            "likely_causes": d.likely_causes if d else [],
            "suggestions": d.suggestions if d else [],
        })

    report = {
        "overall_verdict": out.overall_verdict,
        "headline": out.headline,
        "avg_overall_rating": stats["overall"]["avg_overall"],
        "n_responses": stats["overall"]["n_responses"],
        "strengths": out.strengths,
        "weaknesses": out.weaknesses,
        "topics_needing_attention": attention,
        "comment_themes": out.comment_themes,
        "next_lesson_plan": out.next_lesson_plan,
    }
    saved = db.save_report(lesson_id, stats, report)
    return {"report_id": saved["id"], "report": report, "stats": stats}


def to_markdown(report: dict) -> str:
    r = report
    md = [f"# {r['headline']}", "",
          f"**Verdict:** {r['overall_verdict']}  ·  **Avg rating:** {r['avg_overall_rating']}/5 "
          f"from {r['n_responses']} students", ""]
    if r["topics_needing_attention"]:
        md.append("## Topics needing attention")
        for t in r["topics_needing_attention"]:
            md.append(f"### {t['topic']} — {t['severity']}")
            md += [f"- {e}" for e in t["evidence"]] + [""]
            if t["likely_causes"]:
                md += ["**Likely causes:** " + "; ".join(t["likely_causes"]), ""]
            if t["suggestions"]:
                md.append("**Next time:** " + "; ".join(t["suggestions"]))
            md.append("")
    for title, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses"),
                       ("What students said", "comment_themes"), ("Next lesson plan", "next_lesson_plan")):
        if r[key]:
            md += [f"## {title}"] + [f"- {x}" for x in r[key]] + [""]
    return "\n".join(md)
