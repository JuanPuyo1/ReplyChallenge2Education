from __future__ import annotations

import uuid
from datetime import date
from threading import Thread

from django.core.management.base import BaseCommand
from django.utils import timezone

from agents.dashboard_agent import run_gap_audit
from agents.llm_client import (
    collection_vector_size,
    embed_text,
    ensure_collections,
    get_qdrant_client,
)
from agents.manim_agent import FALLBACK_SCENES, _synthetic_fallback_video
from core.models import (
    Course,
    GapAlert,
    Lecture,
    ManimAnimation,
    SlideChunk,
    StudentCheckIn,
    TranscriptSegment,
)


class Command(BaseCommand):
    help = "Seed the complete 01QOROV Lecture 14 hackathon demo."

    def handle(self, *args, **options):
        course, _ = Course.objects.update_or_create(
            code="01QOROV",
            defaults={
                "title": "Automatic Control",
                "term": "2026/27",
                "slide_count": 32,
                "past_exam_count": 5,
            },
        )
        self._seed_previous_lecture(course)
        lecture, _ = Lecture.objects.update_or_create(
            course=course,
            number=14,
            defaults={
                "title": "Nyquist Stability and Frequency-Domain Margins",
                "date": date(2026, 11, 6),
                "status": Lecture.Status.AUDITED,
                "checkin_questions": [
                    {
                        "text": "How do you trace $G(j\\omega)$ as $\\omega$ goes from $0$ to $\\infty$?",
                        "topic": "Nyquist Plot Construction",
                    },
                    {
                        "text": "Why must the Nyquist contour indent around a pole on the imaginary axis?",
                        "topic": "Nyquist Contour Indentation",
                    },
                    {
                        "text": "How does the phase margin $\\varphi_m$ relate to closed-loop damping?",
                        "topic": "Bode Phase Margin",
                    },
                ],
            },
        )

        # (topic, slides, per-slide second overrides, default seconds, text, symbols).
        # Gain Margin is never reached in the audio.
        topic_plan = [
            ("Nyquist Plot Construction", range(1, 16), {}, 36,
             "Trace G(jω) from ω = 0 to ∞ and mirror it for negative frequencies.",
             [r"G(j\omega)", r"\omega \in [0, \infty)"]),
            ("Nyquist Contour Indentation", range(16, 21), {18: 26, 19: 19}, 0,
             "Indent around imaginary-axis poles with a semicircle of radius ε; N = Z - P.",
             [r"j\omega", r"\varepsilon \to 0^+", "N = Z - P"]),
            ("Bode Phase Margin", range(21, 29), {}, 48,
             "Phase margin at the gain crossover ω_c measures relative stability and damping.",
             [r"\omega_c", r"\varphi_m", r"G(j\omega)"]),
            ("Gain Margin", range(29, 33), {}, 0,
             "Gain margin at the phase crossover ω_pc.",
             [r"\omega_{pc}", r"G_m"]),
        ]
        for topic, numbers, overrides, seconds, text, symbols in topic_plan:
            for number in numbers:
                SlideChunk.objects.update_or_create(
                    lecture=lecture,
                    slide_number=number,
                    defaults={
                        "topic": topic,
                        "text": text,
                        "latex_symbols": symbols,
                        "time_spent_seconds": overrides.get(number, seconds),
                        "best_similarity": 0.94 if number in (6, 18, 24) else 0.0,
                    },
                )
        lecture.gaps.exclude(topic__in=[row[0] for row in topic_plan]).delete()

        TranscriptSegment.objects.filter(lecture=lecture).delete()
        transcript_rows = [
            (600000, 6, 0.90,
             "To build the Nyquist plot, trace $G(j\\omega)$ from zero to infinity. "
             "For example, take $G(s) = \\frac{1}{s+1}$.",
             {"topic": "Nyquist Plot Construction",
              "examples_given": ["$G(s) = \\frac{1}{s+1}$ traced for $\\omega \\in [0, \\infty)$"]}),
            (2472000, 18, 0.94,
             "The Nyquist contour must detour around poles on the imaginary axis as "
             "$\\varepsilon \\to 0^+$.",
             {"topic": "Nyquist Contour Indentation",
              "student_questions": ["Why do we go around the pole and not through it?"],
              "clarity_signals": ["The $\\varepsilon$-semicircle mapping was derived in a single step."]}),
            (2498000, 19, 0.91,
             "Count clockwise encirclements of $-1$ and reconcile them with $N = Z - P$. "
             "Quickly, let's move on.",
             {"topic": "Nyquist Contour Indentation",
              "student_questions": ["Does the small semicircle map to a large arc clockwise or counter-clockwise?"],
              "clarity_signals": ["Professor moved on quickly; the step may need a recap."]}),
            (2524000, 24, 0.96,
             "At $\\omega_c$ the Bode phase margin predicts relative damping. "
             "For example, $G(s) = \\frac{10}{s(s+2)}$.",
             {"topic": "Bode Phase Margin",
              "examples_given": ["Phase margin of $G(s) = \\frac{10}{s(s+2)}$"]}),
        ]
        for seq, (start, slide, score, cleaned, signals) in enumerate(transcript_rows, start=1):
            TranscriptSegment.objects.create(
                lecture=lecture,
                seq=seq,
                start_ms=start,
                duration_ms=26000,
                raw_text=cleaned,
                cleaned_text=cleaned,
                matched_slide_number=slide,
                similarity_score=score,
                signals={
                    "examples_given": [],
                    "student_questions": [],
                    "clarity_signals": [],
                    **signals,
                },
            )

        StudentCheckIn.objects.filter(lecture=lecture).delete()
        values = {"clear": 1.0, "recap": 0.55, "confused": 0.25}
        questions = lecture.pulse_questions()
        # One row per student: answers to q1 (plot), q2 (indentation), q3 (Bode).
        responses = [
            ("clear", "confused", "recap"),
            ("clear", "confused", "recap"),
            ("recap", "recap", "clear"),
            ("clear", "confused", "recap"),
            ("clear", "recap", "recap"),
            ("recap", "confused", "recap"),
        ]
        for index, row in enumerate(responses, start=1):
            answers = {question["key"]: answer for question, answer in zip(questions, row)}
            topic_scores = {
                question["topic"]: values[answer] for question, answer in zip(questions, row)
            }
            confused = [q["topic"] for q, answer in zip(questions, row) if answer == "confused"]
            StudentCheckIn.objects.create(
                lecture=lecture,
                student_ref=f"demo-{index:02d}",
                answers=answers,
                score=sum(topic_scores.values()) / len(topic_scores),
                topic_scores=topic_scores,
                misconception=confused[0] if confused else "",
            )

        run_gap_audit(lecture.id)
        nyquist_gap = lecture.gaps.get(topic="Nyquist Contour Indentation")
        bode_gap = lecture.gaps.get(topic="Bode Phase Margin")
        nyquist_gap.manim_status = GapAlert.ManimStatus.IDLE
        nyquist_gap.save(update_fields=["manim_status", "updated_at"])

        self._seed_animation(lecture, nyquist_gap, "nyquist", completed=False)
        self._seed_animation(lecture, bode_gap, "bode", completed=True)
        qdrant_seed = Thread(
            target=self._seed_qdrant,
            args=(lecture,),
            daemon=True,
            name="polito-qdrant-seed",
        )
        qdrant_seed.start()
        qdrant_seed.join(timeout=20)
        if qdrant_seed.is_alive():
            self.stderr.write(
                "Qdrant synchronization timed out; local demo data is ready."
            )
        self.stdout.write(self.style.SUCCESS(f"Seeded {course} / Lecture {lecture.number}"))

    def _seed_previous_lecture(self, course: Course) -> None:
        """Lecture 13 flagged Bode Phase Margin and planned a recap; Lecture 14 shows the result."""
        previous, _ = Lecture.objects.update_or_create(
            course=course,
            number=13,
            defaults={
                "title": "Frequency Response and Bode Diagrams",
                "date": date(2026, 11, 4),
                "status": Lecture.Status.AUDITED,
                "lesson_brief": {
                    "next_lecture": {
                        "recap": [
                            {
                                "topic": "Bode Phase Margin",
                                "slides": "21–28",
                                "minutes": 5,
                                "severity": GapAlert.Severity.CRITICAL,
                                "why": "34% comprehension, 50s taught",
                                "action": "Read $\\varphi_m$ off the Bode plot on a worked example.",
                            }
                        ],
                        "carry_over": [],
                        "student_questions": [],
                        "warm_up_question": "Where on the Bode plot do you read the phase margin?",
                        "total_minutes": 5,
                    }
                },
            },
        )
        GapAlert.objects.update_or_create(
            lecture=previous,
            topic="Bode Phase Margin",
            defaults={
                "title": "Bode Phase Margin",
                "severity": GapAlert.Severity.CRITICAL,
                "audio_time_seconds": 50,
                "quiz_score": 0.34,
                "exam_weight": 0.6,
                "exam_frequency": "Appears in 3/5 Past Appelli",
                "slide_number": 24,
                "recommendation": "Animate the gain crossover and phase trace together.",
            },
        )

    def _seed_animation(
        self, lecture: Lecture, gap: GapAlert, scene_key: str, *, completed: bool
    ) -> None:
        scene_name, code = FALLBACK_SCENES[scene_key]
        slug = f"lecture-{lecture.id}-{scene_key}"
        try:
            video_url = _synthetic_fallback_video(slug)
        except Exception as exc:
            self.stderr.write(f"Fallback video generation skipped: {exc}")
            video_url = ""
        ManimAnimation.objects.update_or_create(
            gap=gap,
            defaults={
                "lecture": lecture,
                "title": gap.title,
                "slug": slug,
                "scene_name": scene_name,
                "python_code": code,
                "video_url": video_url,
                "latex_intuition": gap.recommendation,
                "exam_link": (
                    "Linked to 2025 Appello Q3"
                    if scene_key == "nyquist"
                    else "Linked to 2024 Appello Q2"
                ),
                "published": bool(video_url) and completed,
                "rendered_at": timezone.now() if video_url else None,
            },
        )
        if completed and video_url:
            gap.manim_status = GapAlert.ManimStatus.COMPLETED
            gap.save(update_fields=["manim_status", "updated_at"])

    def _seed_qdrant(self, lecture: Lecture) -> None:
        try:
            sizes = ensure_collections()
            client = get_qdrant_client()
            if not client:
                return
            from qdrant_client.models import PointStruct

            entries = {
                "course_slides": [
                    (
                        "slide-18",
                        "Nyquist contour indentation around imaginary-axis poles",
                        {
                            "lecture_id": lecture.id,
                            "course_code": lecture.course.code,
                            "slide_number": 18,
                            "topic": "Nyquist Contour Indentation",
                            "text": "Use an infinitesimal semicircle around jω poles and N = Z - P.",
                            "latex_symbols": [r"j\omega", r"\varepsilon \to 0^+", "N=Z-P"],
                        },
                    ),
                    (
                        "slide-24",
                        "Bode phase margin at gain crossover",
                        {
                            "lecture_id": lecture.id,
                            "course_code": lecture.course.code,
                            "slide_number": 24,
                            "topic": "Bode Phase Margin",
                            "text": "Phase margin at omega_c measures relative stability.",
                            "latex_symbols": [r"\omega_c", r"\varphi_m"],
                        },
                    ),
                ],
                "past_exams": [
                    (
                        "exam-2025-q3",
                        "Nyquist imaginary-axis poles",
                        {
                            "lecture_id": lecture.id,
                            "course_code": lecture.course.code,
                            "slide_number": 18,
                            "topic": "Nyquist Contour Indentation",
                            "frequency": "Appears in 4/5 Past Appelli",
                            "question": "Apply Nyquist with poles on the imaginary axis.",
                            "point_weight": 8,
                            "year": 2025,
                            "question_number": 3,
                        },
                    ),
                    (
                        "exam-2024-q2",
                        "Bode phase margin",
                        {
                            "lecture_id": lecture.id,
                            "course_code": lecture.course.code,
                            "slide_number": 24,
                            "topic": "Bode Phase Margin",
                            "frequency": "Appears in 3/5 Past Appelli",
                            "question": "Compute phase margin and infer damping.",
                            "point_weight": 6,
                            "year": 2024,
                            "question_number": 2,
                        },
                    ),
                ],
                "manim_templates": [
                    (
                        "template-nyquist",
                        "Nyquist contour animation",
                        {
                            "lecture_id": lecture.id,
                            "course_code": lecture.course.code,
                            "slide_number": 18,
                            "topic": "Nyquist Contour Indentation",
                            "code": FALLBACK_SCENES["nyquist"][1],
                        },
                    ),
                    (
                        "template-bode",
                        "Bode phase margin animation",
                        {
                            "lecture_id": lecture.id,
                            "course_code": lecture.course.code,
                            "slide_number": 24,
                            "topic": "Bode Phase Margin",
                            "code": FALLBACK_SCENES["bode"][1],
                        },
                    ),
                ],
            }
            for collection, rows in entries.items():
                points = [
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"polito:{key}")),
                        vector=embed_text(text, sizes.get(collection, collection_vector_size(collection))),
                        payload=payload,
                    )
                    for key, text, payload in rows
                ]
                client.upsert(collection_name=collection, points=points, wait=True)
            self.stdout.write(self.style.SUCCESS("Qdrant demo vectors synchronized"))
        except Exception as exc:
            self.stderr.write(f"Qdrant seed skipped; local demo remains available: {exc}")
