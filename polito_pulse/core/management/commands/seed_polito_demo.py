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
        lecture, _ = Lecture.objects.update_or_create(
            course=course,
            number=14,
            defaults={
                "title": "Nyquist Stability and Frequency-Domain Margins",
                "date": date(2026, 11, 6),
                "status": Lecture.Status.AUDITED,
                "checkin_questions": [
                    "Why does the contour indent around imaginary-axis poles?",
                    "What does N = Z - P mean for closed-loop stability?",
                    "How is phase margin related to damping?",
                ],
            },
        )

        for number in range(1, 33):
            nyquist = number <= 20
            topic = "Nyquist Contour Indentation" if nyquist else "Bode Phase Margin"
            text = (
                "Nyquist contour, imaginary-axis pole indentation, and N = Z - P."
                if nyquist
                else "Bode gain crossover, phase margin, and closed-loop damping."
            )
            SlideChunk.objects.update_or_create(
                lecture=lecture,
                slide_number=number,
                defaults={
                    "topic": topic,
                    "text": text,
                    "latex_symbols": (
                        [r"j\omega", r"\varepsilon \to 0^+", "N = Z - P"]
                        if nyquist
                        else [r"\omega_c", r"\varphi_m", r"G(j\omega)"]
                    ),
                    "time_spent_seconds": 1 if number <= 28 else 0,
                    "best_similarity": 0.94 if number in (18, 24) else 0.0,
                },
            )

        transcript_rows = [
            (
                1,
                2472000,
                "The Nyquist contour must detour around poles on the imaginary axis.",
                18,
                0.94,
            ),
            (
                2,
                2498000,
                "Count clockwise encirclements of minus one and reconcile N equals Z minus P.",
                19,
                0.91,
            ),
            (
                3,
                2524000,
                "At omega c the Bode phase margin predicts relative damping.",
                24,
                0.96,
            ),
        ]
        for seq, start, raw, slide, score in transcript_rows:
            TranscriptSegment.objects.update_or_create(
                lecture=lecture,
                seq=seq,
                defaults={
                    "start_ms": start,
                    "duration_ms": 26000,
                    "raw_text": raw,
                    "cleaned_text": raw.replace(
                        "N equals Z minus P", "$N = Z - P$"
                    ).replace("omega c", "$\\omega_c$"),
                    "matched_slide_number": slide,
                    "similarity_score": score,
                },
            )

        StudentCheckIn.objects.filter(lecture=lecture).delete()
        responses = [
            (0.31, "Imaginary-axis pole indentation"),
            (0.55, "Clockwise encirclement sign"),
            (0.62, "Imaginary-axis pole indentation"),
            (0.78, ""),
            (0.85, ""),
        ]
        for index, (score, misconception) in enumerate(responses, start=1):
            StudentCheckIn.objects.create(
                lecture=lecture,
                student_ref=f"demo-{index:02d}",
                answers={"q1": "recap", "q2": "confused", "q3": "clear"},
                score=score,
                misconception=misconception,
            )

        run_gap_audit(lecture.id)
        nyquist_gap = lecture.gaps.get(topic="Nyquist Contour Indentation")
        bode_gap = lecture.gaps.get(topic="Bode Phase Margin")
        nyquist_gap.quiz_score = 0.31
        nyquist_gap.audio_time_seconds = 45
        nyquist_gap.slide_number = 18
        nyquist_gap.severity = GapAlert.Severity.CRITICAL
        nyquist_gap.manim_status = GapAlert.ManimStatus.IDLE
        nyquist_gap.save()
        bode_gap.quiz_score = 0.62
        bode_gap.audio_time_seconds = 72
        bode_gap.slide_number = 24
        bode_gap.severity = GapAlert.Severity.WARNING
        bode_gap.manim_status = GapAlert.ManimStatus.COMPLETED
        bode_gap.save()

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
                "published": bool(video_url),
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
