import io
import math
import os
from datetime import date
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from agents.dashboard_agent import compute_stats, run_gap_audit, severity
from agents.listener_agent import should_run_listener
from agents.llm_client import deterministic_embedding, is_claude_configured, is_groq_configured
from core.models import Course, Lecture, SlideChunk


class FallbackClientTests(TestCase):
    def test_deterministic_embedding_is_normalized_and_stable(self):
        first = deterministic_embedding("Nyquist")
        second = deterministic_embedding("Nyquist")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1536)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0, places=6)

    def test_placeholder_key_is_not_configured(self):
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "sk-ant-PENDING", "GROQ_API_KEY": ""},
            clear=False,
        ):
            self.assertFalse(is_claude_configured())
            self.assertFalse(is_groq_configured())


class PreservedAgentRulesTests(TestCase):
    def test_listener_window_and_final_flush(self):
        self.assertFalse(should_run_listener(29_999))
        self.assertTrue(should_run_listener(30_000))
        self.assertTrue(should_run_listener(1, final=True))

    def test_analyst_severity_thresholds(self):
        self.assertEqual(severity(10, 0.4, 4.0), "high")
        self.assertEqual(severity(10, 0.2, 4.0), "medium")
        self.assertEqual(severity(10, 0.1, 4.0), "low")
        self.assertIsNone(severity(0, 0.0, None))


class ProductViewTests(TestCase):
    def setUp(self):
        course = Course.objects.create(code="01QOROV", title="Automatic Control")
        self.lecture = Lecture.objects.create(
            course=course,
            number=14,
            title="Nyquist",
            date=date(2026, 11, 6),
        )

    def test_login_and_demo_auth(self):
        self.assertEqual(self.client.get(reverse("core:login")).status_code, 200)
        response = self.client.post(reverse("core:demo_login", args=["student"]))
        self.assertRedirects(response, reverse("core:student_dashboard"))
        self.assertContains(self.client.get(reverse("core:login")), "PoliBrige")

    def test_logout_clears_session_and_returns_to_login(self):
        self.client.post(reverse("core:demo_login", args=["professor"]))
        self.assertEqual(self.client.session["role"], "professor")
        response = self.client.post(reverse("core:logout"))
        self.assertRedirects(response, reverse("core:login"))
        self.assertNotIn("role", self.client.session)
        studio = self.client.get(reverse("core:professor_live"))
        self.assertContains(studio, "Log out")

    def test_dashboard_shows_lesson_brief_before_statistics(self):
        SlideChunk.objects.create(
            lecture=self.lecture,
            slide_number=18,
            topic="Nyquist Contour Indentation",
            text="Indent around poles on the imaginary axis.",
            time_spent_seconds=40,
        )
        brief = {
            "summary": "Nyquist was rushed around the indentation.",
            "improvements": ["Slow the epsilon limit before counting encirclements."],
            "animations": [
                {
                    "title": "Nyquist indentation",
                    "why": "Show the semicircle mapping around -1.",
                }
            ],
        }
        with patch("agents.dashboard_agent.chat_json", return_value=brief):
            run_gap_audit(self.lecture.id)

        self.lecture.refresh_from_db()
        self.assertEqual(self.lecture.lesson_brief["summary"], brief["summary"])
        response = self.client.get(reverse("core:professor_dashboard", args=[self.lecture.id]))
        content = response.content.decode()
        self.assertLess(content.index("Lesson analysis"), content.index("Slide Coverage"))
        self.assertContains(response, "Nyquist was rushed around the indentation.")
        self.assertContains(response, "Slow the epsilon limit")
        self.assertContains(response, "Nyquist indentation")

    def test_professor_live_renders(self):
        response = self.client.get(reverse("core:professor_live"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Live Transcript Stream")


class TeachingFeedbackTests(TestCase):
    def setUp(self):
        course = Course.objects.create(code="01QOROV", title="Automatic Control")
        self.lecture = Lecture.objects.create(
            course=course, number=14, title="Nyquist", date=date(2026, 11, 6)
        )
        for number, topic in ((18, "Nyquist Contour Indentation"), (24, "Bode Phase Margin")):
            SlideChunk.objects.create(
                lecture=self.lecture, slide_number=number, topic=topic, text=topic
            )

    def test_listener_saves_signals_and_topic_tagged_questions(self):
        from agents.listener_agent import process_transcript_chunk

        process_transcript_chunk(
            lecture_id=self.lecture.id,
            raw_text="Student asks: why go around the pole? Quickly, let's move on.",
            seq=1, start_ms=0, duration_ms=15000,
        )
        process_transcript_chunk(
            lecture_id=self.lecture.id,
            raw_text="Phase margin at omega c. For example, take G of s equals ten over s.",
            seq=2, start_ms=15000, duration_ms=15000,
        )
        first, second = self.lecture.transcripts.order_by("seq")
        self.assertEqual(first.signals["topic"], "Nyquist Contour Indentation")
        self.assertEqual(first.signals["student_questions"], ["why go around the pole?"])
        self.assertTrue(first.signals["clarity_signals"])
        self.assertEqual(len(second.signals["examples_given"]), 1)

        self.lecture.refresh_from_db()
        topics = [item["topic"] for item in self.lecture.pulse_questions()]
        self.assertEqual(topics, ["Nyquist Contour Indentation", "Bode Phase Margin"])

        stats = compute_stats(self.lecture)
        by_topic = {topic["name"]: topic for topic in stats["topics"]}
        self.assertEqual(by_topic["Nyquist Contour Indentation"]["n_student_questions"], 1)
        self.assertEqual(by_topic["Bode Phase Margin"]["n_examples"], 1)

    def test_checkin_scores_each_topic_separately(self):
        self.lecture.checkin_questions = [
            {"text": "Why indent?", "topic": "Nyquist Contour Indentation"},
            {"text": "Phase margin?", "topic": "Bode Phase Margin"},
        ]
        self.lecture.save()
        url = reverse("core:submit_checkin", args=[self.lecture.id])
        self.client.post(url, {"q1": "confused", "q2": "clear"})
        checkin = self.lecture.checkins.get()
        self.assertEqual(
            checkin.topic_scores,
            {"Nyquist Contour Indentation": 0.25, "Bode Phase Margin": 1.0},
        )
        self.assertEqual(checkin.misconception, "Nyquist Contour Indentation")

        run_gap_audit(self.lecture.id)
        nyquist = self.lecture.gaps.get(topic="Nyquist Contour Indentation")
        bode = self.lecture.gaps.get(topic="Bode Phase Margin")
        self.assertEqual((nyquist.quiz_score, bode.quiz_score), (0.25, 1.0))
        self.assertEqual(nyquist.severity, "critical")

    def test_plan_merge_takes_llm_wording_once(self):
        from agents.dashboard_agent import _merge_plan

        fallback = {
            "recap": [{"topic": "Bode", "minutes": 5, "action": "A", "actions": ["A", "B"]}],
            "warm_up_question": "Q?",
        }
        self.assertEqual(_merge_plan(fallback, fallback)["recap"][0]["actions"], ["A", "B"])
        merged = _merge_plan(
            {"recap": [{"topic": "bode", "action": "LLM move", "minutes": 99}]}, fallback
        )
        self.assertEqual(merged["recap"][0]["actions"], ["LLM move", "A", "B"])
        self.assertEqual(merged["recap"][0]["minutes"], 5)

    def test_past_exam_fallback_is_topic_specific(self):
        from mcp_server.server import match_past_exams

        self.assertEqual(match_past_exams("Nyquist Contour Indentation")[0]["point_weight"], 8)
        self.assertEqual(match_past_exams("Bode Phase Margin")[0]["point_weight"], 6)
        self.assertEqual(match_past_exams("Root Locus"), [])


class SeededDemoTests(TestCase):
    def setUp(self):
        from django.core.management import call_command

        with patch("core.management.commands.seed_polito_demo._synthetic_fallback_video",
                   return_value="/media/animations/fallback/test.mp4"):
            call_command("seed_polito_demo", stdout=io.StringIO(), stderr=io.StringIO())
        self.lecture = Lecture.objects.get(number=14)

    def test_audit_ranks_topics_from_evidence(self):
        severities = {gap.topic: gap.severity for gap in self.lecture.gaps.all()}
        self.assertEqual(
            severities,
            {
                "Nyquist Plot Construction": "optimal",
                "Nyquist Contour Indentation": "critical",
                "Bode Phase Margin": "warning",
                "Gain Margin": "warning",
            },
        )
        indentation = self.lecture.gaps.get(topic="Nyquist Contour Indentation")
        self.assertEqual(indentation.audio_time_seconds, 45)
        self.assertEqual(len(indentation.evidence["student_questions"]), 2)
        self.assertTrue(any("worked example" in a for a in indentation.teaching_actions))

    def test_next_lecture_plan_recaps_gaps_and_carries_over_skipped_slides(self):
        plan = self.lecture.lesson_brief["next_lecture"]
        self.assertEqual(
            [item["topic"] for item in plan["recap"]],
            ["Nyquist Contour Indentation", "Bode Phase Margin"],
        )
        self.assertEqual(plan["carry_over"], [{"topic": "Gain Margin", "slides": "29–32"}])
        response = self.client.get(reverse("core:next_lecture_plan", args=[self.lecture.id]))
        self.assertContains(response, "Plan for Lecture 15")
        self.assertContains(response, "Why do we go around the pole and not through it?")

    def test_dashboard_shows_followup_on_previous_plan(self):
        response = self.client.get(reverse("core:professor_dashboard", args=[self.lecture.id]))
        self.assertContains(response, "Did Lecture 13")
        self.assertContains(response, "34%")
        self.assertContains(response, "Teaching actions for next lecture")

    def test_student_form_shows_question_text(self):
        response = self.client.get(reverse("core:student_dashboard"))
        self.assertContains(response, "Why must the Nyquist contour indent")
        self.assertContains(response, 'name="topic_q2" value="Nyquist Contour Indentation"')
