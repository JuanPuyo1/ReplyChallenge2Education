import math
import os
from datetime import date
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from agents.dashboard_agent import severity
from agents.listener_agent import should_run_listener
from agents.llm_client import deterministic_embedding, is_claude_configured, is_groq_configured
from core.models import Course, Lecture


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

    def test_professor_live_renders(self):
        response = self.client.get(reverse("core:professor_live"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Live Transcript Stream")
