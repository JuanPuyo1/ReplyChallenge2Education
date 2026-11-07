"""End-to-end test on a real Postgres with a fake Groq client.

    DATABASE_URL=postgresql://... pytest -q
"""
import json
import re
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from app import db, llm
from app.main import app

SCRIPT = [  # what the "teacher" says in each 30 s chunk
    "Today we start with the perceptron, weights and bias. For example an AND gate.",
    "The activation function decides the output. Any questions? Student: why a step?",
    "Now backpropagation uses the chain rule, I'll go quickly, this part is hard.",
    "Overfitting happens when we memorize. Thanks for watching",  # hallucinated tail
]


class FakeGroq:
    def __init__(self):
        self.calls = {"stt": 0, "listener": 0, "analyst": 0}
        self.audio = NS(transcriptions=NS(create=self._stt))
        self.chat = NS(completions=NS(create=self._chat))
        self.bad_once = True

    def _stt(self, **kw):
        i = self.calls["stt"]; self.calls["stt"] += 1
        assert kw["model"] == "whisper-large-v3-turbo"
        if i > 0:
            assert "prompt" in kw and len(kw["prompt"]) <= 700
        text, segs = SCRIPT[i], []
        for part in re.split(r"(?<=\.)\s+", text):
            segs.append({"text": part, "no_speech_prob": 0.01, "avg_logprob": -0.2, "compression_ratio": 1.3})
        return NS(model_dump=lambda: {"text": text, "segments": segs})

    def _chat(self, model, messages, **kw):
        user = messages[1]["content"]
        if model.endswith("-20b"):
            self.calls["listener"] += 1
            plan = json.loads(user.split("PLAN (in teaching order):\n")[1].split("\n\n")[0])
            window = user.split('"""')[1].lower()
            tid = next((t["id"] for t in plan if t["name"].lower() in window), None)
            out = {"topic_id": tid, "summary": window[:60], "key_concepts": ["x"],
                   "examples_given": ["AND gate"] if "example" in window else [],
                   "student_questions": ["why a step?"] if "student:" in window else [],
                   "clarity_signals": ["teacher rushing"] if "quickly" in window else [],
                   "confidence": 0.9}
        else:
            self.calls["analyst"] += 1
            if self.bad_once:  # exercise the validation-retry path
                self.bad_once = False
                return NS(choices=[NS(message=NS(content='{"overall_verdict": "great"}'))])
            ids = json.loads(user.split("TOPICS_TO_DIAGNOSE (worst first): ")[1].split("\n")[0])
            out = {"overall_verdict": "mixed", "headline": "Solid start, rushed finish.",
                   "strengths": ["clear perceptron intro"], "weaknesses": ["backprop rushed"],
                   "diagnoses": [{"topic_id": i, "likely_causes": ["rushed"], "suggestions": ["worked example"]} for i in ids]
                   + [{"topic_id": 999999, "likely_causes": ["bogus"], "suggestions": []}],
                   "comment_themes": ["wants more examples"], "next_lesson_plan": ["recap backprop"]}
        return NS(choices=[NS(message=NS(content=json.dumps(out)))])


def test_full_pipeline():
    fake = FakeGroq()
    llm.set_client(fake)
    with TestClient(app) as client:
        lesson = client.post("/api/lessons", json={"title": "Intro to NN", "language": "en", "topics": [
            {"name": "Perceptron"}, {"name": "Backpropagation"}, {"name": "Overfitting"}]}).json()
        lid = lesson["id"]
        topic = {t["name"]: t["id"] for t in lesson["topics"]}

        # 409 before any feedback
        assert client.post(f"/api/lessons/{lid}/analyze").status_code == 409

        for i in range(4):
            r = client.post(f"/api/lessons/{lid}/chunks",
                            data={"seq": i, "start_ms": i * 30000, "duration_ms": 30000, "final": str(i == 3).lower()},
                            files={"audio": (f"c{i}.webm", b"\x1aE\xdf\xa3" + b"0" * 5000, "audio/webm")}).json()
            assert r["annotation"] is not None  # 30 s window -> Listener on every chunk
        # tiny blob is skipped without calling Whisper
        client.post(f"/api/lessons/{lid}/chunks", data={"seq": 4, "start_ms": 120000, "duration_ms": 200},
                    files={"audio": ("c4.webm", b"x" * 100, "audio/webm")})
        assert fake.calls["stt"] == 4 and fake.calls["listener"] == 4

        tr = client.get(f"/api/lessons/{lid}/transcript").json()
        assert "Thanks for watching" not in tr["segments"][3]["text"]
        assert [a["topic_id"] for a in tr["annotations"]] == [topic["Perceptron"], None, topic["Backpropagation"], topic["Overfitting"]]

        # Feedback: Perceptron understood, Backprop not, Overfitting middling.
        ratings = {"Perceptron": [5, 5, 4, 4, 5, 4, 5, 4, 5, 5], "Backpropagation": [1, 2, 2, 1, 3, 2, 1, 2, 4, 2],
                   "Overfitting": [3, 3, 2, 4, 3, 3, 2, 4, 3, 3]}
        with db.pool().connection() as c:
            for s in range(10):
                rid = c.execute("INSERT INTO feedback_responses (lesson_id, overall_rating, pace_rating, comment) "
                                "VALUES (%s,%s,%s,%s) RETURNING id", (lid, [4, 3][s % 2], 4, "more examples" if s < 3 else None)).fetchone()["id"]
                for name, vals in ratings.items():
                    c.execute("INSERT INTO feedback_topic_ratings VALUES (%s,%s,%s)", (rid, topic[name], vals[s]))

        out = client.post(f"/api/lessons/{lid}/analyze").json()
        rep, stats = out["report"], out["stats"]
        att = rep["topics_needing_attention"]
        assert [t["topic"] for t in att] == ["Backpropagation", "Overfitting"]
        assert att[0]["severity"] == "high" and att[1]["severity"] == "medium"
        assert att[0]["evidence"][0].startswith("8/10 students (80%)")
        assert att[0]["likely_causes"] == ["rushed"]
        assert rep["avg_overall_rating"] == 3.5 and rep["n_responses"] == 10
        pt = {t["name"]: t for t in stats["topics"]}
        assert pt["Perceptron"]["minutes_taught"] == 0.5 and pt["Perceptron"]["n_examples"] == 1
        assert pt["Backpropagation"]["clarity_signals"] == ["teacher rushing"]
        assert fake.calls["analyst"] == 2  # one bad JSON + one retry

        md = client.get(f"/api/lessons/{lid}/report.md").text
        assert "## Topics needing attention" in md and "Backpropagation — high" in md
        print("\n" + md)
