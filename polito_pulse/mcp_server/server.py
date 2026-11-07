from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "polito_pulse.settings")

from agents.llm_client import qdrant_search  # noqa: E402

mcp = FastMCP("PoliTo-Pulse-MCP")


def _slide_fallback() -> list[dict[str, Any]]:
    return [
        {
            "text": "Nyquist criterion with imaginary-axis pole indentation; N = Z - P.",
            "slide_number": 18,
            "latex_symbols": [r"j\omega", r"\varepsilon \to 0^+", "N=Z-P"],
            "topic": "Nyquist Contour Indentation",
            "score": 0.94,
        }
    ]


@mcp.tool
def search_course_slides(query: str, lecture_id: int = 1, top_k: int = 3) -> list[dict]:
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        hits = qdrant_search(
            "course_slides",
            query,
            limit=top_k,
            query_filter=Filter(
                must=[FieldCondition(key="lecture_id", match=MatchValue(value=lecture_id))]
            ),
        )
        return hits or _slide_fallback()
    except Exception:
        return _slide_fallback()


@mcp.tool
def match_past_exams(topic_query: str, top_k: int = 3) -> list[dict]:
    try:
        hits = qdrant_search("past_exams", topic_query, limit=top_k)
        if hits:
            return hits
    except Exception:
        pass
    return [
        {
            "frequency": "Appears in 4/5 Past Appelli",
            "question": "Apply Nyquist with poles on the imaginary axis.",
            "point_weight": 8,
            "year": 2025,
            "question_number": 3,
            "score": 0.91,
        }
    ]


@mcp.tool
def get_student_pulse(lecture_id: int) -> dict:
    import django

    django.setup()
    from django.db.models import Avg, Count

    from core.models import StudentCheckIn

    query = StudentCheckIn.objects.filter(lecture_id=lecture_id)
    summary = query.aggregate(count=Count("id"), average=Avg("score"))
    misconceptions: dict[str, int] = {}
    for value in query.exclude(misconception="").values_list("misconception", flat=True):
        misconceptions[value] = misconceptions.get(value, 0) + 1
    return {
        "students_connected": summary["count"] or 0,
        "comprehension_percent": round((summary["average"] or 0) * 100),
        "top_misconceptions": [
            item[0] for item in sorted(misconceptions.items(), key=lambda item: -item[1])[:3]
        ],
    }


@mcp.tool
def search_manim_templates(concept_query: str, top_k: int = 3) -> list[dict]:
    try:
        hits = qdrant_search("manim_templates", concept_query, limit=top_k)
        if hits:
            return hits
    except Exception:
        pass
    return [
        {
            "topic": "Nyquist",
            "code": (
                "axes = Axes(x_range=[-3,3], y_range=[-3,3])\n"
                "curve = ParametricFunction(lambda t: [np.cos(t), np.sin(t), 0], "
                "t_range=[0, TAU])"
            ),
            "score": 0.88,
        }
    ]


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", value).strip("-")[:80] or "polito-scene"


@mcp.tool
def compile_manim_scene(python_code: str, scene_name: str, output_slug: str) -> dict:
    slug = _safe_slug(output_slug)
    safe_scene = re.sub(r"[^a-zA-Z0-9_]", "", scene_name)
    if not safe_scene:
        return {"status": "error", "traceback": "Invalid scene name."}

    media_root = PROJECT_ROOT / "media" / "animations"
    temp_root = media_root / "tmp"
    generated_root = media_root / "generated"
    temp_root.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)
    script_path = temp_root / f"{slug}.py"
    script_path.write_text(python_code, encoding="utf-8")

    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "manim",
                "-ql",
                "--media_dir",
                str(generated_root),
                str(script_path),
                safe_scene,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "error", "traceback": str(exc)}

    if process.returncode:
        return {"status": "error", "traceback": process.stderr[-6000:]}

    candidates = sorted(
        generated_root.rglob(f"{safe_scene}.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"status": "error", "traceback": "Manim exited successfully but no MP4 was found."}
    relative = candidates[0].relative_to(PROJECT_ROOT / "media").as_posix()
    return {"status": "success", "video_url": f"/media/{relative}"}


if __name__ == "__main__":
    mcp.run()
