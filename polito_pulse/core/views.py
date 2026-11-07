from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path

import qrcode
from django.db.models import Avg
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from agents.dashboard_agent import compute_stats, display_lesson_brief
from agents.listener_agent import process_transcript_chunk
from core.models import GapAlert, Lecture, StudentCheckIn
from core.tasks import submit_audit_job, submit_manim_job
from ingestion.pdf_processor import index_pdf
from ingestion.stt_stream import process_audio_chunk
from mcp_server.server import get_student_pulse


def _current_lecture() -> Lecture:
    return Lecture.objects.select_related("course").order_by("-date", "-number").first()


def home(_request: HttpRequest) -> HttpResponse:
    return redirect("core:login")


DEMO_ACCOUNTS = {
    "professor@polito.it": {
        "password": "pulse-prof",
        "role": "professor",
    },
    "student@polito.it": {
        "password": "pulse-student",
        "role": "student",
    },
}


@require_http_methods(["GET", "POST"])
def login_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        account = DEMO_ACCOUNTS.get(email)
        if account and account["password"] == password:
            return demo_login(request, account["role"])
        role = request.POST.get("role", "professor")
        if not email and not password:
            return demo_login(request, role)
        return render(
            request,
            "auth/login.html",
            {
                "error": "Use professor@polito.it / pulse-prof or student@polito.it / pulse-student.",
                "role": role,
            },
            status=401,
        )
    return render(request, "auth/login.html", {"role": "professor"})


@require_POST
def logout_view(request: HttpRequest) -> HttpResponse:
    request.session.flush()
    return redirect("core:login")


@require_POST
def demo_login(request: HttpRequest, role: str) -> HttpResponse:
    role = "student" if role == "student" else "professor"
    request.session["role"] = role
    return redirect(
        "core:student_dashboard" if role == "student" else "core:professor_live"
    )


@require_GET
def professor_live(request: HttpRequest) -> HttpResponse:
    lecture = _current_lecture()
    if not lecture:
        return HttpResponse("Run seed_polito_demo first.", status=503)
    pulse = get_student_pulse(lecture.id)
    return render(
        request,
        "professor/live_studio.html",
        {
            "lecture": lecture,
            "segments": lecture.transcripts.all(),
            "active_slide": lecture.slides.filter(slide_number=18).first()
            or lecture.slides.first(),
            "pulse": pulse,
        },
    )


@require_POST
def audio_chunk(request: HttpRequest) -> HttpResponse:
    lecture = get_object_or_404(Lecture, pk=int(request.POST.get("lecture_id", 0)))
    audio = request.FILES.get("audio")
    if not audio:
        return HttpResponse("Missing audio blob.", status=400)
    result = process_audio_chunk(
        lecture_id=lecture.id,
        blob=audio.read(),
        seq=int(request.POST.get("seq", lecture.transcripts.count() + 1)),
        start_ms=int(request.POST.get("start_ms", 0)),
        duration_ms=int(request.POST.get("duration_ms", 15000)),
    )
    segment = lecture.transcripts.get(pk=result["segment_id"])
    return render(
        request,
        "professor/partials/_transcript_chunk.html",
        {"segment": segment},
    )


@require_POST
def demo_chunk(request: HttpRequest) -> HttpResponse:
    lecture = get_object_or_404(Lecture, pk=int(request.POST.get("lecture_id", 0)))
    lines = [
        "The Nyquist contour must detour around poles on the imaginary axis as epsilon tends to zero.",
        "Count clockwise encirclements of minus one and reconcile them with N equals Z minus P.",
        "At omega c, phase margin links the Bode trace to closed-loop damping.",
    ]
    seq = lecture.transcripts.count() + 1
    raw = lines[(seq - 1) % len(lines)]
    result = process_transcript_chunk(
        lecture_id=lecture.id,
        raw_text=raw,
        seq=seq,
        start_ms=(seq - 1) * 15000,
        duration_ms=15000,
    )
    segment = lecture.transcripts.get(pk=result["segment_id"])
    return render(
        request,
        "professor/partials/_transcript_chunk.html",
        {"segment": segment},
    )


@require_POST
def upload_pdf(request: HttpRequest) -> HttpResponse:
    lecture = get_object_or_404(Lecture, pk=int(request.POST.get("lecture_id", 0)))
    uploaded = request.FILES.get("pdf")
    if not uploaded:
        return HttpResponse("Choose a PDF first.", status=400)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        for chunk in uploaded.chunks():
            handle.write(chunk)
        temp_path = Path(handle.name)
    try:
        indexed = index_pdf(
            temp_path,
            collection="course_slides",
            lecture_id=lecture.id,
            course_code=lecture.course.code,
        )
    finally:
        temp_path.unlink(missing_ok=True)
    response = HttpResponse(f"{len(indexed)} slide chunks synchronized.")
    response["HX-Trigger"] = json.dumps(
        {"quizSubmitted": {"message": f"{len(indexed)} slide chunks synchronized"}}
    )
    return response


@require_GET
def qr_modal(request: HttpRequest) -> HttpResponse:
    lecture = _current_lecture()
    url = request.build_absolute_uri(reverse("core:student_dashboard"))
    image = qrcode.make(url)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return render(
        request,
        "professor/partials/_qr_modal.html",
        {"lecture": lecture, "qr_data": encoded, "checkin_url": url},
    )


@require_GET
def student_pulse(request: HttpRequest, lecture_id: int) -> HttpResponse:
    return render(
        request,
        "professor/partials/_student_pulse.html",
        {"pulse": get_student_pulse(lecture_id), "lecture_id": lecture_id},
    )


@require_POST
def end_audit(request: HttpRequest, lecture_id: int) -> HttpResponse:
    get_object_or_404(Lecture, pk=lecture_id)
    submit_audit_job(lecture_id)
    response = HttpResponse(status=204)
    response["HX-Redirect"] = reverse("core:professor_dashboard", args=[lecture_id])
    response["HX-Trigger"] = json.dumps(
        {"auditComplete": {"message": "Gap audit started"}}
    )
    return response


@require_GET
def professor_dashboard(request: HttpRequest, lecture_id: int) -> HttpResponse:
    lecture = get_object_or_404(
        Lecture.objects.select_related("course"), pk=lecture_id
    )
    stats = compute_stats(lecture)
    critical_count = lecture.gaps.filter(severity=GapAlert.Severity.CRITICAL).count()
    return render(
        request,
        "professor/dashboard.html",
        {
            "lecture": lecture,
            "gaps": lecture.gaps.all(),
            "analysis": display_lesson_brief(lecture, stats),
            "stats": stats,
            "critical_count": critical_count,
            "pulse_percent": round((stats["quiz_average"] or 0) * 100),
        },
    )


@require_POST
def animate_gap(request: HttpRequest, gap_id: int) -> HttpResponse:
    gap = get_object_or_404(GapAlert, pk=gap_id)
    gap.manim_status = GapAlert.ManimStatus.EXTRACTING
    gap.task_error = ""
    gap.save(update_fields=["manim_status", "task_error", "updated_at"])
    submit_manim_job(gap.id)
    return render(request, "professor/partials/_manim_status.html", {"gap": gap})


@require_GET
def gap_status(request: HttpRequest, gap_id: int) -> HttpResponse:
    gap = get_object_or_404(GapAlert, pk=gap_id)
    response = render(request, "professor/partials/_manim_status.html", {"gap": gap})
    if gap.manim_status == GapAlert.ManimStatus.COMPLETED:
        response["HX-Trigger"] = json.dumps(
            {"manimReady": {"message": f"{gap.title} published to students"}}
        )
    return response


@require_GET
def student_dashboard(request: HttpRequest) -> HttpResponse:
    lecture = _current_lecture()
    return render(
        request,
        "student/dashboard.html",
        {"lecture": lecture, "animations": lecture.animations.filter(published=True)},
    )


@require_GET
def student_day_content(request: HttpRequest) -> HttpResponse:
    lecture = _current_lecture()
    return render(
        request,
        "student/partials/_lecture_day_content.html",
        {"lecture": lecture, "animations": lecture.animations.filter(published=True)},
    )


@require_POST
def submit_checkin(request: HttpRequest, lecture_id: int) -> HttpResponse:
    lecture = get_object_or_404(Lecture, pk=lecture_id)
    mapping = {"clear": 1.0, "recap": 0.55, "confused": 0.25}
    answers = {
        key: request.POST.get(key, "recap") for key in ("q1", "q2", "q3")
    }
    score = sum(mapping.get(answer, 0.55) for answer in answers.values()) / 3
    misconception = (
        "Imaginary-axis pole indentation"
        if any(answer == "confused" for answer in answers.values())
        else ""
    )
    StudentCheckIn.objects.create(
        lecture=lecture,
        student_ref=request.session.session_key or "demo-student",
        answers=answers,
        score=score,
        misconception=misconception,
    )
    response = render(
        request,
        "student/partials/_quiz_form.html",
        {"submitted": True, "score": round(score * 100)},
    )
    response["HX-Trigger"] = json.dumps(
        {"quizSubmitted": {"message": "Your class pulse was recorded"}}
    )
    return response
