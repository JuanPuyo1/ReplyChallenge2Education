from django.urls import path

from . import views

app_name = "core"

urlpatterns = [
    path("", views.home, name="home"),
    path("login", views.login_view, name="login"),
    path("logout", views.logout_view, name="logout"),
    path("demo-login/<str:role>/", views.demo_login, name="demo_login"),
    path("professor/live/", views.professor_live, name="professor_live"),
    path(
        "professor/studio/<int:lecture_id>/",
        views.professor_studio,
        name="professor_studio",
    ),
    path("professor/live/audio-chunk/", views.audio_chunk, name="audio_chunk"),
    path("professor/live/demo-chunk/", views.demo_chunk, name="demo_chunk"),
    path("professor/live/upload-pdf/", views.upload_pdf, name="upload_pdf"),
    path("professor/live/qr/", views.qr_modal, name="qr_modal"),
    path("professor/live/pulse/<int:lecture_id>/", views.student_pulse, name="student_pulse"),
    path("professor/live/end/<int:lecture_id>/", views.end_audit, name="end_audit"),
    path(
        "professor/dashboard/<int:lecture_id>/",
        views.professor_dashboard,
        name="professor_dashboard",
    ),
    path(
        "professor/plan/<int:lecture_id>/",
        views.next_lecture_plan,
        name="next_lecture_plan",
    ),
    path("professor/gap/<int:gap_id>/animate/", views.animate_gap, name="animate_gap"),
    path("professor/gap/<int:gap_id>/status/", views.gap_status, name="gap_status"),
    path("student/dashboard/", views.student_dashboard, name="student_dashboard"),
    path("student/day/", views.student_day_content, name="student_day_content"),
    path("student/check-in/<int:lecture_id>/", views.submit_checkin, name="submit_checkin"),
]
