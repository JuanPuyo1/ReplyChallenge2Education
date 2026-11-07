from django.db import models


class Course(models.Model):
    code = models.CharField(max_length=24, unique=True)
    title = models.CharField(max_length=200)
    term = models.CharField(max_length=80, blank=True)
    slide_count = models.PositiveIntegerField(default=0)
    past_exam_count = models.PositiveIntegerField(default=0)

    def __str__(self) -> str:
        return f"{self.code} — {self.title}"


class Lecture(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        LIVE = "live", "Live"
        AUDITED = "audited", "Audited"

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="lectures")
    number = models.PositiveIntegerField()
    title = models.CharField(max_length=200)
    date = models.DateField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.DRAFT)
    recording_started_at = models.DateTimeField(null=True, blank=True)
    recording_ended_at = models.DateTimeField(null=True, blank=True)
    checkin_questions = models.JSONField(default=list, blank=True)
    lesson_brief = models.JSONField(default=dict, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["course", "number"], name="unique_course_lecture")
        ]
        ordering = ["-date", "-number"]

    def __str__(self) -> str:
        return f"{self.course.code} Lecture {self.number}"

    def pulse_questions(self) -> list[dict]:
        """Check-in questions as [{key, text, topic}]; tolerates legacy plain strings."""
        questions = []
        for index, item in enumerate(self.checkin_questions[:3], start=1):
            if isinstance(item, dict):
                text, topic = str(item.get("text", "")), str(item.get("topic", ""))
            else:
                text, topic = str(item), ""
            if text:
                questions.append({"key": f"q{index}", "text": text, "topic": topic})
        return questions


class SlideChunk(models.Model):
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE, related_name="slides")
    slide_number = models.PositiveIntegerField()
    topic = models.CharField(max_length=200, db_index=True)
    text = models.TextField()
    latex_symbols = models.JSONField(default=list, blank=True)
    time_spent_seconds = models.PositiveIntegerField(default=0)
    best_similarity = models.FloatField(default=0.0)
    qdrant_point_id = models.CharField(max_length=64, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["lecture", "slide_number"], name="unique_lecture_slide")
        ]
        ordering = ["slide_number"]


class TranscriptSegment(models.Model):
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE, related_name="transcripts")
    seq = models.PositiveIntegerField()
    start_ms = models.PositiveIntegerField(default=0)
    duration_ms = models.PositiveIntegerField(default=0)
    raw_text = models.TextField(blank=True)
    cleaned_text = models.TextField(blank=True)
    matched_slide_number = models.PositiveIntegerField(null=True, blank=True)
    similarity_score = models.FloatField(default=0.0)
    # Teaching signals from the Listener: topic, key_concepts, examples_given,
    # student_questions, clarity_signals.
    signals = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["lecture", "seq"], name="unique_lecture_segment")
        ]
        ordering = ["seq"]

    @property
    def clock(self) -> str:
        total = int(self.start_ms / 1000)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class StudentCheckIn(models.Model):
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE, related_name="checkins")
    student_ref = models.CharField(max_length=80, blank=True)
    answers = models.JSONField(default=dict)
    score = models.FloatField(default=0.0)
    topic_scores = models.JSONField(default=dict, blank=True)
    misconception = models.CharField(max_length=300, blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["lecture", "-submitted_at"])]


class GapAlert(models.Model):
    class Severity(models.TextChoices):
        CRITICAL = "critical", "Critical"
        WARNING = "warning", "Warning"
        OPTIMAL = "optimal", "Optimal"

    class ManimStatus(models.TextChoices):
        IDLE = "idle", "Idle"
        EXTRACTING = "extracting_notation", "Extracting notation"
        RENDERING = "rendering_manim", "Rendering Manim"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE, related_name="gaps")
    topic = models.CharField(max_length=200)
    title = models.CharField(max_length=240)
    severity = models.CharField(max_length=16, choices=Severity.choices, db_index=True)
    audio_time_seconds = models.PositiveIntegerField(default=0)
    quiz_score = models.FloatField(null=True, blank=True)
    exam_weight = models.FloatField(default=0.0)
    exam_frequency = models.CharField(max_length=100, blank=True)
    slide_number = models.PositiveIntegerField(null=True, blank=True)
    recommendation = models.TextField()
    evidence = models.JSONField(default=dict, blank=True)
    teaching_actions = models.JSONField(default=list, blank=True)
    manim_status = models.CharField(
        max_length=32, choices=ManimStatus.choices, default=ManimStatus.IDLE
    )
    task_error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = [
            models.Case(
                models.When(severity="critical", then=0),
                models.When(severity="warning", then=1),
                default=2,
            ),
            "-exam_weight",
        ]
        constraints = [
            models.UniqueConstraint(fields=["lecture", "topic"], name="unique_lecture_gap_topic")
        ]


class ManimAnimation(models.Model):
    gap = models.OneToOneField(
        GapAlert, on_delete=models.SET_NULL, related_name="animation", null=True, blank=True
    )
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE, related_name="animations")
    title = models.CharField(max_length=240)
    slug = models.SlugField(max_length=120, unique=True)
    scene_name = models.CharField(max_length=120)
    python_code = models.TextField(blank=True)
    video_url = models.CharField(max_length=500, blank=True)
    latex_intuition = models.TextField(blank=True)
    exam_link = models.CharField(max_length=180, blank=True)
    published = models.BooleanField(default=False)
    rendered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-rendered_at", "title"]
