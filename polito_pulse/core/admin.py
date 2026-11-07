from django.contrib import admin

from .models import (
    Course,
    GapAlert,
    Lecture,
    ManimAnimation,
    SlideChunk,
    StudentCheckIn,
    TranscriptSegment,
)

for model in (
    Course,
    Lecture,
    SlideChunk,
    TranscriptSegment,
    StudentCheckIn,
    GapAlert,
    ManimAnimation,
):
    admin.site.register(model)
