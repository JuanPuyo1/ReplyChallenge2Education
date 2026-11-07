from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from django.db import close_old_connections

executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="polito-pulse")
_futures: dict[int, Future] = {}
_lock = Lock()


def _run_manim(gap_id: int) -> dict:
    close_old_connections()
    try:
        from agents.manim_agent import generate_animation_for_gap

        return generate_animation_for_gap(gap_id)
    finally:
        close_old_connections()


def submit_manim_job(gap_id: int) -> Future:
    with _lock:
        current = _futures.get(gap_id)
        if current and not current.done():
            return current
        future = executor.submit(_run_manim, gap_id)
        _futures[gap_id] = future
        return future


def submit_audit_job(lecture_id: int) -> Future:
    def run() -> dict:
        close_old_connections()
        try:
            from agents.dashboard_agent import run_gap_audit

            return run_gap_audit(lecture_id)
        finally:
            close_old_connections()

    return executor.submit(run)
