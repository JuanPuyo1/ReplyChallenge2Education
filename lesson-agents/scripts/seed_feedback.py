"""Fill a lesson with fake student form responses so you can try the Analyst.

    python -m scripts.seed_feedback <lesson_id> [n_students]

The first topic is rated well, the last one badly, so the report has something to say."""
import random
import sys

from app import db

COMMENTS = [
    "", "", "Good pace overall.", "The last part went too fast.",
    "More examples please, the formulas alone were hard to follow.",
    "Loved the live demo.", "I got lost when the slides skipped the derivation.",
    "Could you share the notebook?", "",
]


def main(lesson_id: int, n: int = 25) -> None:
    lesson = db.get_lesson(lesson_id)
    if not lesson:
        sys.exit(f"lesson {lesson_id} not found")
    topics = lesson["topics"]
    with db.pool().connection() as c:
        for _ in range(n):
            r = c.execute(
                "INSERT INTO feedback_responses (lesson_id, overall_rating, pace_rating, comment) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (lesson_id, random.choice([2, 3, 3, 4, 4, 4, 5]), random.choice([3, 3, 4, 4, 5]),
                 random.choice(COMMENTS) or None)).fetchone()
            for i, t in enumerate(topics):
                # Understanding drifts down across the lesson.
                mean = 4.5 - 2.5 * i / max(1, len(topics) - 1)
                u = min(5, max(1, round(random.gauss(mean, 0.9))))
                c.execute("INSERT INTO feedback_topic_ratings VALUES (%s, %s, %s)", (r["id"], t["id"], u))
    print(f"Inserted {n} responses for lesson {lesson_id}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 25)
