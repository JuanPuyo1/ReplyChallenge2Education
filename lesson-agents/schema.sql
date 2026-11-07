-- Lesson Agents schema. Safe to run repeatedly.

CREATE TABLE IF NOT EXISTS lessons (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    course      TEXT,
    language    TEXT,                       -- ISO code passed to Whisper ('it', 'en'); NULL = auto
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ
);

-- The lesson plan: what the teacher intends to cover. Both agents key off these IDs,
-- and the student form should reference the same IDs.
CREATE TABLE IF NOT EXISTS lesson_topics (
    id          BIGSERIAL PRIMARY KEY,
    lesson_id   BIGINT NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    position    INT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT
);

-- One row per uploaded audio chunk (Agent 1, step 1).
CREATE TABLE IF NOT EXISTS transcript_segments (
    id          BIGSERIAL PRIMARY KEY,
    lesson_id   BIGINT NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    seq         INT NOT NULL,
    start_ms    INT NOT NULL,
    duration_ms INT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (lesson_id, seq)
);

-- One row per Listener window (Agent 1, step 2): which planned topic was being taught,
-- and signals about how it was taught.
CREATE TABLE IF NOT EXISTS listener_annotations (
    id                BIGSERIAL PRIMARY KEY,
    lesson_id         BIGINT NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    start_seq         INT NOT NULL,
    end_seq           INT NOT NULL,
    start_ms          INT NOT NULL,
    duration_ms       INT NOT NULL,
    topic_id          BIGINT REFERENCES lesson_topics(id) ON DELETE SET NULL,
    off_plan_topic    TEXT,
    summary           TEXT NOT NULL,
    key_concepts      JSONB NOT NULL DEFAULT '[]',
    examples_given    JSONB NOT NULL DEFAULT '[]',
    student_questions JSONB NOT NULL DEFAULT '[]',
    clarity_signals   JSONB NOT NULL DEFAULT '[]',
    confidence        REAL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (lesson_id, start_seq)
);

-- ---------------------------------------------------------------------------
-- Student form results. If your form already writes to Postgres with its own
-- tables, skip these two and adapt the two queries marked "ADAPT" in app/db.py.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback_responses (
    id             BIGSERIAL PRIMARY KEY,
    lesson_id      BIGINT NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    student_ref    TEXT,                                            -- pseudonymous id, optional
    overall_rating SMALLINT NOT NULL CHECK (overall_rating BETWEEN 1 AND 5),
    pace_rating    SMALLINT CHECK (pace_rating BETWEEN 1 AND 5),     -- 1 too slow, 3 right, 5 too fast
    comment        TEXT,
    submitted_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- "How well did you understand each topic?" 1 = not at all, 5 = fully.
-- A "tick the topics you didn't understand" form maps to this as ticked = 1, unticked = 5.
CREATE TABLE IF NOT EXISTS feedback_topic_ratings (
    response_id   BIGINT NOT NULL REFERENCES feedback_responses(id) ON DELETE CASCADE,
    topic_id      BIGINT NOT NULL REFERENCES lesson_topics(id) ON DELETE CASCADE,
    understanding SMALLINT NOT NULL CHECK (understanding BETWEEN 1 AND 5),
    PRIMARY KEY (response_id, topic_id)
);

-- Agent 2 output.
CREATE TABLE IF NOT EXISTS lesson_reports (
    id         BIGSERIAL PRIMARY KEY,
    lesson_id  BIGINT NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    stats      JSONB NOT NULL,
    report     JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_lesson    ON transcript_segments (lesson_id, seq);
CREATE INDEX IF NOT EXISTS idx_annotations_lesson ON listener_annotations (lesson_id, start_seq);
CREATE INDEX IF NOT EXISTS idx_feedback_lesson    ON feedback_responses (lesson_id);
