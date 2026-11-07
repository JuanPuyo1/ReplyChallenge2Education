# PoliTo Pulse

Django lecture telemetry, exam-gap auditing, and Manim remediation for Politecnico di Torino.

```powershell
cd polito_pulse
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe manage.py migrate
..\.venv\Scripts\python.exe manage.py seed_polito_demo
..\.venv\Scripts\python.exe manage.py runserver
```

Open `http://127.0.0.1:8000/login` and use the 1-click judge logins.

Set `ANTHROPIC_API_KEY` and `GROQ_API_KEY` in `.env` to enable Claude Haiku 4.5 (chat/Manim JSON), Groq Whisper, and Groq embeddings. Until then the UI, Qdrant fallbacks, and Manim pipeline run with deterministic Automatic Control data.
