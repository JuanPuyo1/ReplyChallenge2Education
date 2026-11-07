"""Thin Groq wrapper: one client, one helper that returns validated Pydantic objects."""
from __future__ import annotations

import json
from typing import TypeVar

from groq import Groq
from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)
_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY, max_retries=3, timeout=60)
    return _client


def set_client(c) -> None:
    """Tests inject a fake here."""
    global _client
    _client = c


def call_json(model: str, system: str, user: str, schema: type[T],
              reasoning_effort: str = "low", retries: int = 1) -> T:
    messages = [
        {"role": "system", "content": system + "\n\nReply with ONE JSON object that matches "
         "this JSON Schema, and nothing else:\n" + json.dumps(schema.model_json_schema())},
        {"role": "user", "content": user},
    ]
    err: Exception | None = None
    for _ in range(retries + 1):
        resp = client().chat.completions.create(
            model=model, messages=messages, temperature=0.2,
            response_format={"type": "json_object"},
            extra_body={"reasoning_effort": reasoning_effort},  # gpt-oss: low | medium | high
        )
        raw = resp.choices[0].message.content or ""
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            err = e
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That JSON failed validation:\n{e}\n"
                          "Return the corrected JSON object only."}]
    raise err  # type: ignore[misc]
