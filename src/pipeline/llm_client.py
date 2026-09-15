"""
llm_client.py

Thin wrapper around the OpenAI API. Both synthetic_data.py (data
generation) and report_generator.py (report writing) call through this
one module - swapping providers later only touches this file.

Requires the OPENAI_API_KEY environment variable to be set.
"""

import os
import json

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before running the pipeline: export OPENAI_API_KEY=..."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm_text(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.7) -> str:
    """Plain text completion - used for report generation."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def call_llm_json(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.7) -> dict:
    """
    JSON-mode completion - used for synthetic data generation. Uses
    OpenAI's response_format={"type": "json_object"} to force valid JSON
    output, avoiding the classic "wrapped in markdown fences / extra
    prose" failure mode of plain-text prompting.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    return json.loads(content)