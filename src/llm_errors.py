#!/usr/bin/env python3
"""
Classification of LLM/API-layer exceptions into fatal (halt the whole run) vs
recoverable (log and continue) — used at every LLM call-site boundary in
dataset_generator.py, resume_generator.py, and correction.py.

Content-level problems (bad schema, an LLM batch returning fewer items than
requested) never reach this classifier at all — those are always captured as
invalid records by the validate_job/validate_resume path in validation.py,
never raised. Only genuine LLM/API-layer failures propagate as exceptions.

groq and openai ship separate (not shared) exception hierarchies with
identical class names, so both must be listed explicitly here.
"""

import groq
import openai
from instructor.core.exceptions import InstructorRetryException

FATAL_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    groq.RateLimitError,
    groq.AuthenticationError,
    groq.PermissionDeniedError,
    groq.NotFoundError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    openai.RateLimitError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    InstructorRetryException,
)


def is_fatal_llm_error(exc: BaseException) -> bool:
    """True if exc is a genuine LLM/API-layer failure that should halt the
    entire run (rate limit, auth, connection, or the model/API failing to
    ever produce parseable output) rather than being logged and skipped."""
    return isinstance(exc, FATAL_EXCEPTION_TYPES)
