#!/usr/bin/env python3
"""
Synthetic dataset generator for job postings and resumes.

Usage:
    python dataset_generator.py                          # 1 industry (test run)
    python dataset_generator.py --max-industries 5       # all 5 industries
    python dataset_generator.py --industry "Healthcare"  # specific industry
    python dataset_generator.py --num-jobs 50 --resumes-per-job 5

Output (data/ directory):
    data/jobs_{iteration}.jsonl     — all job postings
    data/resumes_{iteration}.jsonl  — all resumes
    data/pairs_{iteration}.jsonl    — resume-job pairs with metadata

Logs (logs/ directory):
    logs/dataset_log_{iteration}.jsonl  — structured event log
"""

import argparse
import json
import re
import time
import uuid
import traceback
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import os

try:
    from dotenv import load_dotenv
except ImportError:
    # Allow running CLI help even if python-dotenv is not installed in the active env.
    def load_dotenv() -> bool:
        return False

load_dotenv()

import sys

import instructor
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel

from validation import (
    EXPERIENCE_LEVELS,
    CompanyInfo, JobRequirements, JobContent, JobPosting, JobBatch, RawJobBatch,
    normalize_job_content, normalize_experience_level, infer_experience_years,
    parse_experience_years, infer_job_title, default_job_title,
    ContactInfo, EducationEntry, WorkExperience, ProficiencyLevel, SkillEntry,
    ResumeContent, Resume, ResumeBatch, RawResumeBatch,
    ResumePair,
    validate_job, validate_resume,
)
from llm_errors import is_fatal_llm_error
from resume_generator import generate_resumes_for_job

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
NUM_JOBS: int = int(os.getenv("NUM_JOBS", "50"))
NUM_RESUMES_PER_JOB: int = int(os.getenv("NUM_RESUMES_PER_JOB", "5"))
JOB_MAX_COMPLETION_TOKENS: int = int(os.getenv("JOB_MAX_COMPLETION_TOKENS", "2200"))
RESUME_MAX_COMPLETION_TOKENS: int = int(os.getenv("RESUME_MAX_COMPLETION_TOKENS", "3000"))
JUDGE_MAX_COMPLETION_TOKENS: int = int(os.getenv("JUDGE_MAX_COMPLETION_TOKENS", "4000"))

# Per-call network timeout (seconds) for every client.chat.completions.create()
# call in this codebase. None of the underlying SDKs time out on their own in
# any reasonable window (openai-python defaults to 600s) — this is what
# actually bounds worst-case latency, independent of max_completion_tokens.
LLM_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "45"))

# A run only aborts once this many FATAL LLM/API errors happen in a row, with no
# successful batch in between — an isolated flaky batch (e.g. the model garbling
# a single response) no longer kills a whole run, but a genuine persistent
# problem (real rate-limit/auth failure that keeps recurring) still does.
MAX_CONSECUTIVE_FATAL_FAILURES: int = int(os.getenv("MAX_CONSECUTIVE_FATAL_FAILURES", "3"))
ITERATION: int = int(os.getenv("ITERATION", "1"))
RATE_LIMIT_TPM: int = int(os.getenv("RATE_LIMIT_TPM", "5000"))
JOB_MAX_BATCH_SIZE: int = int(os.getenv("JOB_MAX_BATCH_SIZE", "5"))
RESUME_MAX_BATCH_SIZE: int = int(os.getenv("RESUME_MAX_BATCH_SIZE", "2"))

# Total LLM calls (initial + top-ups) allowed per requested batch when the LLM
# returns fewer items than asked for — see generate_jobs_batch / _generate_resumes_batch.
MAX_BATCH_TOPUP_ATTEMPTS: int = 3

INDUSTRIES: list[str] = [
    "Information Technology & Software",
    "Healthcare",
    "Finance & Banking",
    "Manufacturing & Engineering",
    "Sales, Marketing & Customer Success",
]

FIT_LEVELS: list[str] = ["Excellent", "Good", "Partial", "Poor"]

# Resolve relative to project root (one level above src/)
_PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = _PROJECT_ROOT / "data"
LOGS_DIR: Path = _PROJECT_ROOT / "logs"
TEMPLATE_DIR: Path = _PROJECT_ROOT / "templates"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

TEMPLATE_NAMES: list[str] = [
    "formal_corporate",
    "casual_startup_friendly",
    "tech_detail_heavy",
    "achievement_focused_metrics",
    "career_changer_xfer_skills",
]


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Sliding-window rate limiter to prevent exceeding Groq's TPM (tokens per minute) limits.
    Tracks tokens sent in the last 60 seconds and sleeps before requests if approaching the limit.
    """

    def __init__(self, tpm_limit: int) -> None:
        self.tpm_limit = tpm_limit
        self.requests: deque = deque()  # (timestamp, estimated_tokens)

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """
        Check if adding estimated_tokens would exceed the TPM limit.
        Loops and sleeps until enough of the 60s window has cleared to fit the request.
        If the request alone exceeds the limit (e.g. large batch), waits for any
        currently-tracked usage to clear, then proceeds anyway — no amount of
        waiting can ever bring a single over-budget request under budget, so
        looping forever on that check would hang indefinitely.
        """
        while True:
            now = time.time()

            # Remove requests older than 60 seconds
            while self.requests and self.requests[0][0] < now - 60:
                self.requests.popleft()

            # Sum tokens in the last 60 seconds
            tokens_last_60s = sum(tokens for _, tokens in self.requests)

            # Check if we have budget for this request
            if tokens_last_60s + estimated_tokens <= self.tpm_limit:
                # Budget available — record and proceed
                self.requests.append((time.time(), estimated_tokens))
                return

            # This single request alone exceeds the whole budget: once there's no
            # other tracked usage left to wait out, further waiting is pointless
            # (estimated_tokens alone will never fit under tpm_limit) — proceed.
            if estimated_tokens > self.tpm_limit and not self.requests:
                print(f"⚠️  Rate limit: this request needs ~{estimated_tokens} tokens, which alone "
                      f"exceeds the {self.tpm_limit} TPM budget. Proceeding without waiting further.")
                self.requests.append((time.time(), estimated_tokens))
                return

            # Need to wait — sleep until the oldest request expires
            if self.requests:
                oldest_time = self.requests[0][0]
                sleep_time = 60 - (now - oldest_time) + 0.5  # Extra 0.5s buffer
            else:
                # estimated_tokens alone exceeds the limit; wait a full minute
                sleep_time = 61.0
            print(f"⏳ Rate limit: {tokens_last_60s + estimated_tokens}/{self.tpm_limit} tokens needed. Sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    def get_stats(self) -> dict:
        """Return current rate limiter statistics."""
        now = time.time()
        # Remove old requests
        while self.requests and self.requests[0][0] < now - 60:
            self.requests.popleft()
        tokens_last_60s = sum(tokens for _, tokens in self.requests)
        return {
            "tpm_limit": self.tpm_limit,
            "tokens_last_60s": tokens_last_60s,
            "requests_last_60s": len(self.requests),
        }


# Init global rate limiter
_rate_limiter = RateLimiter(RATE_LIMIT_TPM)


class ConsecutiveFailureTracker:
    """Tracks consecutive fatal LLM/API failures across an entire run — shared
    between job-batch and resume-batch call sites in process_industry, and
    NOT reset per industry, so a persistent problem straddling an industry
    boundary is still caught. Any successful batch (job or resume) resets the
    streak to 0; a non-fatal failure leaves it unchanged (it's a different
    signal, doesn't count either way)."""

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.streak = 0

    def record_failure(self) -> bool:
        """Returns True if the run should now abort (streak reached threshold)."""
        self.streak += 1
        return self.streak >= self.threshold

    def record_success(self) -> None:
        self.streak = 0


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    """Writes structured JSONL event log to logs/dataset_log_{iteration}.jsonl."""

    def __init__(self, iteration: int) -> None:
        self._path = LOGS_DIR / f"dataset_log_{iteration}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")

    def _write(self, event: str, **fields) -> None:
        record = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def run_start(self, industries: list, jobs_per_industry: int, resumes_per_job: int,
                  generation_model: str, iteration: int) -> None:
        self._write("run_start", iteration=iteration, industries=industries,
                    jobs_per_industry=jobs_per_industry, resumes_per_job=resumes_per_job,
                    generation_model=generation_model)

    def industry_jobs_batch_start(self, industry: str, job_count: int) -> None:
        self._write("industry_jobs_batch_start", industry=industry, job_count=job_count)

    def industry_resumes_batch_start(self, industry: str, job_count: int,
                                      resumes_per_job: int) -> None:
        self._write("industry_resumes_batch_start", industry=industry, job_count=job_count,
                    total_resumes_planned=job_count * resumes_per_job)

    def job_generation_start(self, industry: str, job_index: int, experience_level: str,
                              trace_id: str) -> None:
        self._write("job_generation_start", industry=industry, job_index=job_index,
                    experience_level=experience_level, trace_id=trace_id)

    def resume_generation_start(self, job_trace_id: str, job_title: str, resume_index: int,
                                  fit_level: str, template: str, trace_id: str) -> None:
        self._write("resume_generation_start", job_trace_id=job_trace_id, job_title=job_title,
                    resume_index=resume_index, fit_level=fit_level, template=template,
                    trace_id=trace_id)

    def llm_error(self, context: str, trace_id: str, error_message: str,
                  attempt: Optional[int] = None, error: Optional[BaseException] = None) -> None:
        """Log any LLM/API error. Classifies 429, retry, and JSON validation errors."""
        error_str = str(error_message)
        if "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
            event = "llm_error_rate_limit"
        elif "retry" in error_str.lower() or "failed_attempts" in error_str.lower() or "max_retries" in error_str.lower():
            event = "llm_error_max_retries"
        elif "json_validate_failed" in error_str or "json validate" in error_str.lower():
            event = "llm_error_json_validation"
        else:
            event = "llm_error"
        payload: dict = {
            "context": context,
            "trace_id": trace_id,
            "error": error_str,
            "error_type": type(error).__name__ if error is not None else None,
        }
        if attempt is not None:
            payload["attempt"] = attempt
        if error is not None:
            payload["error_repr"] = repr(error)
            payload["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self._write(event, **payload)

    def batch_incomplete(self, context: str, trace_id: str, expected: int, actual: int) -> None:
        """Log a batch LLM call that returned fewer items than requested (no exception raised)."""
        self._write("llm_batch_incomplete", context=context, trace_id=trace_id,
                    expected=expected, actual=actual, missing=expected - actual)

    def fatal_error_tolerated(self, context: str, streak: int, threshold: int) -> None:
        """Log a fatal-classified LLM/API error that did NOT abort the run because
        the consecutive-failure streak hadn't reached the threshold yet."""
        self._write("llm_fatal_error_tolerated", context=context, streak=streak, threshold=threshold)

    def run_complete(self, total_jobs: int, total_resumes: int,
                     jobs_path: str, resumes_path: str, pairs_path: str) -> None:
        self._write("run_complete", total_jobs=total_jobs, total_resumes=total_resumes,
                    jobs_path=jobs_path, resumes_path=resumes_path, pairs_path=pairs_path)

    def run_aborted(self, industry: str, error_message: str, error: Optional[BaseException] = None) -> None:
        """Log a fatal LLM/API error that halted the entire run mid-industry."""
        payload: dict = {
            "industry": industry,
            "error_message": error_message,
            "error_type": type(error).__name__ if error is not None else None,
        }
        if error is not None:
            payload["error_repr"] = repr(error)
            payload["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self._write("run_aborted", **payload)

    def correction_run_start(self, iteration: int, jobs_invalid_count: int, resumes_invalid_count: int) -> None:
        self._write("correction_run_start", iteration=iteration,
                    jobs_invalid_count=jobs_invalid_count, resumes_invalid_count=resumes_invalid_count)

    def correction_run_complete(self, iteration: int, jobs_corrected: int, jobs_uncorrectable: int,
                                 resumes_corrected: int, resumes_uncorrectable: int) -> None:
        self._write("correction_run_complete", iteration=iteration, jobs_corrected=jobs_corrected,
                    jobs_uncorrectable=jobs_uncorrectable, resumes_corrected=resumes_corrected,
                    resumes_uncorrectable=resumes_uncorrectable)

    def correction_attempt(self, kind: str, trace_id: str, original_error_type: str) -> None:
        """kind is 'job' or 'resume'."""
        self._write("correction_attempt", kind=kind, trace_id=trace_id, original_error_type=original_error_type)

    def correction_success(self, kind: str, trace_id: str, pair_trace_id: Optional[str] = None) -> None:
        self._write("correction_success", kind=kind, trace_id=trace_id, pair_trace_id=pair_trace_id)

    def correction_failed(self, kind: str, trace_id: str, reason: str, error: Optional[BaseException] = None) -> None:
        payload: dict = {"kind": kind, "trace_id": trace_id, "reason": reason}
        if error is not None:
            payload["error_repr"] = repr(error)
            payload["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self._write("correction_failed", **payload)

    def close(self) -> None:
        self._file.close()


# ── Template Loading ──────────────────────────────────────────────────────────

def load_templates() -> dict[str, dict]:
    templates: dict[str, dict] = {}
    for name in TEMPLATE_NAMES:
        path = TEMPLATE_DIR / f"{name}.json"
        with open(path) as f:
            templates[name] = json.load(f)
    return templates


# ── Instructor Client ─────────────────────────────────────────────────────────

def make_client(model: str = GENERATION_MODEL) -> instructor.Instructor:
    """Build an instructor client for `model`. Provider is picked per-model
    (not globally) so callers can request a client for JUDGE_MODEL, which may
    be on a different provider than GENERATION_MODEL."""
    if "/" in model:
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to your environment or .env file."
            )
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        return instructor.from_openai(client, mode=instructor.Mode.JSON)

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file."
        )
    return instructor.from_groq(Groq(api_key=GROQ_API_KEY), mode=instructor.Mode.JSON)


# ── Distribution Helpers ──────────────────────────────────────────────────────

def distribute_experience_levels(n: int) -> list[str]:
    """Assign experience levels across n jobs, cycling through all 4 levels."""
    return [EXPERIENCE_LEVELS[i % len(EXPERIENCE_LEVELS)] for i in range(n)]


def get_fit_level(global_resume_index: int) -> str:
    """Cycle through fit levels globally so all 4 levels appear equally across the dataset."""
    return FIT_LEVELS[global_resume_index % len(FIT_LEVELS)]


def get_template_name(resume_index_within_job: int) -> str:
    """Each resume within a job uses a different template (all 5 used per job with 5 resumes)."""
    return TEMPLATE_NAMES[resume_index_within_job % len(TEMPLATE_NAMES)]


def get_fit_level_brief(fit_level: str) -> str:
    """Return a short fit-level reminder for batch prompts."""
    if fit_level == "Excellent":
        return "Excellent: strong, highly aligned candidate."
    if fit_level == "Good":
        return "Good: mostly aligned with minor gaps."
    if fit_level == "Partial":
        return "Partial: relevant but with clear gaps."
    return "Poor: noticeably misaligned but still realistic."


# ── Shortcoming Instructions ──────────────────────────────────────────────────

def get_shortcoming_instructions(fit_level: str, job: JobPosting) -> str:
    """Return prompt instructions for intentional resume-job mismatches based on fit level."""
    required = ", ".join(job.requirements.required_skills[:4])
    req_level = job.requirements.experience_level
    level_idx = EXPERIENCE_LEVELS.index(req_level) if req_level in EXPERIENCE_LEVELS else 1

    if fit_level == "Excellent":
        return (
            "The candidate is an excellent fit. They possess all required skills at Advanced or Expert level, "
            "their experience level and years align precisely with the job requirements, and they have directly "
            "relevant domain experience. Write a compelling, well-rounded candidate who would be a top-tier applicant."
        )

    if fit_level == "Good":
        return (
            f"The candidate is a good fit with minor shortcomings. They have most required skills "
            f"({required}) but 1-2 preferred skills may be missing or at a lower proficiency than ideal. "
            f"Experience years are close but perhaps slightly under the requirement. Overall a solid candidate "
            f"who would likely advance to interviews, with just a few areas for growth."
        )

    if fit_level == "Partial":
        below_level = EXPERIENCE_LEVELS[max(0, level_idx - 1)]
        return (
            f"The candidate is a partial fit — some meaningful gaps exist but are not disqualifying on their own. "
            f"Intentionally introduce ALL of the following: "
            f"(1) Missing 2-3 of these required skills: {required} — replace them with adjacent but different skills; "
            f"(2) Experience level is '{below_level}' rather than the required '{req_level}' — they are one level below; "
            f"(3) Background is in an adjacent but slightly different sub-domain or industry vertical. "
            f"Do NOT make these failures obvious — this should look like a real person who genuinely has relevant but mismatched experience."
        )

    # Poor
    above_idx = min(len(EXPERIENCE_LEVELS) - 1, level_idx + 1)
    wrong_level = EXPERIENCE_LEVELS[above_idx] if above_idx != level_idx else EXPERIENCE_LEVELS[max(0, level_idx - 1)]
    return (
        f"The candidate is a poor fit with significant shortcomings. "
        f"Intentionally introduce ALL of the following: "
        f"(1) Missing most required skills — {required} — the candidate's skills are from a clearly different domain; "
        f"(2) Experience level is '{wrong_level}' (misaligned from the required '{req_level}'); "
        f"(3) Work history is in a different industry or function that does not align with this role; "
        f"(4) Achievements and responsibilities listed are from an unrelated field. "
        f"Make it realistic — this should be a real person who genuinely isn't the right fit, "
        f"not an obviously fabricated or incoherent resume. The mismatch should be clear but subtle."
    )


# ── Generation Functions ──────────────────────────────────────────────────────

def build_jobs_batch_prompt(industry: str, experience_levels: list[str]) -> str:
    """Build the batch job-generation prompt. Used for normal batch generation
    (batch_size = len(experience_levels)) and, with a single-element list, by
    correction.py's missing-job regeneration path."""
    batch_size = len(experience_levels)

    level_descriptions = []
    for i, level in enumerate(experience_levels):
        exp_year_range = {
            "Entry": "0–2",
            "Mid": "2–5",
            "Senior": "5–10",
            "Lead/Principal": "8–15",
        }.get(level, "2–5")
        level_descriptions.append(f"  Job {i + 1}: Experience Level = {level} ({exp_year_range} years)")

    levels_prompt = "\n".join(level_descriptions)

    return f"""Generate exactly {batch_size} distinct, realistic job postings for the {industry} industry.

Requirements for each job:
- Return nested objects named `company` and `requirements`.
- `company` must contain realistic values for `name`, `industry`, `size`, and `location`.
- `requirements` must contain realistic values for `required_skills`, `preferred_skills`, `education`, `experience_years`, and `experience_level`.
- Include 4–7 required_skills and 2–5 preferred_skills realistic for the role
- Write a genuine 2–3 paragraph job_description describing the role, team context, and expectations
- Choose company names, sizes, and locations that fit the industry realistically
- Set is_niche_role to true only if this role requires a very specific combination of skills
- Ensure each job posting is distinct and diverse (different sub-domain, company type, seniority)
- Do not use legacy flat fields like company_name, company_size, location, required_skills, preferred_skills, or experience_level at the top level.
- Never output placeholders, template markers, angle-bracket values, or example literals such as `<company_name>`, `required1`, or `This is the first paragraph...`.

Experience levels and year ranges:
{levels_prompt}

Generate {batch_size} unique job postings matching these experience levels exactly."""


def format_job_summary(job: JobPosting) -> str:
    """Compact job summary text injected into resume-generation prompts."""
    return (
        f"Job Title: {job.job_title}\n"
        f"Company: {job.company.name} ({job.company.industry}, {job.company.size} employees, {job.company.location})\n"
        f"Experience Level Required: {job.requirements.experience_level} ({job.requirements.experience_years}+ years)\n"
        f"Required Skills: {', '.join(job.requirements.required_skills)}\n"
        f"Preferred Skills: {', '.join(job.requirements.preferred_skills)}\n"
        f"Education Required: {job.requirements.education}\n"
    )


def generate_jobs_batch(
    client: instructor.Instructor,
    industry: str,
    experience_levels: list[str],
    batch_start_index: int,
    job_counter_start: int,
    max_completion_tokens: int,
    logger: "Logger",
) -> tuple[list[JobPosting], list[tuple[dict, dict]]]:
    """
    Generate multiple job postings, retrying immediately with a smaller
    top-up batch for any slots the LLM doesn't return on the first attempt.

    Args:
        client: Instructor client for LLM calls
        industry: Industry name for job generation
        experience_levels: List of experience levels for this batch
        batch_start_index: Starting job index for logging/diversification
        max_completion_tokens: Max tokens for the batch response
        logger: Logger instance for event tracking

    Returns:
        List of JobPosting objects
    """
    batch_size = len(experience_levels)
    now = datetime.now(UTC).isoformat()

    # Keep whatever the LLM already generated and only ask again for the
    # still-missing count, rather than accepting a short batch outright.
    collected_raw_jobs: list[dict] = []
    remaining_levels = experience_levels
    for attempt in range(1, MAX_BATCH_TOPUP_ATTEMPTS + 1):
        prompt = build_jobs_batch_prompt(industry, remaining_levels)

        # max_completion_tokens is a shared budget for the entire batch response (not per-job)
        estimated_tokens = len(prompt) // 4 + max_completion_tokens
        _rate_limiter.wait_if_needed(estimated_tokens)

        batch_response: RawJobBatch = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_model=RawJobBatch,
            max_retries=3,
            max_completion_tokens=max_completion_tokens,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )

        received = batch_response.jobs[: len(remaining_levels)]
        if len(received) < len(remaining_levels):
            logger.batch_incomplete(
                context=f"jobs_batch_generation industry={industry!r} attempt={attempt}",
                trace_id="N/A",
                expected=len(remaining_levels),
                actual=len(received),
            )
        collected_raw_jobs.extend(received)

        if len(collected_raw_jobs) >= batch_size:
            break
        remaining_levels = experience_levels[len(collected_raw_jobs):]

    # Validate each raw job independently so invalid items can still be persisted.
    job_postings: list[JobPosting] = []
    invalid_jobs: list[tuple[dict, dict]] = []
    for i, raw_job in enumerate(collected_raw_jobs):
        job_number = job_counter_start + i
        trace_id = f"job_{job_number}"
        raw_record = dict(raw_job) if isinstance(raw_job, dict) else {"raw_content": raw_job}
        raw_record["trace_id"] = trace_id
        raw_record["generated_at"] = now
        raw_record["industry"] = industry

        job_posting, error_details = validate_job(raw_job, trace_id, now)
        if job_posting is not None:
            job_postings.append(job_posting)
        else:
            invalid_jobs.append((raw_record, error_details))

    return job_postings, invalid_jobs


def generate_resume(
    client: instructor.Instructor,
    job: JobPosting,
    template: dict,
    fit_level: str,
    trace_id: str,
    resume_index_within_job: int,
    max_completion_tokens: int,
) -> Resume:
    now = datetime.now(UTC).isoformat()
    shortcoming_instructions = get_shortcoming_instructions(fit_level, job)
    job_summary = format_job_summary(job)

    prompt = f"""Generate a realistic resume for a candidate applying to the following job.

═══ TEMPLATE STYLE ═══
Name: {template['display_name']}
Writing Style: {template['writing_style']}
Persona: {template['persona']}
Instructions: {template['instructions']}

═══ JOB CONTEXT ═══
{job_summary}

═══ FIT LEVEL: {fit_level.upper()} ═══
{shortcoming_instructions}

═══ STRICT FIELD REQUIREMENTS ═══
- email: must match format user@domain.tld
- phone: at least 10 characters (digits, dashes, spaces, parentheses allowed)
- All dates (start_date, end_date, graduation_date): ISO format only — YYYY-MM-DD or YYYY-MM or YYYY
- end_date must always be chronologically after start_date (use "Present" for current roles)
- gpa: if included, must be between 0.0 and 4.0
- skills: minimum 3 entries with valid proficiency_level from: Beginner, Intermediate, Advanced, Expert
- Include 2–4 work experience entries
- Write a 2–4 sentence professional summary in the style of the template

This is resume #{resume_index_within_job + 1} for this job. Make it a unique, realistic individual."""

    content: ResumeContent = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=ResumeContent,
        max_retries=3,
        max_completion_tokens=max_completion_tokens,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )

    return Resume(
        trace_id=trace_id,
        contact=content.contact,
        education=content.education,
        experience=content.experience,
        skills=content.skills,
        summary=content.summary,
        generated_at=now,
        prompt_template=template["name"],
        fit_level=fit_level,
        writing_style=template["writing_style"],
    )


# ── Judge Stub ────────────────────────────────────────────────────────────────

def judge_pair(job: JobPosting, resume: Resume) -> Optional[float]:
    """
    Stub: evaluate resume-job fit with the judge model.
    Returns a score 0.0–1.0 or None if skipped.
    TODO: Implement judging logic using JUDGE_MODEL.
    """
    return None


# ── JSONL Writer ──────────────────────────────────────────────────────────────

def append_jsonl(file, record: BaseModel) -> None:
    file.write(record.model_dump_json() + "\n")
    file.flush()


def append_invalid_with_error(file, raw_record: dict, error_details: dict) -> None:
    """Write invalid generated record followed by JSON error metadata on the next line."""
    file.write(json.dumps(raw_record) + "\n")
    file.write(json.dumps(error_details) + "\n")
    file.flush()


def _backup_paths(iteration: int, paths: list[Path]) -> None:
    """Move any of the given paths that exist into a single timestamped backup directory."""
    existing_paths = [p for p in paths if p.exists()]
    if not existing_paths:
        return

    backup_root = DATA_DIR / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"iteration_{iteration}_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in existing_paths:
        path.rename(backup_dir / path.name)

    print(f"  Backed up existing iteration outputs to {backup_dir}")


def backup_iteration_outputs(iteration: int) -> None:
    """Move existing generation outputs (the 7 files dataset_generator.py itself
    writes) into a single timestamped backup directory before overwriting them.
    Does NOT touch correction.py's output files — see backup_correction_outputs."""
    _backup_paths(iteration, [
        DATA_DIR / f"jobs_{iteration}.jsonl",
        DATA_DIR / f"resumes_{iteration}.jsonl",
        DATA_DIR / f"pairs_{iteration}.jsonl",
        DATA_DIR / f"jobs_valid_{iteration}.jsonl",
        DATA_DIR / f"jobs_invalid_{iteration}.jsonl",
        DATA_DIR / f"resumes_valid_{iteration}.jsonl",
        DATA_DIR / f"resumes_invalid_{iteration}.jsonl",
    ])


def find_max_trace_id_counter(prefix: str, *paths: Path) -> int:
    """Scan JSONL files for top-level 'trace_id' fields matching '{prefix}_N'
    and return the max N found (0 if none / files missing). Used by
    correction.py's newly-valid-job resume backfill and by --resume here."""
    pattern = re.compile(rf'"trace_id"\s*:\s*"{re.escape(prefix)}_(\d+)"')
    max_n = 0
    for path in paths:
        if not path.exists():
            continue
        for match in pattern.finditer(path.read_text()):
            max_n = max(max_n, int(match.group(1)))
    return max_n


def count_jsonl_lines(path: Path) -> int:
    """Count non-blank lines in a JSONL file (0 if it doesn't exist)."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def backup_correction_outputs(iteration: int) -> None:
    """Move existing correction outputs (the 5 files correction.py writes) into
    a single timestamped backup directory before overwriting them. Deliberately
    separate from backup_iteration_outputs: correction.py reads jobs_invalid_*/
    resumes_invalid_* as input and must never have them moved out from under it."""
    _backup_paths(iteration, [
        DATA_DIR / f"jobs_corrected_{iteration}.jsonl",
        DATA_DIR / f"jobs_uncorrectable_{iteration}.jsonl",
        DATA_DIR / f"resumes_corrected_{iteration}.jsonl",
        DATA_DIR / f"resumes_uncorrectable_{iteration}.jsonl",
        DATA_DIR / f"pairs_corrected_{iteration}.jsonl",
    ])


# ── Industry Processing ───────────────────────────────────────────────────────

def process_industry(
    client: instructor.Instructor,
    logger: Logger,
    industry: str,
    jobs_per_industry: int,
    resumes_per_job: int,
    job_max_completion_tokens: int,
    resume_max_completion_tokens: int,
    templates: dict[str, dict],
    jobs_file,
    jobs_valid_file,
    jobs_invalid_file,
    resumes_file,
    resumes_valid_file,
    resumes_invalid_file,
    pairs_file,
    global_resume_offset: int,
    job_counter_start: int,
    resume_counter_start: int,
    pair_counter_start: int,
    failure_tracker: ConsecutiveFailureTracker,
) -> int:
    """
    Generate all jobs for one industry (in batches), then generate resumes for each job.
    Returns the total number of resumes successfully generated.

    Abort-on-failure pattern:
    - If jobs phase fails, returns 0 (no resumes generated, logged).
    - If resume phase fails for any job, halts remaining jobs and logs error.
    """
    print(f"\n{'=' * 60}")
    print(f"  Industry: {industry}")
    print(f"  Generating {jobs_per_industry} job postings...")
    print(f"{'=' * 60}")

    experience_levels = distribute_experience_levels(jobs_per_industry)
    jobs: list[JobPosting] = []
    invalid_jobs_count = 0
    last_job_generation_error: Optional[BaseException] = None

    # ── Phase 1: Generate all jobs for this industry (in batches) ──────────────
    logger.industry_jobs_batch_start(industry=industry, job_count=jobs_per_industry)

    def _attempt_job_batch(batch_levels: list[str], next_id: int, label: str) -> Optional[tuple[list, list]]:
        """One attempt at generating a batch of jobs. Returns (valid_jobs,
        invalid_jobs) — writing them out — for any call that didn't raise
        (even if it under-delivered). Returns None if the call raised a
        fatal-but-tolerated error (caller decides whether to retry/defer).
        Raises if this failure pushed the run past the abort threshold."""
        nonlocal last_job_generation_error
        print(f"  [{label}] Generating {len(batch_levels)} jobs...", end=" ", flush=True)
        try:
            batch_jobs, batch_invalid_jobs = generate_jobs_batch(
                client=client,
                industry=industry,
                experience_levels=batch_levels,
                batch_start_index=0,
                job_counter_start=next_id,
                max_completion_tokens=job_max_completion_tokens,
                logger=logger,
            )
            for job in batch_jobs:
                append_jsonl(jobs_file, job)
                append_jsonl(jobs_valid_file, job)
            for raw_job, error_details in batch_invalid_jobs:
                append_invalid_with_error(jobs_file, raw_job, error_details)
                append_invalid_with_error(jobs_invalid_file, raw_job, error_details)
            print(f"OK  ({len(batch_jobs)} jobs)")
            failure_tracker.record_success()
            return batch_jobs, batch_invalid_jobs
        except Exception as exc:
            last_job_generation_error = exc
            print(f"ERR  {exc}")
            context = f"jobs_batch_generation industry={industry!r} {label} size={len(batch_levels)}"
            logger.llm_error(context=context, trace_id="N/A", error_message=str(exc), error=exc)
            if is_fatal_llm_error(exc):
                if failure_tracker.record_failure():
                    raise
                logger.fatal_error_tolerated(context, failure_tracker.streak, failure_tracker.threshold)
                print(f"  ⚠️  Fatal-classified error tolerated ({failure_tracker.streak}/{failure_tracker.threshold} consecutive).")
            return None

    # Split jobs into batches. On a tolerated fatal failure, retry the same
    # batch once immediately; if that also fails, defer it to a final
    # make-up pass (after every batch has had its normal turn) rather than
    # losing those job slots outright — see _attempt_job_batch.
    next_job_id = job_counter_start
    num_batches = (jobs_per_industry + JOB_MAX_BATCH_SIZE - 1) // JOB_MAX_BATCH_SIZE  # ceiling division
    pending_levels: list[str] = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * JOB_MAX_BATCH_SIZE
        batch_end = min(batch_start + JOB_MAX_BATCH_SIZE, jobs_per_industry)
        batch_levels = experience_levels[batch_start:batch_end]
        label = f"Batch {batch_idx + 1:2d}/{num_batches}"

        result = _attempt_job_batch(batch_levels, next_job_id, label)
        if result is None:
            print(f"      Retrying {label.lower()}...", end=" ", flush=True)
            result = _attempt_job_batch(batch_levels, next_job_id, f"{label} retry")

        if result is None:
            pending_levels.extend(batch_levels)
            continue

        batch_jobs, batch_invalid_jobs = result
        jobs.extend(batch_jobs)
        invalid_jobs_count += len(batch_invalid_jobs)
        next_job_id += len(batch_jobs) + len(batch_invalid_jobs)

    if pending_levels:
        print(f"\n  Attempting to make up {len(pending_levels)} job(s) that failed earlier...")
        for i in range(0, len(pending_levels), JOB_MAX_BATCH_SIZE):
            chunk = pending_levels[i:i + JOB_MAX_BATCH_SIZE]
            result = _attempt_job_batch(chunk, next_job_id, "Make-up batch")
            if result is None:
                print(f"  ⚠️  Still could not generate {len(chunk)} job(s) after retries; accepting the shortfall.")
                continue
            batch_jobs, batch_invalid_jobs = result
            jobs.extend(batch_jobs)
            invalid_jobs_count += len(batch_invalid_jobs)
            next_job_id += len(batch_jobs) + len(batch_invalid_jobs)

    if not jobs:
        print(f"\n  ❌ No jobs were successfully generated for {industry}. Skipping resume generation.")
        if last_job_generation_error is not None:
            summary_error = (
                f"Job generation phase failed; 0 of {jobs_per_industry} jobs created. "
                f"Last error: {last_job_generation_error}"
            )
            summary_exception = last_job_generation_error
        elif invalid_jobs_count > 0:
            summary_error = (
                f"Job generation phase failed; 0 of {jobs_per_industry} jobs created. "
                f"All {invalid_jobs_count} generated jobs were schema-invalid."
            )
            summary_exception = RuntimeError(summary_error)
        else:
            summary_error = f"Job generation phase failed; 0 of {jobs_per_industry} jobs created."
            summary_exception = RuntimeError(summary_error)
        logger.llm_error(
            context=f"process_industry {industry}",
            trace_id="N/A",
            error_message=summary_error,
            error=summary_exception,
        )
        return 0

    print(f"\n  Generating resumes ({resumes_per_job}/job) for {len(jobs)} jobs...")
    logger.industry_resumes_batch_start(industry=industry, job_count=len(jobs),
                                         resumes_per_job=resumes_per_job)

    # ── Phase 2: Generate resumes for each job ────────────────────────────────
    resumes_generated = 0

    def _attempt_resumes_for_job(job_i: int, job: JobPosting, label: str) -> bool:
        """One attempt at generating resumes for one job. Returns True on
        success (writes happen internally). Returns False if it failed with a
        fatal-but-tolerated error (caller decides whether to retry/defer).
        Raises for a genuine abort (streak hit threshold) or a non-fatal
        error (caller abandons remaining jobs in this industry, unchanged
        from before)."""
        nonlocal resumes_generated
        fit_levels = [get_fit_level(global_resume_offset + job_i * resumes_per_job + r) for r in range(resumes_per_job)]
        try:
            resume_pairs, invalid_resumes = generate_resumes_for_job(
                client=client,
                logger=logger,
                job=job,
                templates=templates,
                fit_levels=fit_levels,
                global_resume_offset=global_resume_offset + job_i * resumes_per_job,
                resume_counter_start=resume_counter_start + job_i * resumes_per_job,
                pair_counter_start=pair_counter_start + job_i * resumes_per_job,
                max_completion_tokens=resume_max_completion_tokens,
            )
            for resume, pair in resume_pairs:
                append_jsonl(resumes_file, resume)
                append_jsonl(resumes_valid_file, resume)
                append_jsonl(pairs_file, pair)
                resumes_generated += 1
            for raw_resume, error_details in invalid_resumes:
                append_invalid_with_error(resumes_file, raw_resume, error_details)
                append_invalid_with_error(resumes_invalid_file, raw_resume, error_details)
            failure_tracker.record_success()
            return True
        except Exception as exc:
            error_msg = str(exc)
            print(f"\n  ❌ Resume generation failed for job '{job.job_title}' (job_trace_id={job.trace_id}) [{label}].")
            context = f"resume_batch_generation job_trace_id={job.trace_id!r} title={job.job_title!r}"
            logger.llm_error(
                context=context,
                trace_id="N/A",
                error_message=f"Batch resume generation failed: {error_msg[:200]}",
                error=exc,
            )
            if is_fatal_llm_error(exc):
                if failure_tracker.record_failure():
                    print(f"     Halting further resume generation due to error: {error_msg[:100]}")
                    raise
                logger.fatal_error_tolerated(context, failure_tracker.streak, failure_tracker.threshold)
                print(f"     Fatal-classified error tolerated ({failure_tracker.streak}/{failure_tracker.threshold} consecutive).")
                return False
            # Non-fatal: caller abandons remaining jobs in this industry (existing behavior).
            print(f"     Halting further resume generation due to error: {error_msg[:100]}")
            raise

    # On a tolerated fatal failure, retry the same job once immediately; if
    # that also fails, defer it to a final make-up pass (after every job has
    # had its normal turn) instead of losing that job's resumes outright.
    jobs_needing_retry: list[tuple[int, JobPosting]] = []
    for job_i, job in enumerate(jobs):
        print(f"\n  Job {job_i + 1}/{len(jobs)}: {job.job_title}")
        try:
            ok = _attempt_resumes_for_job(job_i, job, "attempt 1")
            if not ok:
                print(f"     Retrying job '{job.job_title}'...")
                ok = _attempt_resumes_for_job(job_i, job, "retry")
            if not ok:
                print(f"     Deferring job '{job.job_title}' to a final make-up pass.")
                jobs_needing_retry.append((job_i, job))
        except Exception:
            # Fatal (streak hit threshold) already re-raises past this point.
            # Non-fatal: abandon remaining jobs in this industry, as before.
            print(f"\n⚠️  Skipped {len(jobs) - job_i - 1} remaining jobs in {industry}.\n")
            break

    if jobs_needing_retry:
        print(f"\n  Attempting to make up resumes for {len(jobs_needing_retry)} job(s) that failed earlier...")
        for job_i, job in jobs_needing_retry:
            try:
                ok = _attempt_resumes_for_job(job_i, job, "make-up")
                if not ok:
                    print(f"  ⚠️  Still could not generate resumes for job '{job.job_title}' after retries; "
                          f"accepting the shortfall (0 resumes for this job).")
            except Exception as exc:
                if is_fatal_llm_error(exc):
                    raise
                print(f"  ⚠️  Non-fatal error during make-up for job '{job.job_title}': {exc}")

    return resumes_generated


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic job postings and resumes for AI training datasets."
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=NUM_JOBS,
        help=f"Total job postings to generate across all industries (default: {NUM_JOBS})",
    )
    parser.add_argument(
        "--resumes-per-job",
        type=int,
        default=NUM_RESUMES_PER_JOB,
        help=f"Resumes to generate per job posting (default: {NUM_RESUMES_PER_JOB})",
    )
    parser.add_argument(
        "--max-industries",
        type=int,
        default=1,
        help="Number of industries to process (default: 1 for test runs; use 5 for full dataset)",
    )
    parser.add_argument(
        "--industry",
        type=str,
        default=None,
        help="Run a specific industry by partial name match (overrides --max-industries)",
    )
    parser.add_argument(
        "--job-max-completion-tokens",
        type=int,
        default=JOB_MAX_COMPLETION_TOKENS,
        help=f"Max completion tokens for job generation calls (default: {JOB_MAX_COMPLETION_TOKENS})",
    )
    parser.add_argument(
        "--resume-max-completion-tokens",
        type=int,
        default=RESUME_MAX_COMPLETION_TOKENS,
        help=f"Max completion tokens for resume generation calls (default: {RESUME_MAX_COMPLETION_TOKENS})",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=ITERATION,
        help=f"Iteration number used in log filename (default: {ITERATION} from .env)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing outputs for --iteration instead of overwriting: skips the "
             "backup-and-truncate step, opens all files in append mode, and continues "
             "job/resume/pair trace-ID numbering from the current max found in the existing "
             "files. Use this to run industries one at a time (e.g. --industry X, then "
             "--resume --industry Y) or to continue after an aborted run without losing or "
             "renumbering prior output.",
    )
    args = parser.parse_args()

    # Determine which industries to run
    if args.industry:
        industries_to_run = [i for i in INDUSTRIES if args.industry.lower() in i.lower()]
        if not industries_to_run:
            print(f"No industry matching '{args.industry}'.")
            print(f"Available industries:\n  " + "\n  ".join(INDUSTRIES))
            return
    else:
        industries_to_run = INDUSTRIES[: args.max_industries]

    if not industries_to_run:
        print("No industries selected.")
        return

    base_jobs_per_industry = args.num_jobs // len(industries_to_run)
    extra_jobs = args.num_jobs % len(industries_to_run)
    jobs_per_industry_plan = [
        base_jobs_per_industry + (1 if index < extra_jobs else 0)
        for index in range(len(industries_to_run))
    ]

    # Load templates
    try:
        templates = load_templates()
    except FileNotFoundError as exc:
        print(f"Template file not found: {exc}\nEnsure the templates/ directory contains all 5 JSON files.")
        return

    client = make_client()
    logger = Logger(iteration=args.iteration)

    jobs_path = DATA_DIR / f"jobs_{args.iteration}.jsonl"
    resumes_path = DATA_DIR / f"resumes_{args.iteration}.jsonl"
    pairs_path = DATA_DIR / f"pairs_{args.iteration}.jsonl"
    jobs_valid_path = DATA_DIR / f"jobs_valid_{args.iteration}.jsonl"
    jobs_invalid_path = DATA_DIR / f"jobs_invalid_{args.iteration}.jsonl"
    resumes_valid_path = DATA_DIR / f"resumes_valid_{args.iteration}.jsonl"
    resumes_invalid_path = DATA_DIR / f"resumes_invalid_{args.iteration}.jsonl"
    log_path = LOGS_DIR / f"dataset_log_{args.iteration}.jsonl"

    if args.resume:
        # Combined files carry every attempted slot (valid + invalid), so scanning
        # them for the current max trace ID avoids reissuing an ID already used by
        # an invalid attempt. Pairs only ever exist for valid resumes, so pairs_path
        # alone is authoritative there.
        job_counter = find_max_trace_id_counter("job", jobs_path) + 1
        resume_counter = find_max_trace_id_counter("res", resumes_path) + 1
        pair_counter = find_max_trace_id_counter("pair", pairs_path) + 1
        global_resume_offset = count_jsonl_lines(resumes_valid_path)
    else:
        backup_iteration_outputs(args.iteration)
        job_counter = 1
        resume_counter = 1
        pair_counter = 1
        global_resume_offset = 0

    # One tracker for the whole run, shared across every industry — see
    # ConsecutiveFailureTracker's docstring for why it isn't reset per industry.
    failure_tracker = ConsecutiveFailureTracker(MAX_CONSECUTIVE_FATAL_FAILURES)

    # Snapshot before this invocation writes anything, so the final summary can
    # report both "this run added N" and "file now has M total" from the actual
    # file contents — not from job_counter/resume_counter/pair_counter, which only
    # advance after a FULLY completed industry and would under-report if a fatal
    # error aborts partway through one (process_industry still writes each job/
    # resume to disk as it goes, right up until the failure).
    jobs_before = count_jsonl_lines(jobs_path)
    resumes_before = count_jsonl_lines(resumes_path)
    pairs_before = count_jsonl_lines(pairs_path)

    print("Dataset Generator")
    print(f"{'─' * 50}")
    print(f"  Iteration      : {args.iteration}")
    if args.resume:
        print(f"  Mode           : RESUME — appending, starting at job_{job_counter}/"
              f"res_{resume_counter}/pair_{pair_counter} ({global_resume_offset} valid resumes so far)")
    print(f"  Industries     : {industries_to_run}")
    print(f"  Jobs total     : {args.num_jobs}")
    print(f"  Jobs plan      : {dict(zip(industries_to_run, jobs_per_industry_plan, strict=False))}")
    print(f"  Resumes/job    : {args.resumes_per_job}")
    print(f"  Job max toks   : {args.job_max_completion_tokens}")
    print(f"  Resume max toks: {args.resume_max_completion_tokens}")
    print(f"  Gen model      : {GENERATION_MODEL}")
    print(f"  Judge model    : {JUDGE_MODEL} (stub)")
    print(f"  Data dir       : {DATA_DIR}")
    print(f"  Log file       : {log_path}")
    print(f"{'─' * 50}")

    logger.run_start(
        industries=industries_to_run,
        jobs_per_industry=args.num_jobs,
        resumes_per_job=args.resumes_per_job,
        generation_model=GENERATION_MODEL,
        iteration=args.iteration,
    )

    total_jobs = 0
    total_resumes = 0
    aborted = False

    file_mode = "a" if args.resume else "w"
    with (
        open(jobs_path, file_mode) as jf,
        open(jobs_valid_path, file_mode) as jvf,
        open(jobs_invalid_path, file_mode) as jif,
        open(resumes_path, file_mode) as rf,
        open(resumes_valid_path, file_mode) as rvf,
        open(resumes_invalid_path, file_mode) as rif,
        open(pairs_path, file_mode) as pf,
    ):
        for industry, jobs_for_industry in zip(industries_to_run, jobs_per_industry_plan, strict=False):
            try:
                n_resumes = process_industry(
                    client=client,
                    logger=logger,
                    industry=industry,
                    jobs_per_industry=jobs_for_industry,
                    resumes_per_job=args.resumes_per_job,
                    job_max_completion_tokens=args.job_max_completion_tokens,
                    resume_max_completion_tokens=args.resume_max_completion_tokens,
                    templates=templates,
                    jobs_file=jf,
                    jobs_valid_file=jvf,
                    jobs_invalid_file=jif,
                    resumes_file=rf,
                    resumes_valid_file=rvf,
                    resumes_invalid_file=rif,
                    pairs_file=pf,
                    global_resume_offset=global_resume_offset,
                    job_counter_start=job_counter,
                    resume_counter_start=resume_counter,
                    pair_counter_start=pair_counter,
                    failure_tracker=failure_tracker,
                )
            except Exception as exc:
                aborted = True
                print(f"\n{'=' * 60}")
                print(f"  ❌ FATAL: run aborted while processing '{industry}': {exc}")
                print(f"  Remaining industries were not processed.")
                print(f"{'=' * 60}")
                logger.run_aborted(industry=industry, error_message=str(exc), error=exc)
                break
            global_resume_offset += n_resumes
            total_resumes += n_resumes
            total_jobs += jobs_for_industry
            job_counter += jobs_for_industry
            # Advance by planned slots to keep trace IDs unique across the full iteration.
            resume_counter += jobs_for_industry * args.resumes_per_job
            pair_counter += jobs_for_industry * args.resumes_per_job

    if not aborted:
        logger.run_complete(
            total_jobs=total_jobs,
            total_resumes=total_resumes,
            jobs_path=str(jobs_path),
            resumes_path=str(resumes_path),
            pairs_path=str(pairs_path),
        )
    logger.close()

    jobs_after = count_jsonl_lines(jobs_path)
    resumes_after = count_jsonl_lines(resumes_path)
    pairs_after = count_jsonl_lines(pairs_path)

    print(f"\n{'=' * 60}")
    print(f"  {'Aborted — partial results below' if aborted else 'Complete!'}")
    print(f"  (This run added {jobs_after - jobs_before} job(s), {resumes_after - resumes_before} resume(s) "
          f"before {'aborting' if aborted else 'finishing'}.)")
    print(f"  Jobs written    : {jobs_path} ({jobs_after} records total)")
    print(f"  Resumes written : {resumes_path} ({resumes_after} records total)")
    print(f"  Pairs written   : {pairs_path} ({pairs_after} records total)")
    print(f"  Jobs valid      : {jobs_valid_path}")
    print(f"  Jobs invalid    : {jobs_invalid_path}")
    print(f"  Resumes valid   : {resumes_valid_path}")
    print(f"  Resumes invalid : {resumes_invalid_path}")
    print(f"  Log written     : {log_path}")
    print(f"{'=' * 60}")

    if aborted:
        sys.exit(1)


if __name__ == "__main__":
    main()
