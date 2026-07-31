#!/usr/bin/env python3
"""
Resume batch generator for a specific job posting.

This module handles generating multiple resumes for a single job,
with configurable fit levels and writing styles. It can be imported
as a library for reusable resume generation workflows.
"""

import json
import uuid
import time
from datetime import UTC, datetime
from typing import Optional

import instructor

from validation import Resume, ResumePair, ResumeContent, RawResumeBatch, JobPosting, validate_resume
from llm_errors import is_fatal_llm_error

# ── Type stub for Logger (avoid circular import with dataset_generator) ───────

# In real usage, Logger is imported from dataset_generator.
# The function signature uses duck typing for it (dataset_generator also
# imports generate_resumes_for_job from this module at top level, so the
# names it owns — GENERATION_MODEL, get_template_name, get_shortcoming_instructions,
# format_job_summary, RESUME_MAX_BATCH_SIZE, _rate_limiter — must stay deferred,
# in-function imports here to avoid a circular import).


def generate_resumes_for_job(
    client: instructor.Instructor,
    logger: "Logger",  # type: ignore
    job: JobPosting,
    templates: dict[str, dict],
    fit_levels: list[str],
    global_resume_offset: int,
    resume_counter_start: int,
    pair_counter_start: int,
    max_completion_tokens: int,
) -> tuple[list[tuple[Resume, ResumePair]], list[tuple[dict, dict]]]:
    """
    Generate multiple resumes for a single job posting (in batches).

    Args:
        client: Groq instructor client configured with structured output
        logger: Logger instance for event tracking
        job: JobPosting object with job_id, job_title, requirements, etc.
        templates: Dict mapping template names to template configs
        fit_levels: List of fit level strings (e.g., ['Excellent', 'Good', 'Partial', 'Poor'])
        global_resume_offset: Starting index for resume numbering (for fit level distribution)
        resume_counter_start: Starting global resume counter for this job
        pair_counter_start: Starting global pair counter for this job
        max_completion_tokens: Token budget for each resume generation call

    Returns:
        List of (Resume, ResumePair) tuples. If any resume fails to generate,
        raises an exception to halt processing (abort-on-failure pattern).

    Raises:
        Exception: If any resume generation fails, with full context in logger and exception message.
    """
    # Import here to avoid circular dependency at module load
    from dataset_generator import (
        get_template_name,
        RESUME_MAX_BATCH_SIZE,
    )

    results: list[tuple[Resume, ResumePair]] = []
    invalid_resumes: list[tuple[dict, dict]] = []
    resumes_per_job = len(fit_levels)

    print(f"\n    Generating {resumes_per_job} resumes for job '{job.job_title}'...")

    # Split resumes into batches
    num_batches = (resumes_per_job + RESUME_MAX_BATCH_SIZE - 1) // RESUME_MAX_BATCH_SIZE
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * RESUME_MAX_BATCH_SIZE
        batch_end = min(batch_start + RESUME_MAX_BATCH_SIZE, resumes_per_job)
        batch_fit_levels = fit_levels[batch_start:batch_end]
        batch_size = len(batch_fit_levels)
        batch_templates = [templates[get_template_name(i)] for i in range(batch_start, batch_end)]
        
        print(f"      [Batch {batch_idx + 1}] Generating {batch_size} resumes...", end=" ", flush=True)
        
        try:
            batch_resumes, batch_invalid_resumes = _generate_resumes_batch(
                client=client,
                job=job,
                batch_templates=batch_templates,
                batch_fit_levels=batch_fit_levels,
                batch_start_index=batch_start,
                resume_counter_start=resume_counter_start + batch_start,
                max_completion_tokens=max_completion_tokens,
                logger=logger,
            )
            
            # Create pairs for each resume in the batch
            for i, resume in enumerate(batch_resumes):
                template_name = get_template_name(batch_start + i)
                fit_level = batch_fit_levels[i]
                
                pair_number = pair_counter_start + batch_start + i
                pair = ResumePair(
                    trace_id=f"pair_{pair_number}",
                    job_trace_id=job.trace_id,
                    resume_trace_id=resume.trace_id,
                    fit_level=fit_level,
                    template_style=template_name,
                    judge_score=None,  # TODO: implement judge step
                    generated_at=resume.generated_at,
                )
                results.append((resume, pair))

            invalid_resumes.extend(batch_invalid_resumes)
            
            print(f"OK ({batch_size} resumes)")
            
        except Exception as exc:
            # Log the error and re-raise to halt processing for this job
            error_msg = str(exc)
            print(f"ERR")
            logger.llm_error(
                context=f"resumes_batch_generation job_trace_id={job.trace_id!r} title={job.job_title!r} batch={batch_idx} size={batch_size}",
                trace_id="N/A",
                error_message=error_msg,
                error=exc,
            )
            if is_fatal_llm_error(exc):
                # Bare raise preserves the original exception type/identity so
                # process_industry's classifier upstream still recognizes it as fatal.
                raise
            raise RuntimeError(
                f"Resume batch generation failed for job '{job.job_title}' (job_trace_id={job.trace_id}), "
                f"batch {batch_idx}: {error_msg[:200]}"
            ) from exc

    return results, invalid_resumes


def build_single_resume_prompt(job: JobPosting, template: dict, fit_level: str, resume_index_within_job: int) -> str:
    """Prompt for generating exactly one resume for one job. Used by
    _generate_single_resume, and by correction.py's missing-resume regeneration path."""
    from dataset_generator import get_shortcoming_instructions, format_job_summary

    shortcoming_instructions = get_shortcoming_instructions(fit_level, job)
    job_summary = format_job_summary(job)

    return f"""Generate a realistic resume for a candidate applying to the following job.

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


def _generate_single_resume(
    client: instructor.Instructor,
    job: JobPosting,
    template: dict,
    fit_level: str,
    trace_id: str,
    resume_index_within_job: int,
    max_completion_tokens: int,
) -> Resume:
    """
    Generate a single resume matching a job with specified fit level and template.

    This is a low-level function called by generate_resumes_for_job.
    It does not catch exceptions; errors propagate to caller.

    Args:
        client: Groq instructor client
        job: JobPosting object
        template: Template config dict with 'display_name', 'instructions', etc.
        fit_level: One of 'Excellent', 'Good', 'Partial', 'Poor'
        trace_id: Unique trace ID for this generation attempt
        resume_index_within_job: 0-based index among resumes for this job
        max_completion_tokens: Token budget for this API call

    Returns:
        Resume object with all fields populated.

    Raises:
        Any exception from the LLM API (e.g., validation, rate limit, etc.)
    """
    from dataset_generator import GENERATION_MODEL

    now = datetime.now(UTC).isoformat()
    prompt = build_single_resume_prompt(job, template, fit_level, resume_index_within_job)

    # Rate limit to prevent exceeding TPM cap
    from dataset_generator import _rate_limiter
    estimated_tokens = len(prompt) // 4 + max_completion_tokens  # rough estimate
    _rate_limiter.wait_if_needed(estimated_tokens)

    content: ResumeContent = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=ResumeContent,
        max_retries=3,
        max_completion_tokens=max_completion_tokens,
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


def _generate_resumes_batch(
    client: instructor.Instructor,
    job: JobPosting,
    batch_templates: list[dict],
    batch_fit_levels: list[str],
    batch_start_index: int,
    resume_counter_start: int,
    max_completion_tokens: int,
    logger: "Logger",  # type: ignore
) -> tuple[list[Resume], list[tuple[dict, dict]]]:
    """
    Generate multiple resumes in a single batch LLM call.
    
    Args:
        client: Groq instructor client
        job: JobPosting object with job details
        batch_templates: List of template configs for this batch
        batch_fit_levels: List of fit levels for this batch
        batch_start_index: Starting index for logging (0-based)
        resume_counter_start: Starting global resume counter for this batch
        max_completion_tokens: Token budget for the batch API call
        logger: Logger instance for event tracking
    
    Returns:
        List of Resume objects
    
    Raises:
        Exception from LLM API if generation fails
    """
    from dataset_generator import GENERATION_MODEL, get_fit_level_brief, format_job_summary

    batch_size = len(batch_templates)
    now = datetime.now(UTC).isoformat()
    trace_id = f"batch_{str(uuid.uuid4())[:8]}"
    
    # Build resume specs for the batch
    resume_specs = []
    for i, (template, fit_level) in enumerate(zip(batch_templates, batch_fit_levels)):
        spec = (
            f"Resume {i + 1}:\n"
            f"  - Template: {template['display_name']} ({template['writing_style']} style)\n"
            f"  - Fit Level: {fit_level}\n"
        )
        resume_specs.append(spec)
    
    specs_prompt = "\n".join(resume_specs)

    job_summary = format_job_summary(job)

    # Build a compact fit legend plus short per-resume tags.
    fit_legend = (
        "Fit level legend:\n"
        "- Excellent: strong, highly aligned candidate.\n"
        "- Good: mostly aligned with minor gaps.\n"
        "- Partial: relevant but with clear gaps.\n"
        "- Poor: noticeably misaligned but still realistic."
    )
    fit_tags = []
    for i, fit_level in enumerate(batch_fit_levels):
        fit_tags.append(f"Resume {i + 1}: {get_fit_level_brief(fit_level)}")
    fit_tags_prompt = "\n".join(fit_tags)

    resume_schema_json = json.dumps(ResumeContent.model_json_schema(), indent=2)

    prompt = f"""Generate exactly {batch_size} distinct, realistic resumes for the following job posting.

═══ JOB CONTEXT ═══
{job_summary}

═══ RESUMES TO GENERATE ═══
{specs_prompt}

═══ FIT LEVEL GUIDE ═══
{fit_legend}

═══ RESUME TAGS ═══
{fit_tags_prompt}

═══ REQUIRED JSON SCHEMA FOR EACH RESUME OBJECT ═══
Each entry in `resumes` MUST conform exactly to this JSON Schema — use these exact key names
at every level (top-level and nested). Do not invent or substitute alternative field names
(e.g. no `work_experience`, `contact_information`, `professional_summary`, `certifications`,
`projects`, `tags`, or any other fields not defined below):

{resume_schema_json}

Additional constraints not fully captured by the schema:
- end_date must always be chronologically after start_date (use "Present" for current roles)
- Include 2–4 work experience entries
- Write a 2–4 sentence professional summary in the style of the template

Generate {batch_size} unique resumes as a batch. Each resume should be a distinct, realistic individual."""

    # Log batch start
    for i, (template, fit_level) in enumerate(zip(batch_templates, batch_fit_levels)):
        logger.resume_generation_start(
            job_trace_id=job.trace_id,
            job_title=job.job_title,
            resume_index=batch_start_index + i,
            fit_level=fit_level,
            template=template["name"],
            trace_id=trace_id,
        )
    
    # max_completion_tokens is a shared budget for the entire batch response (not per-resume)
    from dataset_generator import _rate_limiter
    estimated_tokens = len(prompt) // 4 + max_completion_tokens
    _rate_limiter.wait_if_needed(estimated_tokens)

    # Call LLM for batch generation
    batch_response: RawResumeBatch = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=RawResumeBatch,
        max_retries=3,
        max_completion_tokens=max_completion_tokens,
    )

    if len(batch_response.resumes) < batch_size:
        logger.batch_incomplete(
            context=f"resumes_batch_generation job_trace_id={job.trace_id!r} title={job.job_title!r}",
            trace_id=trace_id,
            expected=batch_size,
            actual=len(batch_response.resumes),
        )

    # Validate each raw resume independently so invalid items are still persisted upstream.
    resumes: list[Resume] = []
    invalid_resumes: list[tuple[dict, dict]] = []
    for i, raw_resume in enumerate(batch_response.resumes):
        template = batch_templates[i]
        fit_level = batch_fit_levels[i]
        
        resume_number = resume_counter_start + i
        trace_id = f"res_{resume_number}"
        raw_record = dict(raw_resume) if isinstance(raw_resume, dict) else {"raw_content": raw_resume}
        raw_record["trace_id"] = trace_id
        raw_record["generated_at"] = now
        raw_record["job_trace_id"] = job.trace_id
        raw_record["job_title"] = job.job_title
        raw_record["fit_level"] = fit_level
        raw_record["template"] = template["name"]

        resume, error_details = validate_resume(raw_resume, trace_id, now, fit_level, template)
        if resume is not None:
            resumes.append(resume)
        else:
            invalid_resumes.append((raw_record, error_details))

    # Slots the LLM silently dropped from the batch response never reach the loop above.
    for i in range(len(batch_response.resumes), batch_size):
        resume_number = resume_counter_start + i
        missing_trace_id = f"res_{resume_number}"
        raw_record = {
            "trace_id": missing_trace_id,
            "generated_at": now,
            "fit_level": batch_fit_levels[i],
            "template": batch_templates[i]["name"],
            "job_trace_id": job.trace_id,
            "job_title": job.job_title,
        }
        error_details = {
            "record_trace_id": missing_trace_id,
            "stage": "resume_generation",
            "error_type": "llm_missing_resume",
            "validation_errors": [
                {"msg": f"LLM returned {len(batch_response.resumes)} of {batch_size} requested resumes; this slot was never generated."}
            ],
        }
        invalid_resumes.append((raw_record, error_details))

    return resumes, invalid_resumes
