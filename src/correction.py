#!/usr/bin/env python3
"""
Correction pass for invalid job and resume records from a prior
dataset_generator.py run.

Reads jobs_invalid_{iteration}.jsonl and resumes_invalid_{iteration}.jsonl,
asks the LLM to fix each record, re-validates with the SAME
validate_job/validate_resume functions used at generation time, and writes
results to *_corrected_{iteration}.jsonl / *_uncorrectable_{iteration}.jsonl.
Never mutates the original run's output files.

Usage:
    python correction.py --iteration 1
    python correction.py --iteration 1 --skip-resumes
    python correction.py --iteration 1 --limit-jobs 10 --limit-resumes 10
    python correction.py --iteration 1 --generate-resumes-for-corrected-jobs
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import instructor

from validation import JobPosting, JobContent, Resume, ResumeContent, ResumePair, RawJobBatch, RawResumeBatch, validate_job, validate_resume
from llm_errors import is_fatal_llm_error
from resume_generator import generate_resumes_for_job
from dataset_generator import (
    DATA_DIR, ITERATION, GENERATION_MODEL,
    JOB_MAX_COMPLETION_TOKENS, RESUME_MAX_COMPLETION_TOKENS, NUM_RESUMES_PER_JOB,
    make_client, load_templates, Logger, backup_correction_outputs,
    append_jsonl, append_invalid_with_error,
    format_job_summary, get_shortcoming_instructions,
    get_fit_level, find_max_trace_id_counter, LLM_REQUEST_TIMEOUT_SECONDS, _rate_limiter,
)


# ── I/O helpers ─────────────────────────────────────────────────────────────

def load_jsonl_pairs(path: Path) -> list[tuple[dict, dict]]:
    """Parse a *_invalid_{iteration}.jsonl file: consecutive (raw_record,
    error_details) line pairs, as written by append_invalid_with_error."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 2 != 0:
        print(f"  WARNING: {path} has an odd number of non-blank lines; dropping the trailing unpaired line.")
        lines = lines[:-1]
    pairs = []
    for i in range(0, len(lines), 2):
        raw_record = json.loads(lines[i])
        error_details = json.loads(lines[i + 1])
        pairs.append((raw_record, error_details))
    return pairs


def load_valid_jobs(iteration: int) -> dict[str, JobPosting]:
    """Read jobs_valid_{iteration}.jsonl into a trace_id -> JobPosting dict."""
    path = DATA_DIR / f"jobs_valid_{iteration}.jsonl"
    jobs: dict[str, JobPosting] = {}
    if not path.exists():
        return jobs
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        job = JobPosting.model_validate_json(line)
        jobs[job.trace_id] = job
    return jobs


# ── Job correction ────────────────────────────────────────────────────────────

JOB_BATCH_WRAPPER_SUFFIX = (
    '\n\nReturn your answer as {"jobs": [ <the corrected/generated job object> ]} '
    "— a single-element list."
)


def build_job_fix_prompt(raw_record: dict, error_details: dict) -> str:
    """'Fix ONLY the flagged fields, preserve everything else' prompt for a job
    that was returned by the LLM but failed schema validation."""
    industry = raw_record.get("industry", "Unknown")
    content_fields = {k: v for k, v in raw_record.items() if k not in ("trace_id", "generated_at", "industry")}
    schema_json = json.dumps(JobContent.model_json_schema(), indent=2)
    errors_json = json.dumps(error_details.get("validation_errors", []), indent=2)

    return f"""The following job posting was generated for the {industry} industry but failed schema
validation. Fix ONLY what is necessary to satisfy the validation errors below — preserve
the rest of the content (title, company, description, skills, etc.) as closely as possible.

═══ ORIGINAL RECORD ═══
{json.dumps(content_fields, indent=2)}

═══ VALIDATION ERRORS TO FIX ═══
{errors_json}

═══ REQUIRED JSON SCHEMA ═══
Use these exact key names at every level (top-level and nested). Do not invent or
substitute alternative field names:

{schema_json}"""


def correct_job_record(
    client: instructor.Instructor,
    raw_record: dict,
    error_details: dict,
    max_completion_tokens: int,
    logger: Logger,
) -> tuple[Optional[JobPosting], Optional[dict], Optional[dict]]:
    """Attempt to correct one invalid job record.

    Returns (job_posting, failure_error_details, corrected_raw_dict) — exactly
    one of the first two is non-None. Raises (does not catch) any exception
    for which is_fatal_llm_error() is True, so it propagates to the caller.
    """
    trace_id = raw_record.get("trace_id") or error_details["record_trace_id"]
    now = datetime.now(UTC).isoformat()
    error_type = error_details.get("error_type")
    logger.correction_attempt(kind="job", trace_id=trace_id, original_error_type=str(error_type))

    prompt = build_job_fix_prompt(raw_record, error_details) + JOB_BATCH_WRAPPER_SUFFIX

    estimated_tokens = len(prompt) // 4 + max_completion_tokens
    _rate_limiter.wait_if_needed(estimated_tokens)

    try:
        batch_response: RawJobBatch = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_model=RawJobBatch,
            max_retries=3,
            max_completion_tokens=max_completion_tokens,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if is_fatal_llm_error(exc):
            raise
        logger.correction_failed(kind="job", trace_id=trace_id, reason="llm_call_failed", error=exc)
        return None, {
            "record_trace_id": trace_id, "stage": "job_correction",
            "error_type": "correction_llm_error", "original_error_type": error_type,
            "validation_errors": [{"msg": str(exc)}],
        }, None

    if not batch_response.jobs:
        error = {
            "record_trace_id": trace_id, "stage": "job_correction",
            "error_type": "correction_failed_empty_response", "original_error_type": error_type,
            "validation_errors": [{"msg": "LLM returned zero jobs for a single-item correction request."}],
        }
        logger.correction_failed(kind="job", trace_id=trace_id, reason="empty_response")
        return None, error, None

    corrected_raw = batch_response.jobs[0]
    job_posting, revalidation_error = validate_job(corrected_raw, trace_id, now)
    if job_posting is not None:
        logger.correction_success(kind="job", trace_id=trace_id)
        return job_posting, None, corrected_raw

    revalidation_error["error_type"] = "correction_failed_still_invalid"
    revalidation_error["original_error_type"] = error_type
    revalidation_error["stage"] = "job_correction"
    logger.correction_failed(kind="job", trace_id=trace_id, reason="still_invalid_after_correction")
    return None, revalidation_error, corrected_raw


def run_job_corrections(
    client: instructor.Instructor,
    logger: Logger,
    jobs_invalid_records: list[tuple[dict, dict]],
    corrected_file,
    uncorrectable_file,
    max_completion_tokens: int,
    limit: Optional[int] = None,
) -> dict[str, JobPosting]:
    """Process invalid job records. Writes successes to corrected_file,
    failures (dual-line, same shape as the original invalid files) to
    uncorrectable_file. Returns trace_id -> JobPosting for every success, for
    merging into the resume-correction job_lookup. Propagates fatal exceptions."""
    records = jobs_invalid_records[:limit] if limit is not None else jobs_invalid_records
    corrected: dict[str, JobPosting] = {}
    for raw_record, error_details in records:
        job_posting, failure_error, _corrected_raw = correct_job_record(
            client, raw_record, error_details, max_completion_tokens, logger,
        )
        if job_posting is not None:
            append_jsonl(corrected_file, job_posting)
            corrected[job_posting.trace_id] = job_posting
        else:
            append_invalid_with_error(uncorrectable_file, raw_record, failure_error)
    return corrected


# ── Resume correction ─────────────────────────────────────────────────────────

RESUME_BATCH_WRAPPER_SUFFIX = (
    '\n\nReturn your answer as {"resumes": [ <the corrected/generated resume object> ]} '
    "— a single-element list."
)


def build_resume_fix_prompt(raw_record: dict, error_details: dict, job: JobPosting, template: dict, fit_level: str) -> str:
    """'Fix ONLY the flagged fields' prompt for a resume that was returned by
    the LLM but failed schema validation."""
    content_fields = {
        k: v for k, v in raw_record.items()
        if k not in ("trace_id", "generated_at", "job_trace_id", "job_title", "fit_level", "template")
    }
    schema_json = json.dumps(ResumeContent.model_json_schema(), indent=2)
    errors_json = json.dumps(error_details.get("validation_errors", []), indent=2)
    job_summary = format_job_summary(job)
    shortcoming_instructions = get_shortcoming_instructions(fit_level, job)

    return f"""The following resume was generated for the job below but failed schema validation.
Fix ONLY what is necessary to satisfy the validation errors — preserve the rest of the
content (contact, experience, education, skills, summary) as closely as possible while
keeping it consistent with the job context and intended fit level.

═══ JOB CONTEXT ═══
{job_summary}

═══ FIT LEVEL: {fit_level.upper()} ═══
{shortcoming_instructions}

═══ ORIGINAL RECORD ═══
{json.dumps(content_fields, indent=2)}

═══ VALIDATION ERRORS TO FIX ═══
{errors_json}

═══ REQUIRED JSON SCHEMA ═══
Use these exact key names at every level (top-level and nested). Do not invent or
substitute alternative field names (e.g. no `work_experience`, `contact_information`,
`professional_summary`, `certifications`, `projects`, `tags`):

{schema_json}"""


def correct_resume_record(
    client: instructor.Instructor,
    raw_record: dict,
    error_details: dict,
    job_lookup: dict[str, JobPosting],
    templates: dict[str, dict],
    max_completion_tokens: int,
    logger: Logger,
) -> tuple[Optional[Resume], Optional[JobPosting], Optional[dict], Optional[dict]]:
    """Attempt to correct one invalid resume record.

    Returns (resume, job_for_pairing, failure_error_details, corrected_raw_dict).
    If job_trace_id / fit_level / template context is missing from raw_record,
    or the referenced job isn't in job_lookup, returns immediately WITHOUT
    calling the LLM. Raises any is_fatal_llm_error() exception; everything
    else is caught and returned as a failure tuple.
    """
    trace_id = raw_record.get("trace_id") or error_details["record_trace_id"]
    now = datetime.now(UTC).isoformat()
    error_type = error_details.get("error_type")

    job_trace_id = raw_record.get("job_trace_id")
    fit_level = raw_record.get("fit_level")
    template_name = raw_record.get("template")
    job = job_lookup.get(job_trace_id) if job_trace_id else None
    template = templates.get(template_name) if template_name else None

    if job is None or fit_level is None or template is None:
        error = {
            "record_trace_id": trace_id, "stage": "resume_correction",
            "error_type": "correction_skipped_missing_job_context", "original_error_type": error_type,
            "validation_errors": [{
                "msg": f"Cannot correct without resolvable job context "
                       f"(job_trace_id={job_trace_id!r}, fit_level={fit_level!r}, template={template_name!r})."
            }],
        }
        logger.correction_failed(kind="resume", trace_id=trace_id, reason="missing_job_context")
        return None, None, error, None

    logger.correction_attempt(kind="resume", trace_id=trace_id, original_error_type=str(error_type))

    prompt = build_resume_fix_prompt(raw_record, error_details, job, template, fit_level) + RESUME_BATCH_WRAPPER_SUFFIX

    estimated_tokens = len(prompt) // 4 + max_completion_tokens
    _rate_limiter.wait_if_needed(estimated_tokens)

    try:
        batch_response: RawResumeBatch = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_model=RawResumeBatch,
            max_retries=3,
            max_completion_tokens=max_completion_tokens,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if is_fatal_llm_error(exc):
            raise
        logger.correction_failed(kind="resume", trace_id=trace_id, reason="llm_call_failed", error=exc)
        return None, None, {
            "record_trace_id": trace_id, "stage": "resume_correction",
            "error_type": "correction_llm_error", "original_error_type": error_type,
            "validation_errors": [{"msg": str(exc)}],
        }, None

    if not batch_response.resumes:
        error = {
            "record_trace_id": trace_id, "stage": "resume_correction",
            "error_type": "correction_failed_empty_response", "original_error_type": error_type,
            "validation_errors": [{"msg": "LLM returned zero resumes for a single-item correction request."}],
        }
        logger.correction_failed(kind="resume", trace_id=trace_id, reason="empty_response")
        return None, job, error, None

    corrected_raw = batch_response.resumes[0]
    resume, revalidation_error = validate_resume(corrected_raw, trace_id, now, fit_level, template)
    if resume is not None:
        return resume, job, None, corrected_raw

    revalidation_error["error_type"] = "correction_failed_still_invalid"
    revalidation_error["original_error_type"] = error_type
    revalidation_error["stage"] = "resume_correction"
    logger.correction_failed(kind="resume", trace_id=trace_id, reason="still_invalid_after_correction")
    return None, job, revalidation_error, corrected_raw


def make_corrected_pair(resume: Resume, job: JobPosting, fit_level: str, template_name: str) -> ResumePair:
    """Pair trace_id reuses the number embedded in the resume's own preserved
    trace_id (res_N -> pair_N). Since pair numbering is always 1:1 with resume
    numbering in the original run, this is guaranteed collision-free against
    the existing pairs_{iteration}.jsonl without needing a new counter."""
    return ResumePair(
        trace_id="pair_" + resume.trace_id.removeprefix("res_"),
        job_trace_id=job.trace_id,
        resume_trace_id=resume.trace_id,
        fit_level=fit_level,
        template_style=template_name,
        judge_score=None,
        generated_at=resume.generated_at,
    )


def run_resume_corrections(
    client: instructor.Instructor,
    logger: Logger,
    resumes_invalid_records: list[tuple[dict, dict]],
    job_lookup: dict[str, JobPosting],
    templates: dict[str, dict],
    resumes_corrected_file,
    resumes_uncorrectable_file,
    pairs_corrected_file,
    max_completion_tokens: int,
    limit: Optional[int] = None,
) -> None:
    """Process invalid resume records. For each success, also writes
    the ResumePair — this is the only place ResumePairs get created for
    corrected records, and only happens when the job resolves in job_lookup.
    Propagates fatal exceptions."""
    records = resumes_invalid_records[:limit] if limit is not None else resumes_invalid_records
    for raw_record, error_details in records:
        resume, job, failure_error, _corrected_raw = correct_resume_record(
            client, raw_record, error_details, job_lookup, templates, max_completion_tokens, logger,
        )
        if resume is not None and job is not None:
            fit_level = raw_record.get("fit_level")
            template_name = raw_record.get("template")
            pair = make_corrected_pair(resume, job, fit_level, template_name)
            append_jsonl(resumes_corrected_file, resume)
            append_jsonl(pairs_corrected_file, pair)
            logger.correction_success(kind="resume", trace_id=resume.trace_id, pair_trace_id=pair.trace_id)
        else:
            append_invalid_with_error(resumes_uncorrectable_file, raw_record, failure_error)


# ── Follow-on: generate resumes for jobs that are short of them ──────────────
#
# Two situations feed into this same mechanism:
#   1. A job was invalid at generation time and never went through resume
#      generation at all (phase 2 there only loops over jobs already valid at
#      that point) — becomes valid via correct_job_record, needs a full set.
#   2. A job was valid and DID have resumes attempted, but the run aborted
#      partway through phase 2 (or hit a shortfall the retry/make-up passes
#      in dataset_generator.py couldn't fully cover) — needs however many
#      more it's short of `--resumes-per-job`/`--resumes-per-corrected-job`.
# Both are "this job needs N more resumes" — same generate_resumes_for_job()
# call, just a different reason N > 0.

def find_jobs_missing_resumes(
    iteration: int, jobs: dict[str, JobPosting], resumes_per_job: int,
) -> dict[str, int]:
    """Returns {job_trace_id: shortfall} for every job in `jobs` with fewer
    than `resumes_per_job` existing pairs in pairs_{iteration}.jsonl +
    pairs_corrected_{iteration}.jsonl (a job with zero pairs has shortfall ==
    resumes_per_job). Only jobs with shortfall > 0 are included."""
    pair_counts: dict[str, int] = defaultdict(int)
    for path in (DATA_DIR / f"pairs_{iteration}.jsonl", DATA_DIR / f"pairs_corrected_{iteration}.jsonl"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            job_trace_id = json.loads(line).get("job_trace_id")
            if job_trace_id:
                pair_counts[job_trace_id] += 1

    return {
        trace_id: resumes_per_job - pair_counts.get(trace_id, 0)
        for trace_id in jobs
        if pair_counts.get(trace_id, 0) < resumes_per_job
    }


def generate_resumes_for_jobs(
    client: instructor.Instructor,
    logger: Logger,
    iteration: int,
    job_resume_counts: dict[str, tuple[JobPosting, int]],
    templates: dict[str, dict],
    max_completion_tokens: int,
    resumes_corrected_file,
    resumes_uncorrectable_file,
    pairs_corrected_file,
) -> tuple[int, int]:
    """Generates exactly `count` more resumes for each (job, count) in
    job_resume_counts.values(), reusing generate_resumes_for_job() unchanged.
    Returns (resumes_written, invalid_written).

    Counter safety: scans resumes_{iteration}.jsonl + resumes_corrected_{iteration}.jsonl
    for the current high-water mark and continues from there, so get_fit_level()'s
    dataset-wide Excellent/Good/Partial/Poor cycling isn't disrupted and no
    trace_id collides with an existing resume.
    """
    next_n = find_max_trace_id_counter(
        "res", DATA_DIR / f"resumes_{iteration}.jsonl", DATA_DIR / f"resumes_corrected_{iteration}.jsonl",
    ) + 1
    resumes_written = 0
    invalid_written = 0
    for job, count in job_resume_counts.values():
        if count <= 0:
            continue
        fit_levels = [get_fit_level(next_n - 1 + r) for r in range(count)]
        print(f"  {job.trace_id} ({job.job_title}): generating {count} resume(s)...")
        resume_pairs, invalid_resumes = generate_resumes_for_job(
            client=client,
            logger=logger,
            job=job,
            templates=templates,
            fit_levels=fit_levels,
            global_resume_offset=next_n - 1,
            resume_counter_start=next_n,
            pair_counter_start=next_n,
            max_completion_tokens=max_completion_tokens,
        )
        for resume, pair in resume_pairs:
            append_jsonl(resumes_corrected_file, resume)
            append_jsonl(pairs_corrected_file, pair)
            resumes_written += 1
        for raw_resume, error_details in invalid_resumes:
            append_invalid_with_error(resumes_uncorrectable_file, raw_resume, error_details)
            invalid_written += 1
        next_n += count
    return resumes_written, invalid_written


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct invalid job and resume records from a prior dataset_generator.py run."
    )
    parser.add_argument(
        "--iteration", type=int, default=ITERATION,
        help=f"Iteration to correct — must match an existing dataset_generator.py run (default: {ITERATION} from .env)",
    )
    parser.add_argument("--job-max-completion-tokens", type=int, default=JOB_MAX_COMPLETION_TOKENS)
    parser.add_argument("--resume-max-completion-tokens", type=int, default=RESUME_MAX_COMPLETION_TOKENS)
    parser.add_argument("--limit-jobs", type=int, default=None, help="Cap number of invalid job records processed (testing/cost control)")
    parser.add_argument("--limit-resumes", type=int, default=None, help="Cap number of invalid resume records processed")
    parser.add_argument("--skip-jobs", action="store_true")
    parser.add_argument("--skip-resumes", action="store_true")
    parser.add_argument(
        "--generate-resumes-for-corrected-jobs", action="store_true",
        help="Opt-in: also generate a first batch of resumes for jobs that became valid via correction (additional LLM cost)",
    )
    parser.add_argument("--resumes-per-corrected-job", type=int, default=NUM_RESUMES_PER_JOB)
    parser.add_argument(
        "--backfill-missing-resumes", action="store_true",
        help="Opt-in: generate resumes for any valid job with fewer than --resumes-per-job existing "
             "pairs — e.g. jobs that never got resumes because a run aborted partway through phase 2, "
             "or where the retry/make-up passes in dataset_generator.py still fell short. Independent "
             "of --generate-resumes-for-corrected-jobs; runs even if there are no invalid records to correct.",
    )
    parser.add_argument("--resumes-per-job", type=int, default=NUM_RESUMES_PER_JOB,
                         help=f"Target resume count per job for --backfill-missing-resumes (default: {NUM_RESUMES_PER_JOB})")
    args = parser.parse_args()

    jobs_invalid_path = DATA_DIR / f"jobs_invalid_{args.iteration}.jsonl"
    resumes_invalid_path = DATA_DIR / f"resumes_invalid_{args.iteration}.jsonl"
    jobs_corrected_path = DATA_DIR / f"jobs_corrected_{args.iteration}.jsonl"
    jobs_uncorrectable_path = DATA_DIR / f"jobs_uncorrectable_{args.iteration}.jsonl"
    resumes_corrected_path = DATA_DIR / f"resumes_corrected_{args.iteration}.jsonl"
    resumes_uncorrectable_path = DATA_DIR / f"resumes_uncorrectable_{args.iteration}.jsonl"
    pairs_corrected_path = DATA_DIR / f"pairs_corrected_{args.iteration}.jsonl"

    if not jobs_invalid_path.exists() and not resumes_invalid_path.exists() and not args.backfill_missing_resumes:
        print(f"No jobs_invalid_{args.iteration}.jsonl or resumes_invalid_{args.iteration}.jsonl found, "
              f"and --backfill-missing-resumes wasn't passed. Nothing to do.")
        return

    backup_correction_outputs(args.iteration)

    templates = load_templates()
    client = make_client()
    logger = Logger(iteration=args.iteration)  # appends into the SAME dataset_log_{iteration}.jsonl

    jobs_invalid_records = load_jsonl_pairs(jobs_invalid_path)
    resumes_invalid_records = load_jsonl_pairs(resumes_invalid_path)
    logger.correction_run_start(args.iteration, len(jobs_invalid_records), len(resumes_invalid_records))

    print(f"{'─' * 50}")
    print(f"  Correction pass — iteration {args.iteration}")
    print(f"  Invalid jobs    : {len(jobs_invalid_records)}")
    print(f"  Invalid resumes : {len(resumes_invalid_records)}")
    print(f"{'─' * 50}")

    aborted = False
    corrected_jobs: dict[str, JobPosting] = {}
    jobs_corrected_count = 0
    jobs_uncorrectable_count = 0
    job_lookup = load_valid_jobs(args.iteration)

    try:
        with (
            open(jobs_corrected_path, "w") as jcf,
            open(jobs_uncorrectable_path, "w") as juf,
            open(resumes_corrected_path, "w") as rcf,
            open(resumes_uncorrectable_path, "w") as ruf,
            open(pairs_corrected_path, "w") as pcf,
        ):
            if not args.skip_jobs and jobs_invalid_records:
                print("\n  Correcting jobs...")
                corrected_jobs = run_job_corrections(
                    client, logger, jobs_invalid_records, jcf, juf,
                    args.job_max_completion_tokens, args.limit_jobs,
                )
                attempted_jobs = jobs_invalid_records[:args.limit_jobs] if args.limit_jobs is not None else jobs_invalid_records
                jobs_corrected_count = len(corrected_jobs)
                jobs_uncorrectable_count = len(attempted_jobs) - jobs_corrected_count
                job_lookup.update(corrected_jobs)  # so resume correction/pairing sees newly-valid jobs too
                print(f"  Jobs: {jobs_corrected_count} corrected, {jobs_uncorrectable_count} uncorrectable")

            if not args.skip_resumes and resumes_invalid_records:
                print("\n  Correcting resumes...")
                run_resume_corrections(
                    client, logger, resumes_invalid_records, job_lookup, templates,
                    rcf, ruf, pcf, args.resume_max_completion_tokens, args.limit_resumes,
                )

            if args.generate_resumes_for_corrected_jobs and corrected_jobs:
                print(f"\n  Generating resumes for {len(corrected_jobs)} newly-valid job(s)...")
                job_resume_counts = {
                    trace_id: (job, args.resumes_per_corrected_job) for trace_id, job in corrected_jobs.items()
                }
                generate_resumes_for_jobs(
                    client, logger, args.iteration, job_resume_counts, templates,
                    args.resume_max_completion_tokens, rcf, ruf, pcf,
                )

            if args.backfill_missing_resumes:
                shortfalls = find_jobs_missing_resumes(args.iteration, job_lookup, args.resumes_per_job)
                if shortfalls:
                    print(f"\n  Backfilling resumes for {len(shortfalls)} job(s) missing some/all resumes...")
                    job_resume_counts = {
                        trace_id: (job_lookup[trace_id], count) for trace_id, count in shortfalls.items()
                    }
                    written, invalid = generate_resumes_for_jobs(
                        client, logger, args.iteration, job_resume_counts, templates,
                        args.resume_max_completion_tokens, rcf, ruf, pcf,
                    )
                    print(f"  Backfill: {written} resume(s) written, {invalid} invalid.")
                else:
                    print("\n  No jobs are missing resumes — nothing to backfill.")
    except Exception as exc:
        aborted = True
        print(f"\n❌ Correction run aborted due to a fatal LLM/API error: {exc}")
        logger.run_aborted(industry="correction", error_message=str(exc), error=exc)

    if not aborted:
        logger.correction_run_complete(
            iteration=args.iteration,
            jobs_corrected=jobs_corrected_count,
            jobs_uncorrectable=jobs_uncorrectable_count,
            resumes_corrected=0,  # exact counts are visible per-record in the log stream
            resumes_uncorrectable=0,
        )
        print(f"\n{'=' * 60}")
        print("  Correction complete!")
        print(f"  Jobs corrected     : {jobs_corrected_path}")
        print(f"  Jobs uncorrectable : {jobs_uncorrectable_path}")
        print(f"  Resumes corrected  : {resumes_corrected_path}")
        print(f"  Resumes uncorrectable: {resumes_uncorrectable_path}")
        print(f"  Pairs corrected    : {pairs_corrected_path}")
        print()
        print(f"  jobs_valid_{args.iteration}.jsonl + jobs_corrected_{args.iteration}.jsonl = full valid job set.")
        print(f"  resumes_valid_{args.iteration}.jsonl + resumes_corrected_{args.iteration}.jsonl = full valid resume set.")
        print(f"  pairs_{args.iteration}.jsonl + pairs_corrected_{args.iteration}.jsonl = full pair set.")
        print(f"{'=' * 60}")
    logger.close()
    if aborted:
        sys.exit(1)


if __name__ == "__main__":
    main()
