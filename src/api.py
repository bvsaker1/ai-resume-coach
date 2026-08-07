#!/usr/bin/env python3
"""
FastAPI service exposing the same job/resume matching logic as analysis.py.

Endpoints:
    POST /review-resume            Score one job/resume pair on demand.
                                    Rule-based by default; set use_llm_judge
                                    to true to route through JUDGE_MODEL instead.
    GET  /health                   Liveness check.
    GET  /analysis/failure-rates   Serves an already-computed
                                    failure_labels_{iteration}.jsonl (written
                                    by `python analysis.py`), plus aggregate
                                    failure rates.

Shares its scoring logic with analysis.py (score_pair / run_llm_judgment) —
no duplicated rules between the CLI script and this API.

Must be run with src/ as the working directory, same as every other script
in this project:
    cd src
    uvicorn api:app --reload --port 8000
    # or: python api.py
"""

import json
from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from validation import JobContent, ResumeContent, JobPosting, Resume
from dataset_generator import DATA_DIR, ITERATION, JUDGE_MODEL, JUDGE_MAX_COMPLETION_TOKENS, make_client
from llm_errors import is_fatal_llm_error
from analysis import score_pair, run_llm_judgment, SKILLS_OVERLAP_THRESHOLD

app = FastAPI(
    title="Resume/Job Match API",
    description="Scores job/resume pairs against the same 6 rules used by analysis.py.",
)

# Judge client is created lazily on first use, not at import time — the
# default is no LLM judge, so most deployments never need it.
_judge_client = None


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        _judge_client = make_client(JUDGE_MODEL)
    return _judge_client


def _build_job_and_resume(job: JobContent, resume: ResumeContent) -> tuple[JobPosting, Resume]:
    """Synthesize the system-metadata fields JobContent/ResumeContent don't
    carry, mirroring what generation does at validation time. prompt_template/
    fit_level/writing_style are required Resume fields but are never read by
    any scoring function, so placeholders are safe here."""
    now = datetime.now(UTC).isoformat()
    job_posting = JobPosting(trace_id=f"api_job_{uuid4().hex[:8]}", generated_at=now, **job.model_dump())
    resume_obj = Resume(
        trace_id=f"api_res_{uuid4().hex[:8]}", generated_at=now,
        prompt_template="api_submission", fit_level="Unknown", writing_style="",
        **resume.model_dump(),
    )
    return job_posting, resume_obj


# ── Schemas ─────────────────────────────────────────────────────────────────

class ReviewResumeRequest(BaseModel):
    job: JobContent
    resume: ResumeContent
    use_llm_judge: bool = False
    skills_overlap_threshold: float = SKILLS_OVERLAP_THRESHOLD


class FailureLabel(BaseModel):
    trace_id: str
    industry: str
    labeler: str
    skills_overlap: float
    experience_mismatch: int
    level_mismatch: int
    missing_skills: int
    hallucinate_skills: int
    awkward_language: int
    job_resume_match: int


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/review-resume", response_model=FailureLabel)
def review_resume(body: ReviewResumeRequest) -> FailureLabel:
    job_posting, resume_obj = _build_job_and_resume(body.job, body.resume)

    if body.use_llm_judge:
        try:
            client = _get_judge_client()
            judgment = run_llm_judgment(
                client, JUDGE_MODEL, job_posting, resume_obj,
                body.skills_overlap_threshold, JUDGE_MAX_COMPLETION_TOKENS,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=f"LLM judge not configured: {exc}") from exc
        except Exception as exc:
            status = 502
            detail = f"LLM judge call failed: {exc}"
            raise HTTPException(status_code=status, detail=detail) from exc
        fields = judgment.model_dump()
        labeler = "llm_judge"
    else:
        fields = score_pair(job_posting, resume_obj, body.skills_overlap_threshold)
        labeler = "judge"

    return FailureLabel(
        trace_id=f"review_{uuid4().hex[:8]}",
        industry=job_posting.company.industry,
        labeler=labeler,
        **fields,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/analysis/failure-rates")
def failure_rates(
    iteration: int = Query(default=ITERATION, description="Iteration to read (must have been analyzed already)"),
    labeler: str = Query(default="judge", pattern="^(judge|llm_judge)$",
                          description="'judge' reads failure_labels_{iteration}.jsonl (rule-based); "
                                      "'llm_judge' reads failure_labels_llm_{iteration}.jsonl"),
    skills_overlap_threshold: float = Query(
        default=SKILLS_OVERLAP_THRESHOLD,
        description="Only used to compute the low_skills_overlap rate below — the stored file has the raw float, not a flag.",
    ),
) -> dict:
    suffix = "" if labeler == "judge" else "_llm"
    path = DATA_DIR / f"failure_labels{suffix}_{iteration}.jsonl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{path.name} not found. Run `python analysis.py --iteration {iteration}"
                   f"{' --llm-judge' if labeler == 'llm_judge' else ''}` first.",
        )

    labels = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    count = len(labels)

    def rate(flag: str) -> Optional[float]:
        if count == 0:
            return None
        if flag == "low_skills_overlap":
            return sum(1 for r in labels if r["skills_overlap"] < skills_overlap_threshold) / count
        if flag == "level_mismatch":
            return sum(1 for r in labels if r["level_mismatch"] > 1) / count
        return sum(r[flag] for r in labels) / count

    failure_rate_fields = [
        "low_skills_overlap", "experience_mismatch", "level_mismatch",
        "missing_skills", "hallucinate_skills", "awkward_language", "job_resume_match",
    ]
    failure_rates_out = {flag: rate(flag) for flag in failure_rate_fields}

    return {
        "iteration": iteration,
        "labeler": labeler,
        "count": count,
        "failure_rates": failure_rates_out,
        "labels": labels,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
