#!/usr/bin/env python3
"""
Rule-based (non-LLM) judge for generated job/resume pairs, with an optional
LLM-judge mode.

Loops over every pair for an iteration, scores it against 6 failure
criteria (skills overlap, experience mismatch, seniority mismatch, missing
core skills, hallucinated skills, awkward language), writes one label
record per pair to data/failure_labels_{iteration}.jsonl, and renders 6
diagnostic charts (matplotlib, .png) to data/visualizations/.

With --llm-judge, additionally runs the SAME 6 rules through JUDGE_MODEL (an
LLM call per pair, via instructor) instead of Python, and writes a second,
identically-shaped file: data/failure_labels_llm_{iteration}.jsonl
(labeler: "llm_judge"). Off by default — extra LLM cost, opt-in.

Standalone, manual step — never invoked by dataset_generator.py,
correction.py, or pipeline.py.

Usage:
    python analysis.py --iteration 1
    python analysis.py --iteration 1 --skills-overlap-threshold 0.4
    python analysis.py --iteration 1 --skip-charts
    python analysis.py --iteration 1 --llm-judge
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import instructor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from validation import JobPosting, Resume, ResumePair, ProficiencyLevel
from dataset_generator import (
    DATA_DIR, ITERATION, TEMPLATE_NAMES, FIT_LEVELS,
    JUDGE_MODEL, JUDGE_MAX_COMPLETION_TOKENS, LLM_REQUEST_TIMEOUT_SECONDS, make_client, _rate_limiter,
)
from llm_errors import is_fatal_llm_error
from correction import load_jsonl_pairs

# ── Tunable thresholds ─────────────────────────────────────────────────────────

SKILLS_OVERLAP_THRESHOLD = 0.5  # below this, "low skills overlap" fails
EXPERIENCE_GAP_YEARS = 1.0
JOB_LEVELS = {"Entry": 0, "Mid": 1, "Senior": 2, "Lead/Principal": 3}
RESUME_LEVEL_NAMES = ["Entry", "Mid", "Senior", "Lead/Principal", "Exec"]

HALLUCINATION_PHRASES = [
    "expert in all", "certified in everything", "master of all", "expert in everything",
]

JARGON_PHRASES = [
    "synergy", "synergies", "think outside the box", "thinking outside the box",
    "move the needle", "circle back", "low-hanging fruit", "low hanging fruit",
    "paradigm shift", "deep dive", "touch base", "value add", "value-add",
    "best-in-class", "best in class", "game changer", "game-changing", "gamechanging",
    "disrupt", "disruptive",
]

STOPWORDS = {
    "the", "and", "a", "an", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "this", "that", "as", "at", "by", "from", "or", "be", "has",
    "have", "had", "our", "their", "its", "it's", "we", "our", "you", "your",
}

FAILURE_FLAGS = [
    "low_skills_overlap", "experience_mismatch", "level_mismatch_flag",
    "missing_skills", "hallucinate_skills", "awkward_language",
]
FAILURE_LABELS_DISPLAY = [
    "Low Skills Overlap", "Experience Mismatch", "Level Mismatch",
    "Missing Core Skills", "Hallucinated Skills", "Awkward Language",
]


# ── Skills normalization + overlap ────────────────────────────────────────────

def normalize_skill(name: str) -> str:
    """lowercase -> strip parenthetical content -> strip version-like tokens
    -> drop punctuation -> collapse whitespace -> strip trailing plural 's'."""
    s = name.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\bv?\d+(\.\d+)*\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s


def skills_overlap(job: JobPosting, resume: Resume) -> float:
    """|normalized(required_skills) ∩ normalized(resume.skills)| / |normalized(required_skills)|.
    Deliberately excludes preferred_skills: those are optional nice-to-haves,
    and including them (avg ~10.7 combined job skills vs. resumes averaging
    only ~4.6 skills listed) made low_skills_overlap fire on almost every
    pair regardless of true fit, washing out any signal between fit levels."""
    job_skills = {normalize_skill(s) for s in job.requirements.required_skills}
    resume_skills = {normalize_skill(s.name) for s in resume.skills}
    if not job_skills:
        return 0.0
    return len(job_skills & resume_skills) / len(job_skills)


def missing_core_skills(job: JobPosting, resume: Resume) -> int:
    """1 if any of the top-3 required skills is absent from the resume, else 0."""
    top3 = {normalize_skill(s) for s in job.requirements.required_skills[:3]}
    resume_skills = {normalize_skill(s.name) for s in resume.skills}
    return int(bool(top3 - resume_skills))


# ── Experience / timeline helpers ─────────────────────────────────────────────

def parse_date(date_str: Optional[str], reference_dt: datetime) -> datetime:
    """Accepts YYYY, YYYY-MM, YYYY-MM-DD, 'Present', or None (treated as ongoing)."""
    if not date_str or date_str == "Present":
        return reference_dt
    parts = date_str.split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    return datetime(year, month, day, tzinfo=reference_dt.tzinfo)


def _experience_intervals(resume: Resume) -> list[tuple[datetime, datetime]]:
    reference_dt = datetime.fromisoformat(resume.generated_at)
    intervals = [
        (parse_date(e.start_date, reference_dt), parse_date(e.end_date, reference_dt))
        for e in resume.experience
    ]
    return sorted(intervals, key=lambda t: t[0])


def total_experience_years(resume: Resume) -> float:
    """Sum of (end - start) in years across every experience entry. Overlaps
    simply add (not deduplicated) — a simple additive total, as specified."""
    total = 0.0
    for start, end in _experience_intervals(resume):
        total += max(0.0, (end - start).days / 365.25)
    return total


def has_experience_gap(resume: Resume, gap_years: float = EXPERIENCE_GAP_YEARS) -> bool:
    """True if any two (sorted) experience entries have a gap >= gap_years."""
    intervals = _experience_intervals(resume)
    for (_, end), (next_start, _) in zip(intervals, intervals[1:]):
        gap = (next_start - end).days / 365.25
        if gap >= gap_years:
            return True
    return False


def has_overlapping_jobs(resume: Resume) -> bool:
    """True if any two experience entries have genuinely overlapping date ranges."""
    max_end = None
    for start, end in _experience_intervals(resume):
        if max_end is not None and start < max_end:
            return True
        max_end = end if max_end is None else max(max_end, end)
    return False


def infer_resume_level(total_years: float) -> int:
    """0=Entry, 1=Mid, 2=Senior, 3=Lead/Principal, 4=Exec — bands mirror
    build_jobs_batch_prompt's own experience-year ranges, with an Exec band
    added since no job in this dataset ever requires it but resumes can
    still claim 15+ years."""
    if total_years < 2:
        return 0
    if total_years < 5:
        return 1
    if total_years < 10:
        return 2
    if total_years < 15:
        return 3
    return 4


# ── Hallucinated skills ────────────────────────────────────────────────────────

def _resume_text(resume: Resume) -> str:
    parts = [resume.summary or ""]
    for e in resume.experience:
        parts.extend(e.responsibilities)
        parts.extend(e.achievements)
    return " ".join(parts)


def hallucinated_skills(resume: Resume, total_years: float) -> int:
    expert_count = sum(1 for s in resume.skills if s.proficiency_level == ProficiencyLevel.EXPERT)
    total_skills = len(resume.skills)

    entry_level_overclaim = total_years < 2 and expert_count >= 10
    mostly_expert_bloat = total_skills >= 30 and expert_count > total_skills / 2

    text = _resume_text(resume).lower()
    phrase_hit = any(phrase in text for phrase in HALLUCINATION_PHRASES)

    concurrent_present = sum(1 for e in resume.experience if e.end_date in (None, "Present"))
    timeline_inconsistent = has_overlapping_jobs(resume) or concurrent_present > 1

    return int(entry_level_overclaim or mostly_expert_bloat or phrase_hit or timeline_inconsistent)


# ── Awkward language ───────────────────────────────────────────────────────────

def _jargon_count(text: str) -> int:
    lowered = text.lower()
    return sum(lowered.count(phrase) for phrase in JARGON_PHRASES)


def _has_repetitive_word_pattern(text: str, window: int = 25, min_count: int = 3) -> bool:
    words = [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if len(w) > 2 and w not in STOPWORDS]
    for i in range(len(words)):
        counts = Counter(words[i:i + window])
        if counts and max(counts.values()) >= min_count:
            return True
    return False


def awkward_language(resume: Resume) -> int:
    text = _resume_text(resume)
    jargon_overload = _jargon_count(text) > 5
    repetitive = _has_repetitive_word_pattern(text)
    return int(jargon_overload or repetitive)


# ── Per-pair analysis ──────────────────────────────────────────────────────────

def score_pair(job: JobPosting, resume: Resume, threshold: float = SKILLS_OVERLAP_THRESHOLD) -> dict:
    """Score one job/resume pair against the 6 criteria. Pure function of
    (job, resume) — no ResumePair needed, so this is what api.py calls
    directly for ad-hoc submissions as well as what analyze_pair wraps for
    the bulk CLI path. Returns just the 7 computed fields (no trace_id/
    industry/labeler — those are identity metadata the caller attaches)."""
    total_years = total_experience_years(resume)
    exp_insufficient = total_years < job.requirements.experience_years / 2
    exp_gap = has_experience_gap(resume)
    experience_mismatch = int(exp_insufficient or exp_gap)

    resume_level = infer_resume_level(total_years)
    job_level = JOB_LEVELS.get(job.requirements.experience_level, 1)
    level_diff = abs(job_level - resume_level)

    overlap = skills_overlap(job, resume)
    missing = missing_core_skills(job, resume)
    halluc = hallucinated_skills(resume, total_years)
    awkward = awkward_language(resume)

    low_overlap = overlap < threshold
    level_fail = level_diff > 1

    job_resume_match = int(not (low_overlap or experience_mismatch or level_fail or missing or halluc or awkward))

    return {
        "skills_overlap": round(overlap, 4),
        "experience_mismatch": experience_mismatch,
        "level_mismatch": level_diff,
        "missing_skills": missing,
        "hallucinate_skills": halluc,
        "awkward_language": awkward,
        "job_resume_match": job_resume_match,
    }


def analyze_pair(
    job: JobPosting, resume: Resume, pair: ResumePair, threshold: float = SKILLS_OVERLAP_THRESHOLD,
) -> tuple[dict, dict]:
    """Returns (label_record, chart_context) — label_record is exactly the
    failure_labels_{iteration}.jsonl schema; chart_context carries extra
    fields (fit_level, template_style, industry grouping dims) used only
    for the in-memory charting DataFrame, never written to the label file."""
    scored = score_pair(job, resume, threshold)

    label_record = {
        "trace_id": pair.trace_id,
        "industry": job.company.industry,
        "labeler": "judge",
        **scored,
    }

    total_years = total_experience_years(resume)
    resume_level = infer_resume_level(total_years)
    job_level = JOB_LEVELS.get(job.requirements.experience_level, 1)
    chart_context = {
        "fit_level": pair.fit_level,
        "template_style": pair.template_style,
        "is_niche_role": job.is_niche_role,
        "job_level": job_level,
        "resume_level": resume_level,
        "low_skills_overlap": int(scored["skills_overlap"] < threshold),
        "level_mismatch_flag": int(scored["level_mismatch"] > 1),
    }
    return label_record, chart_context


# ── LLM judge (optional) ──────────────────────────────────────────────────────

class LLMJudgment(BaseModel):
    """Structured output for the LLM-based judge — same 7 computed fields as
    analyze_pair's label_record (trace_id/industry/labeler are added
    programmatically afterward, not asked of the LLM)."""
    skills_overlap: float = Field(ge=0.0, le=1.0, description="Fraction of the job's required_skills also present on the resume")
    experience_mismatch: int = Field(ge=0, le=1, description="1 if the experience-mismatch rule fails, else 0")
    level_mismatch: int = Field(ge=0, le=4, description="Absolute seniority level difference between job and resume")
    missing_skills: int = Field(ge=0, le=1, description="1 if any of the job's top-3 required skills is absent from the resume")
    hallucinate_skills: int = Field(ge=0, le=1, description="1 if any hallucination sub-check trips")
    awkward_language: int = Field(ge=0, le=1, description="1 if any awkward-language sub-check trips")
    job_resume_match: int = Field(ge=0, le=1, description="1 only if all 6 criteria pass, else 0")


def build_judge_prompt(job: JobPosting, resume: Resume, threshold: float) -> str:
    """Encodes the exact same 6 rules analyze_pair() computes in Python as
    explicit instructions, so the LLM judge is evaluated against the same
    logic rather than its own free-form notion of 'good fit'."""
    job_json = job.model_dump_json(indent=2)
    resume_json = resume.model_dump_json(indent=2)
    schema_json = json.dumps(LLMJudgment.model_json_schema(), indent=2)
    jargon_list = ", ".join(f'"{p}"' for p in JARGON_PHRASES)
    hallucination_list = ", ".join(f'"{p}"' for p in HALLUCINATION_PHRASES)

    return f"""You are a rule-based resume/job matching judge. Apply EXACTLY the 6 rules
below to the job posting and resume JSON given — do not substitute your own subjective
notion of "good fit"; only these explicit rules. Compute each field precisely as instructed.

═══ JOB POSTING ═══
{job_json}

═══ RESUME ═══
{resume_json}

═══ RULES ═══

1. skills_overlap (float 0-1): Normalize every skill name (lowercase, strip version
   numbers and punctuation, ignore a trailing plural "s"). Compute
   |normalized(job.requirements.required_skills) ∩ normalized(resume.skills[].name)|
   / |normalized(job.requirements.required_skills)|. This criterion FAILS if the
   result is below {threshold}.

2. experience_mismatch (0 or 1): Sum the resume's work experience durations in years
   across every entry in resume.experience (treat "Present" or a missing end_date as
   ongoing until today). Set experience_mismatch = 1 if EITHER:
   (a) that total is less than job.requirements.experience_years / 2, OR
   (b) sorting resume.experience by start_date, any two consecutive entries have a
       gap of 1 year or more between one entry's end and the next's start.
   Otherwise 0.

3. level_mismatch (integer 0-4): Infer the resume's seniority level from its total
   experience years: <2 years = Entry(0), 2-5 = Mid(1), 5-10 = Senior(2),
   10-15 = Lead/Principal(3), 15+ = Exec(4). Map the job's required
   experience_level the same way: Entry=0, Mid=1, Senior=2, Lead/Principal=3.
   level_mismatch = the absolute difference between the two levels. This criterion
   FAILS the overall match when level_mismatch > 1.

4. missing_skills (0 or 1): Set to 1 if any of the FIRST THREE entries in
   job.requirements.required_skills (in order) is absent from the resume's skills
   (case/format-insensitive), else 0.

5. hallucinate_skills (0 or 1): Set to 1 if ANY of the following is true, else 0:
   (a) the resume's total experience is under 2 years AND 10 or more of its skills
       have proficiency_level "Expert";
   (b) the resume lists 30 or more skills total AND more than half are "Expert";
   (c) the resume's summary or any responsibility/achievement text contains a phrase
       like: {hallucination_list};
   (d) inconsistent timeline: two experience entries have genuinely overlapping date
       ranges, OR more than one entry has end_date "Present" or null (i.e. more than
       one simultaneous "current" job).

6. awkward_language (0 or 1): Set to 1 if EITHER:
   (a) across the resume's summary and all experience responsibilities/achievements
       combined, phrases from this corporate-jargon list appear more than 5 times
       total: {jargon_list};
   (b) the same word (ignoring common stopwords) appears 3 or more times within any
       ~25-word span of that same combined text.

7. job_resume_match (0 or 1): 1 ONLY IF skills_overlap >= {threshold} AND
   experience_mismatch == 0 AND level_mismatch <= 1 AND missing_skills == 0 AND
   hallucinate_skills == 0 AND awkward_language == 0. Otherwise 0.

═══ REQUIRED OUTPUT SCHEMA ═══
{schema_json}"""


def run_llm_judgment(
    client: instructor.Instructor, model: str, job: JobPosting, resume: Resume,
    threshold: float, max_completion_tokens: int,
) -> LLMJudgment:
    """Pure LLM call: builds the prompt (same 6 rules as score_pair, in prose)
    and returns the raw structured judgment. No ResumePair needed — this is
    what api.py calls directly for ad-hoc submissions. Raises any
    is_fatal_llm_error() exception; caller decides how to handle it.

    max_retries=2 (not the usual 3) and an explicit per-call timeout: this is
    the one call site a synchronous HTTP request waits on directly (POST
    /review-resume), so worst-case latency is bounded at
    2 * LLM_REQUEST_TIMEOUT_SECONDS rather than left open-ended."""
    prompt = build_judge_prompt(job, resume, threshold)

    estimated_tokens = len(prompt) // 4 + max_completion_tokens
    _rate_limiter.wait_if_needed(estimated_tokens)

    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_model=LLMJudgment,
        max_retries=2,
        max_completion_tokens=max_completion_tokens,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )


def llm_judge_pair(
    client: instructor.Instructor, model: str, job: JobPosting, resume: Resume, pair: ResumePair,
    threshold: float, max_completion_tokens: int,
) -> dict:
    """Same output schema as analyze_pair's label_record, but every judged
    field is computed by the LLM instead of Python. Thin wrapper around
    run_llm_judgment that attaches trace_id/industry/labeler for the bulk
    CLI path. Raises any is_fatal_llm_error() exception — caller decides
    whether to abort or skip this pair."""
    judgment = run_llm_judgment(client, model, job, resume, threshold, max_completion_tokens)

    return {
        "trace_id": pair.trace_id,
        "industry": job.company.industry,
        "labeler": "llm_judge",
        "skills_overlap": round(judgment.skills_overlap, 4),
        "experience_mismatch": judgment.experience_mismatch,
        "level_mismatch": judgment.level_mismatch,
        "missing_skills": judgment.missing_skills,
        "hallucinate_skills": judgment.hallucinate_skills,
        "awkward_language": judgment.awkward_language,
        "job_resume_match": judgment.job_resume_match,
    }


# ── Data loading ────────────────────────────────────────────────────────────────

def _load_jsonl_models(path: Path, model) -> dict[str, object]:
    items: dict[str, object] = {}
    if not path.exists():
        return items
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = model.model_validate_json(line)
        items[obj.trace_id] = obj
    return items


def load_jobs(iteration: int) -> dict[str, JobPosting]:
    jobs = _load_jsonl_models(DATA_DIR / f"jobs_valid_{iteration}.jsonl", JobPosting)
    jobs.update(_load_jsonl_models(DATA_DIR / f"jobs_corrected_{iteration}.jsonl", JobPosting))
    return jobs


def load_resumes(iteration: int) -> dict[str, Resume]:
    resumes = _load_jsonl_models(DATA_DIR / f"resumes_valid_{iteration}.jsonl", Resume)
    resumes.update(_load_jsonl_models(DATA_DIR / f"resumes_corrected_{iteration}.jsonl", Resume))
    return resumes


def load_pairs(iteration: int) -> list[ResumePair]:
    pairs: list[ResumePair] = []
    for path in (DATA_DIR / f"pairs_{iteration}.jsonl", DATA_DIR / f"pairs_corrected_{iteration}.jsonl"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            pairs.append(ResumePair.model_validate_json(line))
    return pairs


# ── Charts ──────────────────────────────────────────────────────────────────────

def _grouped_failure_rate_chart(
    df: pd.DataFrame, group_col: str, group_order: list[str], title: str, out_path: Path,
) -> None:
    rates = df.groupby(group_col)[FAILURE_FLAGS].mean().reindex(group_order).fillna(0.0)
    x = np.arange(len(group_order))
    n = len(FAILURE_FLAGS)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (flag, display) in enumerate(zip(FAILURE_FLAGS, FAILURE_LABELS_DISPLAY)):
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, rates[flag].values, width, label=display)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=20, ha="right")
    ax.set_ylabel("Failure rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _correlation_heatmap(df: pd.DataFrame, iteration: int, out_path: Path) -> None:
    corr = df[FAILURE_FLAGS].corr().fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FAILURE_FLAGS)))
    ax.set_xticklabels(FAILURE_LABELS_DISPLAY, rotation=45, ha="right")
    ax.set_yticks(range(len(FAILURE_FLAGS)))
    ax.set_yticklabels(FAILURE_LABELS_DISPLAY)
    for i in range(len(FAILURE_FLAGS)):
        for j in range(len(FAILURE_FLAGS)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title(f"Failure Mode Correlation Matrix (iteration {iteration})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _schema_validation_heatmap(industry_counts: dict[str, dict[str, int]], iteration: int, out_path: Path) -> None:
    industries = sorted(industry_counts)
    if not industries:
        print("  Skipping schema validation heatmap: no industry data found.")
        return

    def rate(ic: dict[str, int], valid_key: str, invalid_key: str) -> float:
        denom = ic[valid_key] + ic[invalid_key]
        return ic[invalid_key] / denom if denom else 0.0

    jobs_rate = [rate(industry_counts[i], "jobs_valid", "jobs_invalid") for i in industries]
    resumes_rate = [rate(industry_counts[i], "resumes_valid", "resumes_invalid") for i in industries]
    matrix = np.array([jobs_rate, resumes_rate]).T

    fig, ax = plt.subplots(figsize=(6, max(4, len(industries) * 0.6)))
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Jobs Invalid Rate", "Resumes Invalid Rate"])
    ax.set_yticks(range(len(industries)))
    ax.set_yticklabels(industries)
    for i in range(len(industries)):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Invalid rate")
    ax.set_title(f"Schema Validation Heatmap (iteration {iteration})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _hallucination_by_seniority_chart(df: pd.DataFrame, iteration: int, out_path: Path) -> None:
    df = df.copy()
    df["resume_level_label"] = df["resume_level"].map(dict(enumerate(RESUME_LEVEL_NAMES)))
    counts = df.groupby(["resume_level_label", "hallucinate_skills"]).size().unstack(fill_value=0)
    counts = counts.reindex(RESUME_LEVEL_NAMES).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(counts))
    for flag_val, color, label in ((0, "tab:blue", "No Hallucination Flag"), (1, "tab:red", "Hallucination Flagged")):
        vals = counts[flag_val].values if flag_val in counts.columns else np.zeros(len(counts))
        ax.bar(counts.index.astype(str), vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_ylabel("Number of resumes")
    ax.set_title(f"Hallucination Flags by Resume Seniority (iteration {iteration})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_charts(
    df: pd.DataFrame, iteration: int, charts_dir: Path, industry_counts: dict[str, dict[str, int]],
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    _correlation_heatmap(df, iteration, charts_dir / f"failure_correlation_heatmap_{iteration}.png")

    _grouped_failure_rate_chart(
        df, "fit_level", FIT_LEVELS,
        f"Failure Rates by Fit Level (iteration {iteration})",
        charts_dir / f"failure_rates_by_fit_level_{iteration}.png",
    )

    _grouped_failure_rate_chart(
        df, "template_style", TEMPLATE_NAMES,
        f"Failure Rates by Template (iteration {iteration})",
        charts_dir / f"failure_rates_by_template_{iteration}.png",
    )

    df_niche = df.copy()
    df_niche["niche_label"] = df_niche["is_niche_role"].map({True: "Niche", False: "Standard"})
    _grouped_failure_rate_chart(
        df_niche, "niche_label", ["Niche", "Standard"],
        f"Niche vs. Standard Roles — Failure Rates (iteration {iteration})",
        charts_dir / f"niche_vs_standard_{iteration}.png",
    )

    _schema_validation_heatmap(industry_counts, iteration, charts_dir / f"schema_validation_heatmap_{iteration}.png")

    _hallucination_by_seniority_chart(df, iteration, charts_dir / f"hallucination_by_seniority_{iteration}.png")


def compute_industry_counts(
    jobs: dict[str, JobPosting], resumes: dict[str, Resume], pairs: list[ResumePair], iteration: int,
) -> dict[str, dict[str, int]]:
    """Per-industry valid/invalid counts for jobs and resumes, for the schema
    validation heatmap. 'Valid' here means currently valid (post-correction,
    if correction has run) — a job counted invalid may also be counted valid
    if it was subsequently corrected; this is a diagnostic simplification."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {
        "jobs_valid": 0, "jobs_invalid": 0, "resumes_valid": 0, "resumes_invalid": 0,
    })

    for job in jobs.values():
        counts[job.company.industry]["jobs_valid"] += 1

    jobs_invalid_records = load_jsonl_pairs(DATA_DIR / f"jobs_invalid_{iteration}.jsonl")
    for raw, _err in jobs_invalid_records:
        counts[raw.get("industry", "Unknown")]["jobs_invalid"] += 1

    resume_to_job = {p.resume_trace_id: p.job_trace_id for p in pairs}
    for resume_trace_id in resumes:
        job_trace_id = resume_to_job.get(resume_trace_id)
        job = jobs.get(job_trace_id) if job_trace_id else None
        industry = job.company.industry if job else "Unknown"
        counts[industry]["resumes_valid"] += 1

    resumes_invalid_records = load_jsonl_pairs(DATA_DIR / f"resumes_invalid_{iteration}.jsonl")
    for raw, _err in resumes_invalid_records:
        job_trace_id = raw.get("job_trace_id")
        job = jobs.get(job_trace_id) if job_trace_id else None
        industry = job.company.industry if job else "Unknown"
        counts[industry]["resumes_invalid"] += 1

    return dict(counts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based judge: score job/resume pairs against 6 match criteria and chart the results."
    )
    parser.add_argument("--iteration", type=int, default=ITERATION,
                         help=f"Iteration to analyze — must match an existing dataset_generator.py run (default: {ITERATION} from .env)")
    parser.add_argument("--skills-overlap-threshold", type=float, default=SKILLS_OVERLAP_THRESHOLD,
                         help=f"Below this, skills overlap counts as 'low' and fails the pair (default: {SKILLS_OVERLAP_THRESHOLD})")
    parser.add_argument("--skip-charts", action="store_true",
                         help="Only write failure_labels_{iteration}.jsonl; skip chart generation")
    parser.add_argument("--llm-judge", action="store_true",
                         help="Additionally run the same 6 rules through JUDGE_MODEL (one LLM call per pair) "
                              "and write failure_labels_llm_{iteration}.jsonl. Off by default — extra LLM cost.")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL,
                         help=f"Model to use for --llm-judge (default: {JUDGE_MODEL} from .env JUDGE_MODEL)")
    parser.add_argument("--judge-max-completion-tokens", type=int, default=JUDGE_MAX_COMPLETION_TOKENS,
                         help=f"Token budget per judge call (default: {JUDGE_MAX_COMPLETION_TOKENS})")
    args = parser.parse_args()

    jobs = load_jobs(args.iteration)
    resumes = load_resumes(args.iteration)
    pairs = load_pairs(args.iteration)

    if not pairs:
        print(f"No pairs found for iteration {args.iteration} "
              f"(checked pairs_{args.iteration}.jsonl / pairs_corrected_{args.iteration}.jsonl).")
        return

    labels: list[dict] = []
    rows: list[dict] = []
    skipped = 0
    for pair in pairs:
        job = jobs.get(pair.job_trace_id)
        resume = resumes.get(pair.resume_trace_id)
        if job is None or resume is None:
            skipped += 1
            continue
        label_record, chart_context = analyze_pair(job, resume, pair, threshold=args.skills_overlap_threshold)
        labels.append(label_record)
        rows.append({**label_record, **chart_context})

    if skipped:
        print(f"  WARNING: skipped {skipped} pair(s) with unresolved job/resume records.")

    labels_path = DATA_DIR / f"failure_labels_{args.iteration}.jsonl"
    with open(labels_path, "w") as f:
        for record in labels:
            f.write(json.dumps(record) + "\n")

    match_rate = sum(r["job_resume_match"] for r in labels) / len(labels) if labels else 0.0
    print(f"Wrote {len(labels)} failure labels to {labels_path}")
    print(f"Overall job_resume_match rate: {match_rate:.1%}")

    if args.llm_judge:
        print(f"\nRunning LLM judge ({args.judge_model}) over {len(pairs)} pair(s)...")
        client = make_client(args.judge_model)
        llm_labels: list[dict] = []
        llm_skipped = 0
        try:
            for pair in pairs:
                job = jobs.get(pair.job_trace_id)
                resume = resumes.get(pair.resume_trace_id)
                if job is None or resume is None:
                    continue
                try:
                    llm_labels.append(llm_judge_pair(
                        client, args.judge_model, job, resume, pair,
                        args.skills_overlap_threshold, args.judge_max_completion_tokens,
                    ))
                except Exception as exc:
                    if is_fatal_llm_error(exc):
                        raise
                    print(f"  WARNING: LLM judge failed for {pair.trace_id}, skipping: {exc}")
                    llm_skipped += 1
        except Exception as exc:
            print(f"\n❌ LLM judge run aborted due to a fatal LLM/API error: {exc}")
            sys.exit(1)

        if llm_skipped:
            print(f"  WARNING: skipped {llm_skipped} pair(s) after non-fatal LLM judge errors.")

        llm_labels_path = DATA_DIR / f"failure_labels_llm_{args.iteration}.jsonl"
        with open(llm_labels_path, "w") as f:
            for record in llm_labels:
                f.write(json.dumps(record) + "\n")

        llm_match_rate = sum(r["job_resume_match"] for r in llm_labels) / len(llm_labels) if llm_labels else 0.0
        print(f"Wrote {len(llm_labels)} LLM failure labels to {llm_labels_path}")
        print(f"LLM judge job_resume_match rate: {llm_match_rate:.1%}")

    if args.skip_charts:
        return

    df = pd.DataFrame(rows)
    industry_counts = compute_industry_counts(jobs, resumes, pairs, args.iteration)
    charts_dir = DATA_DIR / "visualizations"
    build_charts(df, args.iteration, charts_dir, industry_counts)
    print(f"Wrote 6 charts to {charts_dir}")


if __name__ == "__main__":
    main()
