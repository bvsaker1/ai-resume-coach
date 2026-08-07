#!/usr/bin/env python3
"""
Pydantic schemas for jobs, resumes, and resume-job pairs, plus the shared
normalize+validate+construct logic used identically by generation
(dataset_generator.py, resume_generator.py) and correction (correction.py).

This module owns the model definitions (rather than just importing them) so
that both dataset_generator.py and resume_generator.py can depend on it
without recreating the existing dataset_generator <-> resume_generator
circular-import relationship — validation.py depends on neither of them.
"""

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

EXPERIENCE_LEVELS: list[str] = ["Entry", "Mid", "Senior", "Lead/Principal"]


# ── Pydantic Models — Job Posting ─────────────────────────────────────────────

class CompanyInfo(BaseModel):
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    size: str = Field(description="Employee count range, e.g. '1-50', '51-200', '201-1000', '1000+'")
    location: str = Field(description="City, State or City, Country")


class JobRequirements(BaseModel):
    required_skills: list[str] = Field(min_length=3, description="4-7 must-have skills")
    preferred_skills: list[str] = Field(description="2-5 nice-to-have skills")
    education: str = Field(description="Required education level and field, e.g. \"Bachelor's in Computer Science or equivalent\"")
    experience_years: int = Field(ge=0, le=30, description="Minimum years of experience required")
    experience_level: str = Field(description="Exactly one of: Entry, Mid, Senior, Lead/Principal")


class JobContent(BaseModel):
    """LLM-generated portion of a job posting (no system-assigned metadata)."""
    job_title: str
    company: CompanyInfo
    job_description: str = Field(description="2-3 paragraph description of the role, team, and expectations")
    requirements: JobRequirements
    is_niche_role: bool = Field(
        description="True if this is a highly specialized role that only a small subset of qualified professionals would meet"
    )


class JobPosting(BaseModel):
    """Full job posting including system metadata."""
    trace_id: str
    job_title: str
    company: CompanyInfo
    job_description: str
    requirements: JobRequirements
    is_niche_role: bool
    generated_at: str  # ISO 8601 datetime


class JobBatch(BaseModel):
    """Batch of job postings returned by LLM."""
    jobs: list[JobContent] = Field(description="List of job postings generated")


class RawJobBatch(BaseModel):
    """Loosely-typed job batch so per-item validation can be handled manually."""
    jobs: list[dict] = Field(description="List of raw job records generated")


def normalize_job_content(raw_job: dict) -> dict:
    """Accept both the current nested schema and older flat job shapes from the model."""
    normalized = dict(raw_job)

    company_name = normalized.pop("company_name", None)
    company_size = normalized.pop("company_size", None)
    location = normalized.pop("location", None)
    experience_level = normalized.pop("experience_level", None)
    experience_years = normalized.pop("experience_years", None)
    required_skills = normalized.pop("required_skills", None)
    preferred_skills = normalized.pop("preferred_skills", None)
    education = normalized.pop("education", None)

    if "company" not in normalized:
        normalized["company"] = {
            "name": company_name or "Unknown Company",
            "industry": normalized.get("industry") or "Unknown",
            "size": company_size or "Unknown",
            "location": location or "Unknown",
        }

    if "requirements" not in normalized:
        normalized["requirements"] = {
            "required_skills": required_skills or [],
            "preferred_skills": preferred_skills or [],
            "education": education or "Bachelor's degree or equivalent",
            "experience_years": experience_years if isinstance(experience_years, int) else infer_experience_years(experience_level),
            "experience_level": normalize_experience_level(experience_level),
        }

    if isinstance(normalized.get("requirements"), dict):
        requirements = dict(normalized["requirements"])
        if "is_niche_role" in requirements and "is_niche_role" not in normalized:
            normalized["is_niche_role"] = requirements.pop("is_niche_role")

        years_value = requirements.get("experience_years")
        if not isinstance(years_value, int):
            requirements["experience_years"] = parse_experience_years(years_value, requirements.get("experience_level"))

        requirements["experience_level"] = normalize_experience_level(requirements.get("experience_level"))
        normalized["requirements"] = requirements

    if "is_niche_role" not in normalized:
        normalized["is_niche_role"] = bool(raw_job.get("is_niche_role", False))

    if not normalized.get("job_title"):
        inferred_title = infer_job_title(raw_job)
        normalized["job_title"] = inferred_title or default_job_title(normalized)

    return normalized


def normalize_experience_level(value: Optional[str]) -> str:
    """Map verbose experience labels to the constrained enum used by the schema."""
    if not value:
        return "Mid"

    normalized = value.strip()
    lower = normalized.lower()
    if lower.startswith("entry"):
        return "Entry"
    if lower.startswith("mid"):
        return "Mid"
    if lower.startswith("senior"):
        return "Senior"
    if lower.startswith("lead") or lower.startswith("principal"):
        return "Lead/Principal"
    return normalized if normalized in EXPERIENCE_LEVELS else "Mid"


def infer_experience_years(experience_level: Optional[str]) -> int:
    """Provide a reasonable default when the model omits years but includes a level label."""
    level = normalize_experience_level(experience_level)
    defaults = {
        "Entry": 1,
        "Mid": 3,
        "Senior": 6,
        "Lead/Principal": 9,
    }
    return defaults.get(level, 3)


def parse_experience_years(value: object, experience_level: Optional[str]) -> int:
    """Normalize free-form experience year labels like '0-2' to an integer minimum."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return infer_experience_years(experience_level)


def infer_job_title(raw_job: dict) -> Optional[str]:
    """Recover a plausible title when the model omits job_title but the description contains it."""
    direct_title = raw_job.get("title") or raw_job.get("role") or raw_job.get("position")
    if isinstance(direct_title, str) and direct_title.strip():
        return direct_title.strip()

    description = raw_job.get("job_description")
    if not isinstance(description, str):
        return None

    patterns = [
        r"As a[n]? ([^,\n]+),",
        r"seeking a[n]? ([^,\n]+?) to join",
        r"([A-Za-z-]+ software engineer) to join",
        r"([A-Za-z-]+ developer) to join",
    ]
    for pattern in patterns:
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip(".")
            if candidate and len(candidate) <= 80 and "you'll" not in candidate.lower():
                return candidate

    return None


def default_job_title(raw_job: dict) -> str:
    """Provide a generic but usable title when the model omits one entirely."""
    requirements = raw_job.get("requirements") if isinstance(raw_job.get("requirements"), dict) else {}
    experience_level = normalize_experience_level(requirements.get("experience_level") or raw_job.get("experience_level"))
    required_skills = requirements.get("required_skills") or raw_job.get("required_skills") or []
    skills_blob = " ".join(required_skills).lower() if isinstance(required_skills, list) else str(required_skills).lower()

    if "html" in skills_blob or "css" in skills_blob or "javascript" in skills_blob:
        base_title = "Frontend Developer"
    elif "data" in skills_blob or "sql" in skills_blob:
        base_title = "Software Developer"
    else:
        base_title = "Software Engineer"

    return f"{experience_level} {base_title}"


# ── Pydantic Models — Resume ──────────────────────────────────────────────────

def _validate_iso_date_str(v: str) -> str:
    """Accept YYYY, YYYY-MM, or YYYY-MM-DD."""
    if not re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", v):
        raise ValueError(f"Date must be ISO format (YYYY, YYYY-MM, or YYYY-MM-DD), got: {v!r}")
    return v


class ContactInfo(BaseModel):
    name: str
    email: str = Field(description="Valid email address, e.g. jane.doe@example.com")
    phone: str = Field(min_length=10, description="Phone number, at least 10 characters including country code if applicable")
    location: str = Field(description="City, State or City, Country")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL (optional)")
    portfolio: Optional[str] = Field(None, description="Portfolio or GitHub URL (optional)")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email format: {v!r}")
        return v


class EducationEntry(BaseModel):
    degree: str = Field(description="Degree and major, e.g. \"Bachelor of Science in Computer Science\"")
    institution: str = Field(description="University or college name")
    graduation_date: str = Field(description="ISO format: YYYY or YYYY-MM or YYYY-MM-DD")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA on 4.0 scale (optional)")
    coursework: Optional[list[str]] = Field(None, description="Relevant courses (optional)")

    @field_validator("graduation_date")
    @classmethod
    def validate_graduation_date(cls, v: str) -> str:
        return _validate_iso_date_str(v)


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str = Field(description="ISO format: YYYY-MM or YYYY-MM-DD")
    end_date: Optional[str] = Field(
        None,
        description="ISO format (YYYY-MM or YYYY-MM-DD) or the literal string 'Present'"
    )
    responsibilities: list[str] = Field(min_length=2, description="2-5 key responsibilities")
    achievements: list[str] = Field(description="1-4 quantified or notable achievements")

    @field_validator("start_date")
    @classmethod
    def validate_start(cls, v: str) -> str:
        return _validate_iso_date_str(v)

    @field_validator("end_date")
    @classmethod
    def validate_end(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "Present":
            return v
        return _validate_iso_date_str(v)

    @model_validator(mode="after")
    def end_after_start(self) -> "WorkExperience":
        if self.end_date and self.end_date != "Present":
            # Compare YYYY-MM prefix for robustness
            start_prefix = self.start_date[:7]
            end_prefix = self.end_date[:7]
            if end_prefix < start_prefix:
                raise ValueError(
                    f"end_date {self.end_date!r} must be after start_date {self.start_date!r}"
                )
        return self


class ProficiencyLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


class SkillEntry(BaseModel):
    name: str
    proficiency_level: ProficiencyLevel
    years: Optional[float] = Field(None, ge=0, description="Years of experience with this skill (optional)")


class ResumeContent(BaseModel):
    """LLM-generated portion of a resume (no system-assigned metadata)."""
    contact: ContactInfo
    education: list[EducationEntry] = Field(min_length=1)
    experience: list[WorkExperience] = Field(min_length=1)
    skills: list[SkillEntry] = Field(min_length=3)
    summary: Optional[str] = Field(None, description="2-4 sentence professional summary")


class Resume(BaseModel):
    """Full resume including system metadata."""
    trace_id: str
    contact: ContactInfo
    education: list[EducationEntry]
    experience: list[WorkExperience]
    skills: list[SkillEntry]
    summary: Optional[str]
    generated_at: str
    prompt_template: str
    fit_level: str
    writing_style: str


class ResumeBatch(BaseModel):
    """Batch of resumes returned by LLM."""
    resumes: list[ResumeContent] = Field(description="List of resumes generated")


class RawResumeBatch(BaseModel):
    """Loosely-typed resume batch so per-item validation can be handled manually."""
    resumes: list[dict] = Field(description="List of raw resume records generated")


# ── Pair Model ────────────────────────────────────────────────────────────────

class ResumePair(BaseModel):
    trace_id: str
    job_trace_id: str
    resume_trace_id: str
    fit_level: str
    template_style: str
    judge_score: Optional[float] = None  # populated by judge step (stubbed)
    generated_at: str


# ── Shared validate-and-construct functions ───────────────────────────────────
#
# Used identically by generation (dataset_generator.generate_jobs_batch,
# resume_generator._generate_resumes_batch) and by correction.py's
# re-validation of LLM-corrected output. Never raise — a ValidationError is
# always caught and returned as error_details instead.

def validate_job(raw_job: dict, trace_id: str, generated_at: str) -> tuple[Optional[JobPosting], Optional[dict]]:
    """Normalize + validate a raw job dict into a JobPosting.

    Returns (JobPosting, None) on success, (None, error_details) on failure.
    """
    try:
        normalized_job = normalize_job_content(raw_job)
        job_content = JobContent.model_validate(normalized_job)
    except ValidationError as exc:
        return None, {
            "record_trace_id": trace_id,
            "stage": "job_validation",
            "error_type": "pydantic_validation_error",
            "validation_errors": exc.errors(include_context=False),
        }

    job_posting = JobPosting(
        trace_id=trace_id,
        job_title=job_content.job_title,
        company=job_content.company,
        job_description=job_content.job_description,
        requirements=job_content.requirements,
        is_niche_role=job_content.is_niche_role,
        generated_at=generated_at,
    )
    return job_posting, None


def validate_resume(
    raw_resume: dict, trace_id: str, generated_at: str, fit_level: str, template: dict,
) -> tuple[Optional[Resume], Optional[dict]]:
    """Validate a raw resume dict into a Resume. Mirrors validate_job's contract.

    Returns (Resume, None) on success, (None, error_details) on failure.
    """
    try:
        resume_content = ResumeContent.model_validate(raw_resume)
    except ValidationError as exc:
        return None, {
            "record_trace_id": trace_id,
            "stage": "resume_validation",
            "error_type": "pydantic_validation_error",
            "validation_errors": exc.errors(include_context=False),
        }

    resume = Resume(
        trace_id=trace_id,
        contact=resume_content.contact,
        education=resume_content.education,
        experience=resume_content.experience,
        skills=resume_content.skills,
        summary=resume_content.summary,
        generated_at=generated_at,
        prompt_template=template["name"],
        fit_level=fit_level,
        writing_style=template["writing_style"],
    )
    return resume, None
