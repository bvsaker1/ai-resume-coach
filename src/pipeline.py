#!/usr/bin/env python3
"""
Pipeline: run dataset_generator.py, then correction.py, for the same
iteration.

Each stage is invoked as a real subprocess (not an in-process function call)
so that dataset_generator.py's and correction.py's module-level state
(RateLimiter, open Logger file handles, argparse) never leak into each other
— this pipeline just chains the two CLIs the way you'd run them by hand,
guaranteeing they target the same --iteration.

Usage:
    python pipeline.py --num-jobs 50 --resumes-per-job 5
    python pipeline.py --max-industries 5 --iteration 2
    python pipeline.py --num-jobs 50 --skip-correction
    python pipeline.py --num-jobs 50 --generate-resumes-for-corrected-jobs
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dataset_generator import ITERATION, NUM_JOBS, NUM_RESUMES_PER_JOB, JOB_MAX_COMPLETION_TOKENS, RESUME_MAX_COMPLETION_TOKENS

SRC_DIR = Path(__file__).parent


def run_stage(label: str, args: list[str]) -> int:
    """Run a pipeline stage as a subprocess from src/, streaming its output live."""
    print(f"\n{'#' * 60}")
    print(f"# {label}")
    print(f"# {' '.join(args)}")
    print(f"{'#' * 60}\n")
    result = subprocess.run([sys.executable, *args], cwd=SRC_DIR)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dataset_generator.py then correction.py for the same iteration."
    )
    # Generation flags (passed through to dataset_generator.py)
    parser.add_argument("--num-jobs", type=int, default=NUM_JOBS,
                         help=f"Total job postings to generate across all industries (default: {NUM_JOBS})")
    parser.add_argument("--resumes-per-job", type=int, default=NUM_RESUMES_PER_JOB,
                         help=f"Resumes to generate per job posting (default: {NUM_RESUMES_PER_JOB})")
    parser.add_argument("--max-industries", type=int, default=1,
                         help="Number of industries to process (default: 1 for test runs; use 5 for full dataset)")
    parser.add_argument("--industry", type=str, default=None,
                         help="Run a specific industry by partial name match (overrides --max-industries)")
    parser.add_argument("--job-max-completion-tokens", type=int, default=JOB_MAX_COMPLETION_TOKENS)
    parser.add_argument("--resume-max-completion-tokens", type=int, default=RESUME_MAX_COMPLETION_TOKENS)
    parser.add_argument("--iteration", type=int, default=ITERATION,
                         help=f"Iteration number shared by both stages (default: {ITERATION} from .env)")

    # Correction flags (passed through to correction.py)
    parser.add_argument("--skip-correction", action="store_true",
                         help="Only run generation; do not run the correction pass afterward")
    parser.add_argument("--limit-jobs", type=int, default=None,
                         help="Correction stage: cap number of invalid job records processed")
    parser.add_argument("--limit-resumes", type=int, default=None,
                         help="Correction stage: cap number of invalid resume records processed")
    parser.add_argument("--generate-resumes-for-corrected-jobs", action="store_true",
                         help="Correction stage: also generate resumes for jobs that became valid via correction")
    parser.add_argument("--resumes-per-corrected-job", type=int, default=NUM_RESUMES_PER_JOB)

    args = parser.parse_args()

    generation_args = [
        "dataset_generator.py",
        "--num-jobs", str(args.num_jobs),
        "--resumes-per-job", str(args.resumes_per_job),
        "--max-industries", str(args.max_industries),
        "--job-max-completion-tokens", str(args.job_max_completion_tokens),
        "--resume-max-completion-tokens", str(args.resume_max_completion_tokens),
        "--iteration", str(args.iteration),
    ]
    if args.industry:
        generation_args += ["--industry", args.industry]

    returncode = run_stage("Stage 1/2: Generation", generation_args)
    if returncode != 0:
        print(f"\n❌ Generation failed (exit code {returncode}). Skipping correction.")
        sys.exit(returncode)

    if args.skip_correction:
        print("\n✅ Generation complete. Skipping correction (--skip-correction).")
        return

    correction_args = [
        "correction.py",
        "--iteration", str(args.iteration),
        "--job-max-completion-tokens", str(args.job_max_completion_tokens),
        "--resume-max-completion-tokens", str(args.resume_max_completion_tokens),
        "--resumes-per-corrected-job", str(args.resumes_per_corrected_job),
    ]
    if args.limit_jobs is not None:
        correction_args += ["--limit-jobs", str(args.limit_jobs)]
    if args.limit_resumes is not None:
        correction_args += ["--limit-resumes", str(args.limit_resumes)]
    if args.generate_resumes_for_corrected_jobs:
        correction_args += ["--generate-resumes-for-corrected-jobs"]

    returncode = run_stage("Stage 2/2: Correction", correction_args)
    if returncode != 0:
        print(f"\n❌ Correction failed (exit code {returncode}).")
        sys.exit(returncode)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete for iteration {args.iteration}.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
