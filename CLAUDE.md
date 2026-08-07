# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A synthetic dataset generator that produces paired job postings and resumes (with intentional fit/mismatch levels) for training or evaluating resume-job matching models. It calls an LLM (via Groq or OpenRouter) through `instructor` for structured, Pydantic-validated output, and writes JSONL datasets plus structured event logs.

## Environment setup

```bash
cd /Users/bvsaker/Dev/ai-bootcamp/mini-project2
source .venv/bin/activate        # copied venv, Python 3.13
python -V
```

To rebuild from scratch:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt   # exact pinned versions (preferred)
# or: python -m pip install -r requirements.txt  # loose baseline
```

`.venv` is not committed — only dependency files and source. Copy `.env.example` to `.env` and fill in API keys before running anything that calls an LLM.

## Running the generator

Entry point is `src/dataset_generator.py`. It must be run with `src/` as the working directory (it imports `resume_generator`, `validation`, and `llm_errors` as top-level sibling modules, not a package):

```bash
cd src
python dataset_generator.py                          # 1 industry, test run
python dataset_generator.py --max-industries 5        # all 5 industries
python dataset_generator.py --industry "Healthcare"   # partial-name match on one industry
python dataset_generator.py --num-jobs 50 --resumes-per-job 5
python dataset_generator.py --job-max-completion-tokens 2200 --resume-max-completion-tokens 3000
python dataset_generator.py --iteration 2              # separate output file set
```

A fatal LLM/API error (rate limit, auth, connection failure, or instructor exhausting retries — see `llm_errors.is_fatal_llm_error`) halts the whole run and exits non-zero. Everything else (schema-invalid content, an LLM batch returning fewer items than requested) is captured as an invalid record instead — see **Fatal vs. recoverable LLM errors** below.

### Correcting invalid records

`src/correction.py` is a separate, standalone pass — **it is never invoked automatically by `dataset_generator.py`**. Run it manually after a generation run, targeting the same `--iteration`:

```bash
cd src
python correction.py --iteration 1
python correction.py --iteration 1 --skip-resumes                       # jobs only
python correction.py --iteration 1 --limit-jobs 10 --limit-resumes 10   # cost-capped test run
python correction.py --iteration 1 --generate-resumes-for-corrected-jobs # also backfill resumes for newly-valid jobs
```

It reads `jobs_invalid_{iteration}.jsonl` / `resumes_invalid_{iteration}.jsonl`, asks the LLM to fix (or, for missing batch slots, freshly generate) each record, re-validates with the same `validate_job`/`validate_resume` functions generation uses, and writes results to `*_corrected_{iteration}.jsonl` / `*_uncorrectable_{iteration}.jsonl` — it never mutates the original run's files. See **Correction pass** below for the fix-vs-regenerate distinction and pairing rules.

### Running both stages together

`src/pipeline.py` chains generation then correction (as two real subprocesses, so their module-level state — rate limiters, open log file handles — never leaks between stages) against the same iteration:

```bash
cd src
python pipeline.py --num-jobs 50 --resumes-per-job 5
python pipeline.py --max-industries 5 --iteration 2
python pipeline.py --num-jobs 50 --skip-correction                      # generation only
python pipeline.py --num-jobs 50 --generate-resumes-for-corrected-jobs
```

If generation fails or aborts fatally, the pipeline exits with that same code and skips correction.

### Analyzing match quality

`src/analysis.py` is a rule-based judge by default (no LLM calls) — also a separate, manual step, never invoked by the other three scripts. It fills the gap left by `judge_pair()` (stub, always returns `None`/`judge_score` stays `null`):

```bash
cd src
python analysis.py --iteration 1
python analysis.py --iteration 1 --skills-overlap-threshold 0.3
python analysis.py --iteration 1 --skip-charts
python analysis.py --iteration 1 --llm-judge          # also judge every pair via JUDGE_MODEL
```

Scores every pair against 6 criteria and writes `data/failure_labels_{iteration}.jsonl`
plus 6 `.png` charts to `data/visualizations/`. `--llm-judge` additionally runs the
identical 6 rules through `JUDGE_MODEL` (prose-ified into the prompt, not "is this a
good fit?") and writes `data/failure_labels_llm_{iteration}.jsonl` (`labeler: "llm_judge"`).
See **Match analysis** below for the exact rules and the required-skills-only decision.

### Running the review API

`src/api.py` is a FastAPI service exposing the same scoring functions (`score_pair()`/
`run_llm_judgment()`, imported directly from `analysis.py` — no duplicated rules) over
HTTP. Never invoked by the other scripts either.

```bash
cd src
uvicorn api:app --reload --port 8000
# or: python api.py
```

`POST /review-resume` (`use_llm_judge` defaults to `false`), `GET /health`,
`GET /analysis/failure-rates?iteration=N&labeler=judge|llm_judge` (serves an
already-computed `failure_labels*` file — 404s if `analysis.py` hasn't been run for
that iteration yet). See **API service** below for request/response shapes.

There is no test suite (`src/test.py` is a standalone script that lists available Groq models via `client.models.list()`, not a pytest test — despite `pytest`/`pytest-mock` being in `requirements.txt`).

## Configuration (`.env`)

- `GROQ_API_KEY` — used when a given model string has no `/` (Groq-native model ID, e.g. `llama-3.3-70b-versatile`).
- `OPENROUTER_API_KEY` — used when the model string contains `/` (OpenRouter slug, e.g. `qwen/qwen3-32b`). `make_client(model=GENERATION_MODEL)` in `dataset_generator.py` picks the provider **per-model** (not globally), so it can also be called as `make_client(JUDGE_MODEL)` when `GENERATION_MODEL`/`JUDGE_MODEL` are on different providers.
- `GENERATION_MODEL`, `JUDGE_MODEL` — `judge_pair()` in `dataset_generator.py` remains an unused stub (`judge_score` in every pair is always `null`), but `JUDGE_MODEL` itself is actively used by `analysis.py --llm-judge` and `api.py`'s `POST /review-resume` with `use_llm_judge: true` — both call it only when explicitly requested, never by default.
- `NUM_JOBS`, `NUM_RESUMES_PER_JOB`, `JOB_MAX_COMPLETION_TOKENS`, `RESUME_MAX_COMPLETION_TOKENS`, `JUDGE_MAX_COMPLETION_TOKENS` — defaults, overridable via CLI flags. `JUDGE_MAX_COMPLETION_TOKENS` defaults to `4000` (not a small value like the label schema itself would need) because reasoning models (e.g. `gpt-oss-120b` via OpenRouter) spend most of the budget on hidden/visible reasoning tokens before the answer; too small a budget causes truncation, which forces expensive retries rather than actually saving time.
- `LLM_REQUEST_TIMEOUT_SECONDS` (default `45`) — per-call `timeout=` passed to every `client.chat.completions.create(...)` call in the codebase (7 call sites across `dataset_generator.py`/`resume_generator.py`/`correction.py`/`analysis.py`). Added because none of the underlying SDKs impose a reasonable timeout on their own (openai-python defaults to 600s) — without this, a slow/looping call could hang indefinitely, which is unacceptable for `api.py`'s synchronous `POST /review-resume`. `analysis.py`'s `run_llm_judgment()` (the one call site an HTTP request waits on directly) additionally uses `max_retries=2` instead of the usual 3, bounding worst-case latency at `2 * LLM_REQUEST_TIMEOUT_SECONDS` rather than 3x.
- `ITERATION`, `RATE_LIMIT_TPM`, `JOB_MAX_BATCH_SIZE`, `RESUME_MAX_BATCH_SIZE` — iteration/output-file suffix, and batching/rate-limit tuning (Groq free-tier TPM defaults to 5000; batch sizes are small — 5 jobs/batch, 2 resumes/batch — to stay under it). `RateLimiter.wait_if_needed()` (in `dataset_generator.py`) has a specific fix worth knowing about: if a *single* call's estimated tokens alone exceed `RATE_LIMIT_TPM` (easy to hit with a large `JUDGE_MAX_COMPLETION_TOKENS` against a modest TPM budget), the naive "wait until there's room" loop can never succeed — waiting doesn't shrink the request. It now detects that case and proceeds once any other tracked usage has cleared, instead of sleeping in an infinite loop (this was a real, reproducible hang before the fix).

## Architecture

**Module layout**: `dataset_generator.py` (entry point, CLI, `process_industry`, `Logger`, `RateLimiter`, prompt builders, config) and `resume_generator.py` (`generate_resumes_for_job`, batch/single resume generation) are the generation pipeline. `validation.py` owns every Pydantic model (`JobContent`/`JobPosting`/`ResumeContent`/`Resume`/`ResumePair`/`RawJobBatch`/`RawResumeBatch`/etc.) plus the shared `validate_job()`/`validate_resume()` functions, so it has zero dependency on the other two — this is what lets both `dataset_generator.py` and `resume_generator.py` (and `correction.py`) import the same validation logic without a three-way circular import. `llm_errors.py` owns `is_fatal_llm_error()`, dependency-light so it's importable everywhere. `correction.py`, `pipeline.py`, and `analysis.py` sit on top and import from the others, but nothing imports from them — except `api.py`, which sits one layer above `analysis.py` and imports its pair-scoring functions directly. Nothing imports from `api.py`.

**Two-phase generation per industry** (`process_industry` in `dataset_generator.py`): first all job postings for an industry are generated in batches, then resumes are generated per job.
- If job generation for an industry produces zero valid jobs, that industry is skipped entirely (0 resumes) — logged, not fatal.
- If resume generation fails for a job with a non-fatal error, remaining jobs in that industry are abandoned (partial results already written are kept), but other industries still run.
- A **fatal** LLM/API error (see below) skips all of that and propagates straight out of `process_industry`, aborting the entire run.

**Fatal vs. recoverable LLM errors** (`llm_errors.py`): `is_fatal_llm_error(exc)` classifies exceptions raised at the `client.chat.completions.create(...)` call sites. Fatal: `RateLimitError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `APIConnectionError`/`APITimeoutError` (both `groq` and `openai` packages — they're separate, same-named exception hierarchies, so both are listed explicitly), and instructor's `InstructorRetryException` (retries exhausted without ever getting parseable output). Non-fatal: `BadRequestError`, `UnprocessableEntityError`, `ConflictError`, `InternalServerError`, bare `APIStatusError`. Fatal exceptions are always re-raised **bare** (never wrapped in `RuntimeError`) at every boundary — `_generate_resumes_batch` → `generate_resumes_for_job` → `process_industry` → `main()` — because wrapping loses the `isinstance`-checkable type the classifier needs. `main()` catches the propagated exception, logs `run_aborted`, prints partial results, and `sys.exit(1)`s instead of continuing to the next industry.

Content-level problems never reach this classifier at all — they're handled entirely by the validation/dual-write path below and never raise.

**Batching**: jobs and resumes are requested from the LLM in small batches (`JOB_MAX_BATCH_SIZE`, `RESUME_MAX_BATCH_SIZE`) rather than one-by-one, to reduce API calls while staying under `RATE_LIMIT_TPM`. `RateLimiter` (sliding 60s window) in `dataset_generator.py` throttles before every batch call based on an estimated token count (`prompt_chars // 4 + max_completion_tokens`).

**Loose-then-validate pattern**: the LLM is asked to return `RawJobBatch`/`RawResumeBatch` (lists of untyped `dict`s via `instructor`) rather than the strict `JobBatch`/`ResumeBatch` models directly — this is also why instructor's own retry-on-validation-failure loop never fires for ordinary content problems (only for responses that aren't even list-shaped). Each item is then individually normalized and validated via `validation.validate_job()`/`validate_resume()` against the strict Pydantic schema (`JobContent`/`ResumeContent`). One malformed item in a batch doesn't fail the whole batch — it's captured as an invalid record instead. `normalize_job_content()` also backfills/repairs older or flat-shaped LLM output (e.g. missing `job_title`, legacy top-level fields) before validation. If the LLM returns *fewer* items than requested, the missing slots are logged (`llm_batch_incomplete`) and synthesized as invalid records too (`error_type: "llm_missing_job"` / `"llm_missing_resume"`), carrying enough context (`industry` for jobs; `job_trace_id`, `job_title`, `fit_level`, `template` for resumes) for `correction.py` to regenerate them later.

**Dual-write for valid/invalid records**: every run writes to both a combined file (`jobs_{iter}.jsonl`, `resumes_{iter}.jsonl`) and split valid/invalid files (`jobs_valid_{iter}.jsonl`, `jobs_invalid_{iter}.jsonl`, etc.). Invalid records are written as two consecutive JSONL lines: the raw record, then a JSON object with `validation_errors` (or a synthetic message for missing-slot records).

**Correction pass** (`correction.py`, never invoked automatically): reads `jobs_invalid_{iteration}.jsonl` / `resumes_invalid_{iteration}.jsonl` and, for each record, either asks the LLM to fix the flagged validation errors (`error_type: "pydantic_validation_error"` — real content, something wrong in a field) or to freshly regenerate it (`error_type: "llm_missing_job"`/`"llm_missing_resume"` — the LLM never returned this slot, so there's nothing to fix). Both paths call the LLM for a single-item `RawJobBatch`/`RawResumeBatch` (never the strict model directly, for the same instructor-retry reason as generation) and re-validate with the identical `validate_job`/`validate_resume` functions. Trace IDs are preserved (`res_37` stays `res_37`), only `generated_at` refreshes. Results go to new files — `jobs_corrected_{iteration}.jsonl`, `jobs_uncorrectable_{iteration}.jsonl`, `resumes_corrected_{iteration}.jsonl`, `resumes_uncorrectable_{iteration}.jsonl`, `pairs_corrected_{iteration}.jsonl` — correction.py never mutates the original run's 7 files. A `ResumePair` is only ever created when both sides are valid: job corrections run first and merge into a `job_lookup` (originally-valid + corrected), and a resume is only paired (and only sent to the LLM at all) if its `job_trace_id` resolves there — otherwise it's marked `correction_skipped_missing_job_context` with no LLM call. Pair trace IDs reuse the number embedded in the resume's own trace_id (`res_37` → `pair_37`), which is provably collision-free since pair numbering is always 1:1 with resume numbering in the original run. `--generate-resumes-for-corrected-jobs` (opt-in, off by default) additionally runs full resume generation for jobs that were invalid at generation time and therefore never went through Phase 2 at all — it scans existing `res_N` trace IDs to pick safe non-colliding counters so `get_fit_level()`'s dataset-wide cycling isn't disrupted.

**Pipeline** (`pipeline.py`): chains generation then correction as two subprocesses (not in-process calls) against the same `--iteration`, so their module-level singletons (`RateLimiter`, open `Logger` file handles) never leak between stages. Propagates whichever stage's exit code is non-zero and skips correction if generation fails.

**Match analysis** (`analysis.py`, never invoked automatically, no LLM calls): loads every pair (`pairs_{iteration}.jsonl` + `pairs_corrected_{iteration}.jsonl` if present) with the `JobPosting`/`Resume` they reference, and scores each against 6 criteria — `job_resume_match = 1` only if all 6 pass:
- **Skills overlap** — `|normalized(job.requirements.required_skills) ∩ normalized(resume.skills)| / |normalized(required_skills)|` (`normalize_skill()`: lowercase → strip parentheticals/version tokens → strip punctuation → strip trailing plural `s`). Fails below `--skills-overlap-threshold` (default `0.5`). **Deliberately `required_skills` only — `preferred_skills` was tried and reverted**: jobs here average ~10.7 required+preferred skills combined vs. resumes averaging only ~4.6 skills listed, so overlap against the combined set made `low_skills_overlap` fail on almost every pair regardless of actual fit, collapsing the match rate toward 0% with no discriminative signal between fit levels.
- **Experience mismatch** — total resume experience (`total_experience_years()`, summing all entries, `"Present"`/`None` resolved against the resume's own `generated_at` for reproducibility) `< job.requirements.experience_years / 2`, **or** `has_experience_gap()` finds any two sorted entries with a gap `>= 1` year.
- **Seniority mismatch** — `infer_resume_level()` buckets total experience into the same bands `build_jobs_batch_prompt` uses for jobs (`<2→Entry`, `2–5→Mid`, `5–10→Senior`, `10–15→Lead/Principal`, `15+→Exec`; jobs never require the 5th "Exec" band, only resumes can land there) and diffs it against `JOB_LEVELS[job.requirements.experience_level]`. `level_mismatch` in the output is the **raw integer difference** (not binarized); it fails the pair when `> 1`.
- **Missing core skills** — binary: any of `required_skills[:3]` absent from the resume.
- **Hallucinated skills** — binary, any sub-check trips it: `<2yr` resume with `>=10` "Expert" skills; `>=30` skills where most are "Expert"; a hard-coded phrase (`"expert in all"`, `"certified in everything"`, etc.); or a timeline inconsistency (`has_overlapping_jobs()`, or `>1` entries with `end_date` in `(None, "Present")`).
- **Awkward language** — binary: total occurrences of a jargon-phrase list (`"synergy"`, `"move the needle"`, `"circle back"`, etc.) across `summary` + all `responsibilities`/`achievements` exceeds `5`, or the same non-stopword appears `>=3` times in any 25-word sliding window.

Writes `data/failure_labels_{iteration}.jsonl` (true JSONL, one object per line — not the wrapped-array shape sometimes used to *illustrate* a schema) with `trace_id`, `industry` (from `job.company.industry`), `labeler: "judge"`, `skills_overlap`, `level_mismatch`, and the 4 other binary flags. A second, richer in-memory `pandas.DataFrame` (same computation pass, extra columns `fit_level`/`template_style`/`is_niche_role`/inferred levels — never written to the label file) drives 6 matplotlib charts saved to `data/visualizations/`: a failure-mode correlation heatmap, failure rates grouped by fit level / template / niche-vs-standard, a schema-validation invalid-rate heatmap by industry (reuses `correction.load_jsonl_pairs()` on the `*_invalid_*` files — this one chart is about generation-time schema validity, not pair-match quality), and hallucination flags stacked by inferred resume seniority.

**Scoring logic is split into pure functions specifically so `api.py` can reuse it**: `score_pair(job, resume, threshold) -> dict` holds everything `analyze_pair()` computes (the 6 rules), but takes no `ResumePair` and returns only the 7 scored fields — no `trace_id`/`industry`/`labeler`, which are identity metadata the caller attaches. `analyze_pair(job, resume, pair, threshold)` is now a thin wrapper: calls `score_pair()`, prepends `trace_id`/`industry`/`labeler: "judge"` from `pair`/`job`, and separately still returns `chart_context` (`fit_level`/`template_style` from `pair`, bulk-script-only). Same split for the LLM path: `run_llm_judgment(client, model, job, resume, threshold, max_completion_tokens) -> LLMJudgment` is the pure LLM call (`build_judge_prompt()` + `client.chat.completions.create()`); `llm_judge_pair(...)` wraps it the same way `analyze_pair` wraps `score_pair`. This was a pure extraction — confirmed the bulk CLI path's `failure_labels_{iteration}.jsonl` output is byte-for-byte unchanged after the split.

**LLM judge** (`analysis.py --llm-judge` / `api.py`'s `use_llm_judge: true`, off by default in both): `build_judge_prompt(job, resume, threshold)` restates the exact same 6 rules as explicit prose instructions (thresholds, formulas, phrase lists spelled out verbatim) rather than asking the model's own notion of "good fit" — the point is evaluating the LLM against identical logic to the rule-based judge, not a different rubric. Response schema is `LLMJudgment` (a plain strict Pydantic model, not the loose-then-validate `RawJobBatch`/`RawResumeBatch` pattern generation uses — a single ad-hoc judgment doesn't need batch semantics). `run_llm_judgment()` is the only call site with `max_retries=2` (every other call site uses 3) and is the one call site an HTTP request waits on synchronously, so its worst-case latency is deliberately bounded tighter. In spot checks, the LLM judge's output on a given pair has matched the rule-based judge's almost exactly, which is the intended outcome of prompting from the same rules rather than a free-form rubric.

**API service** (`api.py`, FastAPI, never invoked by the other scripts): `POST /review-resume` accepts `JobContent`/`ResumeContent` (the LLM-output-shaped schemas with no system metadata — FastAPI validates them automatically as the request body) and synthesizes the `JobPosting`/`Resume` metadata fields (`trace_id`, `generated_at`, and for resumes the unused-by-scoring `prompt_template`/`fit_level`/`writing_style` placeholders) before calling `score_pair()` or `run_llm_judgment()` directly — same functions, same rules, as `analysis.py`. The judge client is created lazily on first `use_llm_judge: true` request (module-level cache in `api.py`), not at import time, matching "off by default." `GET /analysis/failure-rates` doesn't compute anything — it reads whichever `failure_labels*_{iteration}.jsonl` `analysis.py` already wrote and 404s if that file doesn't exist yet, plus computes an aggregate `failure_rates` dict alongside the verbatim `labels` array.

**Fit-level and template distribution is deterministic and cyclic, not random**:
- `distribute_experience_levels(n)` cycles Entry → Mid → Senior → Lead/Principal across jobs in an industry.
- `get_fit_level(global_resume_index)` cycles Excellent → Good → Partial → Poor across *all* resumes in the run (global index), so fit levels balance across the whole dataset, not per-job.
- `get_template_name(resume_index_within_job)` cycles through the 5 writing-style templates per job.
- `get_shortcoming_instructions()` turns a fit level into concrete prompt instructions that tell the LLM exactly which mismatches to introduce (wrong experience level, missing skills, unrelated domain, etc.) for Partial/Poor fits — the "poor fit" resumes are still meant to read as realistic people, not garbage.

**`resume_generator.py`** imports `Resume`/`ResumePair`/`ResumeContent`/`RawResumeBatch`/`validate_resume`/`JobPosting` from `validation.py` at module top level (no circularity risk — `validation.py` depends on neither `dataset_generator` nor `resume_generator`). It still imports a handful of names *inside* its functions (`GENERATION_MODEL`, `get_template_name`, `get_shortcoming_instructions`, `format_job_summary`, `RESUME_MAX_BATCH_SIZE`, `_rate_limiter`, `get_fit_level_brief`) to avoid a circular import, since those still live in `dataset_generator.py`, which also imports `generate_resumes_for_job` from `resume_generator.py` at module top level. All files must be run/imported from `src/` for this to resolve.

**Templates** (`templates/*.json`) define writing-style personas (formal_corporate, casual_startup_friendly, tech_detail_heavy, achievement_focused_metrics, career_changer_xfer_skills), each with `display_name`, `writing_style`, `persona`, and `instructions` fields injected directly into resume-generation prompts.

**Output layout**:
- `data/jobs_{iteration}.jsonl`, `data/resumes_{iteration}.jsonl`, `data/pairs_{iteration}.jsonl` — combined generation outputs.
- `data/{jobs,resumes}_valid_{iteration}.jsonl`, `data/{jobs,resumes}_invalid_{iteration}.jsonl` — split by validation outcome.
- `data/{jobs,resumes}_corrected_{iteration}.jsonl`, `data/{jobs,resumes}_uncorrectable_{iteration}.jsonl`, `data/pairs_corrected_{iteration}.jsonl` — written only by `correction.py`. A full valid dataset is `jobs_valid_* + jobs_corrected_*` (same for resumes/pairs).
- `data/failure_labels_{iteration}.jsonl` (`labeler: "judge"`), `data/failure_labels_llm_{iteration}.jsonl` (`labeler: "llm_judge"`, only with `--llm-judge`), `data/visualizations/*_{iteration}.png` — written only by `analysis.py`. `api.py` only ever *reads* `failure_labels*` files (`GET /analysis/failure-rates`); it never writes any of the per-iteration data files.
- `logs/dataset_log_{iteration}.jsonl` — structured JSONL event log (`Logger` class), shared by both generation and correction (correction appends into the same file): run start/complete/aborted, batch starts, `llm_batch_incomplete`, classified LLM errors (`llm_error_rate_limit`, `llm_error_max_retries`, `llm_error_json_validation`, generic `llm_error`) each with full traceback, and correction events (`correction_run_start/complete`, `correction_attempt`, `correction_success`, `correction_failed`).
- Re-running `dataset_generator.py` with the same `--iteration` **overwrites** prior outputs, but `backup_iteration_outputs()` first moves any existing *generation* files for that iteration into a timestamped folder under `data/backups/`. `correction.py` has its own analogous `backup_correction_outputs()` for its 5 files — deliberately separate, since correction *reads* the invalid files as input and must never have them moved out from under it by a shared backup call.
- Trace IDs (`job_N`, `res_N`, `pair_N`) are assigned from global counters that advance by the *planned* slot count per industry, keeping IDs unique and stable even when generation partially fails mid-industry. Correction preserves these IDs rather than reissuing them.
