# mini-project2 — Synthetic Resume/Job Dataset Generator

Generates paired job postings and resumes (with intentional fit/mismatch levels — Excellent, Good, Partial, Poor) for training or evaluating resume-job matching models. Calls an LLM (Groq or OpenRouter) through `instructor` for structured, Pydantic-validated output, and writes JSONL datasets plus structured event logs.

Three scripts, run in sequence:

1. **`dataset_generator.py`** — generates jobs and resumes.
2. **`correction.py`** *(optional, manual)* — fixes or regenerates any invalid records the generator produced.
3. **`pipeline.py`** *(optional)* — runs the two above back to back.

## Setup

### Use copied environment (local quick start)

```bash
cd /Users/bvsaker/Dev/ai-bootcamp/mini-project2
source .venv/bin/activate
python -V
```

### Recreate environment from lock file (recommended for third parties)

```bash
cd /Users/bvsaker/Dev/ai-bootcamp/mini-project2
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

### Alternative baseline install

```bash
python -m pip install -r requirements.txt
```

Notes:
- `requirements.lock.txt` was generated from the active `mini-project1` `.venv` to capture exact package versions.
- For GitHub sharing, do **not** commit `.venv`; commit only dependency files and source code.
- Copy `.env.example` to `.env` and fill in API keys before running anything that calls an LLM.
- All three scripts must be run with `src/` as the working directory (`cd src` first) — they import each other as sibling modules, not a package.

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Used when `GENERATION_MODEL`/`JUDGE_MODEL` has no `/` (Groq-native model ID, e.g. `llama-3.3-70b-versatile`). |
| `OPENROUTER_API_KEY` | Used when the model string contains `/` (OpenRouter slug, e.g. `qwen/qwen3-32b`). The provider is picked based on this alone. |
| `GENERATION_MODEL` | Model used for all job/resume/correction generation calls. |
| `JUDGE_MODEL` | Currently unused — the judge step is a stub that always returns `None`/`null`. |
| `NUM_JOBS`, `NUM_RESUMES_PER_JOB` | Default job/resume counts, overridable via CLI flags. |
| `JOB_MAX_COMPLETION_TOKENS`, `RESUME_MAX_COMPLETION_TOKENS` | Default token budgets per batch call, overridable via CLI flags. |
| `ITERATION` | Default iteration number (output-file suffix), overridable via `--iteration`. |
| `RATE_LIMIT_TPM` | Tokens-per-minute cap; a sliding-window rate limiter sleeps before any batch call that would exceed it (Groq free tier defaults to 5000). |
| `JOB_MAX_BATCH_SIZE`, `RESUME_MAX_BATCH_SIZE` | How many jobs/resumes are requested per LLM call (small, e.g. 5/2, to stay under the TPM cap). |

## 1. Generate a dataset — `dataset_generator.py`

```bash
cd src
python dataset_generator.py                          # 1 industry, quick test run
python dataset_generator.py --max-industries 5        # all 5 industries
python dataset_generator.py --industry "Healthcare"   # partial-name match on one industry
python dataset_generator.py --num-jobs 50 --resumes-per-job 5
python dataset_generator.py --iteration 2             # separate output file set
```

| Flag | Default | Meaning |
|---|---|---|
| `--num-jobs` | `NUM_JOBS` (.env) | Total job postings across all selected industries. |
| `--resumes-per-job` | `NUM_RESUMES_PER_JOB` (.env) | Resumes generated per job posting. |
| `--max-industries` | `1` | How many of the 5 industries to run (in fixed order). |
| `--industry` | none | Run one specific industry by partial-name match; overrides `--max-industries`. |
| `--job-max-completion-tokens` | `JOB_MAX_COMPLETION_TOKENS` (.env) | Token budget per job-batch call. |
| `--resume-max-completion-tokens` | `RESUME_MAX_COMPLETION_TOKENS` (.env) | Token budget per resume-batch call. |
| `--iteration` | `ITERATION` (.env) | Suffix for all output files — lets you keep multiple runs side by side. |

**What it produces** (in `data/`, all suffixed `_{iteration}.jsonl`):

| File | Contents |
|---|---|
| `jobs_{iteration}.jsonl` | All generated job postings (valid + invalid, dual-written). |
| `resumes_{iteration}.jsonl` | All generated resumes (valid + invalid). |
| `pairs_{iteration}.jsonl` | Resume↔job pairs with fit level, template style, trace IDs. |
| `jobs_valid_{iteration}.jsonl` / `jobs_invalid_{iteration}.jsonl` | Jobs split by schema validation outcome. |
| `resumes_valid_{iteration}.jsonl` / `resumes_invalid_{iteration}.jsonl` | Resumes split by schema validation outcome. |
| `logs/dataset_log_{iteration}.jsonl` | Structured JSONL event log — run start/complete/aborted, batch starts, classified LLM errors, each with a full traceback where relevant. |

Invalid records are written as two consecutive lines: the raw record, then a JSON object of `validation_errors`. If an LLM batch call returns *fewer* items than requested, the missing slots are also recorded as invalid (`error_type: "llm_missing_job"`/`"llm_missing_resume"`), carrying enough context for the correction pass to regenerate them later.

Re-running with the same `--iteration` **overwrites** those files, but any existing ones are backed up first into a timestamped folder under `data/backups/` — nothing is silently lost.

**A genuine LLM/API failure halts the run**: rate limits, auth errors, connection failures, or the model persistently failing to return usable output will stop the whole script (non-zero exit) rather than continuing to the next batch or industry. Ordinary content problems (a malformed field, a short batch) never do this — they're captured as invalid records instead, for the correction pass to handle.

## 2. Correct invalid records — `correction.py`

A separate, **manual** step — it is never run automatically by `dataset_generator.py`. Run it after a generation run, pointing at the same iteration:

```bash
cd src
python correction.py --iteration 1
python correction.py --iteration 1 --skip-resumes                        # jobs only
python correction.py --iteration 1 --skip-jobs                           # resumes only
python correction.py --iteration 1 --limit-jobs 10 --limit-resumes 10    # cap cost while testing
python correction.py --iteration 1 --generate-resumes-for-corrected-jobs # also backfill resumes for newly-valid jobs
```

It reads `jobs_invalid_{iteration}.jsonl` / `resumes_invalid_{iteration}.jsonl` and, for each record, asks the LLM to either fix the specific validation errors (if real content came back but failed schema checks) or generate it fresh (if the slot was simply never returned by the batch call). Each result is re-validated with the exact same rules generation used.

| Flag | Default | Meaning |
|---|---|---|
| `--iteration` | `ITERATION` (.env) | Must match an existing `dataset_generator.py` run. |
| `--job-max-completion-tokens`, `--resume-max-completion-tokens` | same as generator | Token budgets for correction calls. |
| `--limit-jobs`, `--limit-resumes` | none (no cap) | Process at most N invalid records of each kind — useful for a cheap test before running the full correction pass. |
| `--skip-jobs`, `--skip-resumes` | off | Skip one half of the correction pass entirely. |
| `--generate-resumes-for-corrected-jobs` | off | Opt-in: a job invalid at generation time never had resumes generated for it (phase 2 only ran for already-valid jobs). This flag backfills a full batch of resumes for any job that becomes valid via correction. Additional LLM cost — off by default. |
| `--resumes-per-corrected-job` | `NUM_RESUMES_PER_JOB` (.env) | Resumes per job for the flag above. |

**What it produces** (new files — the original run's 7 files are never touched):

| File | Contents |
|---|---|
| `jobs_corrected_{iteration}.jsonl` / `resumes_corrected_{iteration}.jsonl` | Records that became valid after correction. Same trace IDs as before (`res_37` stays `res_37`), just fixed content. |
| `jobs_uncorrectable_{iteration}.jsonl` / `resumes_uncorrectable_{iteration}.jsonl` | Records still invalid after a correction attempt, or skipped outright (e.g. a resume whose job was never itself corrected) — same dual-line format as the original invalid files, so they can be fed through another pass. |
| `pairs_corrected_{iteration}.jsonl` | New pairs for corrected resumes. **Only created when both the resume and its job are valid** — a resume is never paired (or even sent to the LLM) if its job can't be resolved. |

A full valid dataset after correction is `jobs_valid_* + jobs_corrected_*` (same idea for resumes and pairs). Correction events append into the same `logs/dataset_log_{iteration}.jsonl` as the generation run. Re-running correction for the same iteration backs up its own prior output files first (separately from the generator's backup — correction never disturbs the invalid files it reads as input).

## 3. Run both stages together — `pipeline.py`

Chains generation then correction as two subprocesses against the same iteration, so you don't have to remember to run `correction.py` separately or keep the `--iteration` values in sync:

```bash
cd src
python pipeline.py --num-jobs 50 --resumes-per-job 5
python pipeline.py --max-industries 5 --iteration 2
python pipeline.py --num-jobs 50 --skip-correction                       # generation only
python pipeline.py --num-jobs 50 --generate-resumes-for-corrected-jobs   # passed through to correction
```

Accepts the union of both scripts' flags (generation flags plus `--skip-correction`, `--limit-jobs`, `--limit-resumes`, `--generate-resumes-for-corrected-jobs`, `--resumes-per-corrected-job`). If generation fails or aborts fatally, the pipeline exits with that same code and skips correction entirely.

## Notes

- There is no automated test suite. `src/test.py` is a standalone script that lists available Groq models (`client.models.list()`), not a pytest test.
- `validation.py` and `llm_errors.py` are internal library modules (Pydantic schemas, shared validation functions, fatal-error classification) — they aren't run directly, only imported by the three scripts above.
- `JUDGE_MODEL` / the judge step is an unimplemented stub; `judge_score` in every pair is always `null`.
