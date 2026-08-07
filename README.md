# mini-project2 — Synthetic Resume/Job Dataset Generator

Generates paired job postings and resumes (with intentional fit/mismatch levels — Excellent, Good, Partial, Poor) for training or evaluating resume-job matching models. Calls an LLM (Groq or OpenRouter) through `instructor` for structured, Pydantic-validated output, and writes JSONL datasets plus structured event logs.

Five scripts:

1. **`dataset_generator.py`** — generates jobs and resumes.
2. **`correction.py`** *(optional, manual)* — fixes or regenerates any invalid records the generator produced.
3. **`pipeline.py`** *(optional)* — runs the two above back to back.
4. **`analysis.py`** *(optional, manual)* — rule-based (or optionally LLM-based) judge that scores each job/resume pair and charts the results.
5. **`api.py`** *(optional)* — FastAPI service exposing the same scoring logic as `analysis.py` over HTTP.

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
- All five scripts must be run with `src/` as the working directory (`cd src` first) — they import each other as sibling modules, not a package.

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Used when `GENERATION_MODEL`/`JUDGE_MODEL` has no `/` (Groq-native model ID, e.g. `llama-3.3-70b-versatile`). |
| `OPENROUTER_API_KEY` | Used when the model string contains `/` (OpenRouter slug, e.g. `qwen/qwen3-32b`). The provider is picked per-model, not globally, so `GENERATION_MODEL` and `JUDGE_MODEL` can be on different providers. |
| `GENERATION_MODEL` | Model used for all job/resume/correction generation calls. |
| `JUDGE_MODEL` | Model used only when LLM-based judging is explicitly requested (`analysis.py --llm-judge`, or `POST /review-resume` with `use_llm_judge: true`) — never called otherwise. `judge_pair()` in `dataset_generator.py` remains an unused stub. |
| `NUM_JOBS`, `NUM_RESUMES_PER_JOB` | Default job/resume counts, overridable via CLI flags. |
| `JOB_MAX_COMPLETION_TOKENS`, `RESUME_MAX_COMPLETION_TOKENS` | Default token budgets per batch call, overridable via CLI flags. |
| `JUDGE_MAX_COMPLETION_TOKENS` | Token budget per LLM-judge call (default `4000` — reasoning models like `gpt-oss-120b` spend most of this on hidden/visible reasoning before the answer, so this needs real headroom, not just space for the small JSON output). |
| `LLM_REQUEST_TIMEOUT_SECONDS` | Per-call network timeout (default `45`) applied to every LLM API call in the codebase. None of the underlying SDKs time out on their own in any reasonable window otherwise (openai-python defaults to 600s). |
| `ITERATION` | Default iteration number (output-file suffix), overridable via `--iteration`. |
| `RATE_LIMIT_TPM` | Tokens-per-minute cap; a sliding-window rate limiter sleeps before any batch call that would exceed it (Groq free tier defaults to 5000). If a single call's estimated tokens alone exceed the whole budget (common for judge calls with a large `JUDGE_MAX_COMPLETION_TOKENS`), it proceeds anyway once other tracked usage clears rather than waiting forever — waiting can never make an over-budget request fit. |
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

## 4. Analyze match quality — `analysis.py`

A separate, **manual** rule-based judge — never run automatically. `judge_pair()` in `dataset_generator.py` is an unimplemented stub (`judge_score` is always `null`); this script is the actual quality signal on a generated dataset, and needs no LLM calls at all. Run it after generation (and correction, if you ran it):

```bash
cd src
python analysis.py --iteration 1
python analysis.py --iteration 1 --skills-overlap-threshold 0.3   # tune the "low overlap" cutoff
python analysis.py --iteration 1 --skip-charts                    # labels only, no PNGs
python analysis.py --iteration 1 --llm-judge                      # also judge every pair via JUDGE_MODEL
```

| Flag | Default | Meaning |
|---|---|---|
| `--iteration` | `ITERATION` (.env) | Must match an existing `dataset_generator.py` run. |
| `--skills-overlap-threshold` | `0.5` | Below this, skills overlap counts as "low" and fails the pair. |
| `--skip-charts` | off | Only write `failure_labels_{iteration}.jsonl`; skip the 6 PNGs. |
| `--llm-judge` | off | Additionally run the same 6 rules through `JUDGE_MODEL` (one LLM call per pair) and write `failure_labels_llm_{iteration}.jsonl`. Extra LLM cost — opt-in. |
| `--judge-model` | `JUDGE_MODEL` (.env) | Override the model used for `--llm-judge`. |
| `--judge-max-completion-tokens` | `JUDGE_MAX_COMPLETION_TOKENS` (.env) | Token budget per judge call. |

It loads every pair (`pairs_{iteration}.jsonl` + `pairs_corrected_{iteration}.jsonl` if present) together with the jobs/resumes they reference, and scores each pair against 6 criteria — a pair only counts as a match (`job_resume_match: 1`) if **all 6** pass:

| Criterion | Rule |
|---|---|
| Skills overlap | `\|normalized(job.required_skills) ∩ normalized(resume.skills)\| / \|normalized(job.required_skills)\|`. Fails ("low") below `--skills-overlap-threshold` (default `0.5`). |
| Experience mismatch | Fails if total resume experience (summed across all entries) is under half the job's required years, **or** any two jobs on the resume have a gap ≥ 1 year. |
| Seniority mismatch | Resume's inferred level (from total experience: Entry/Mid/Senior/Lead-Principal/Exec) vs. the job's required level — fails if they differ by more than 1 level. |
| Missing core skills | Fails if any of the job's top-3 required skills is absent from the resume. |
| Hallucinated skills | Fails on: an under-2-year resume claiming "Expert" in 10+ skills; 30+ skills where most are "Expert"; phrases like "expert in all"/"certified in everything"; or inconsistent timelines (overlapping jobs, 2+ simultaneous "Present" roles). |
| Awkward language | Fails on: more than 5 corporate-jargon hits ("synergy", "move the needle", "circle back", etc.) across the summary/experience text; or the same word repeated 3+ times within a 25-word span. |

**Skills overlap deliberately uses `required_skills` only, not `preferred_skills`.** Preferred skills are optional nice-to-haves; including them was tried and reverted — jobs in this dataset list ~10.7 required+preferred skills combined while resumes list only ~4.6 skills on average, so requiring overlap against the *combined* set made `low_skills_overlap` fail on almost every pair regardless of actual fit quality, collapsing the match rate toward 0% and erasing any signal between good and bad pairs. Required-skills-only keeps the denominator small enough that overlap is actually discriminative.

**`--llm-judge` runs the exact same 6 rules through an LLM instead of Python** — the prompt spells out every threshold and formula above verbatim as instructions (not "is this a good match?"), so the LLM is evaluated against the same logic, not its own notion of fit. Useful as a sanity check on the rule-based judge, or to compare where they disagree. In spot checks the two have matched almost exactly on the same pair. Judge calls to reasoning models (e.g. `gpt-oss-120b`) occasionally degenerate into unparseable output — this now fails fast with a clear error (`InstructorRetryException`, which is fatal — see `LLM_REQUEST_TIMEOUT_SECONDS` above) rather than hanging.

**What it produces:**

| File | Contents |
|---|---|
| `data/failure_labels_{iteration}.jsonl` | One record per pair: `trace_id`, `industry`, `labeler: "judge"`, `skills_overlap` (float), `experience_mismatch`/`missing_skills`/`hallucinate_skills`/`awkward_language`/`job_resume_match` (0/1), and `level_mismatch` (the raw integer level difference, not binarized). True JSONL — one object per line. |
| `data/failure_labels_llm_{iteration}.jsonl` | Same schema, `labeler: "llm_judge"` — written only with `--llm-judge`. |
| `data/visualizations/failure_correlation_heatmap_{iteration}.png` | Pearson correlation between the 6 binarized failure flags. |
| `data/visualizations/failure_rates_by_fit_level_{iteration}.png` | Grouped bars: failure rate per criterion, by the generator's intended fit level (Excellent/Good/Partial/Poor). |
| `data/visualizations/failure_rates_by_template_{iteration}.png` | Same, grouped by writing-style template. |
| `data/visualizations/niche_vs_standard_{iteration}.png` | Same, grouped by niche vs. standard role. |
| `data/visualizations/schema_validation_heatmap_{iteration}.png` | Separate from the 6 match criteria — jobs/resumes schema-invalid rate per industry (from the original generation run's valid/invalid files). |
| `data/visualizations/hallucination_by_seniority_{iteration}.png` | Stacked bar of hallucination-flagged vs. not, by the resume's inferred seniority level. |

## 5. Review API — `api.py`

A FastAPI service exposing the same scoring logic as `analysis.py` over HTTP — `score_pair()`/`run_llm_judgment()` are shared functions `analysis.py`'s bulk CLI path and this API both call directly, so there is exactly one implementation of the 6 rules, not two.

```bash
cd src
uvicorn api:app --reload --port 8000
# or: python api.py
```

Interactive docs at `http://127.0.0.1:8000/docs` once running (FastAPI's built-in OpenAPI UI).

| Endpoint | Description |
|---|---|
| `POST /review-resume` | Score one job/resume pair on demand. Body: `{"job": {...JobContent fields...}, "resume": {...ResumeContent fields...}, "use_llm_judge": false, "skills_overlap_threshold": 0.5}`. **`use_llm_judge` defaults to `false`** (rule-based, instant, no network call) — set `true` to route through `JUDGE_MODEL` instead. Response is a single `failure_labels`-shaped record (`trace_id`, `industry`, `labeler`, `skills_overlap`, `experience_mismatch`, `level_mismatch`, `missing_skills`, `hallucinate_skills`, `awkward_language`, `job_resume_match`). A fatal LLM error (timeout, rate limit, auth, etc.) returns HTTP 502 with details rather than hanging. |
| `GET /health` | Liveness check — `{"status": "ok"}`. |
| `GET /analysis/failure-rates?iteration=1&labeler=judge` | Serves an already-computed `failure_labels_{iteration}.jsonl` (or `failure_labels_llm_{iteration}.jsonl` with `labeler=llm_judge`) — i.e. you must have run `analysis.py` for that iteration first, or this 404s with a message telling you so. Returns `{"iteration", "labeler", "count", "failure_rates": {...aggregate rate per criterion...}, "labels": [...the raw per-pair records, verbatim...]}`. |

The job/resume LLM judge client is created lazily on first `use_llm_judge: true` request, not at server startup — the default is no LLM judge, so a deployment that never uses it never even validates the judge API key.

## Notes

- There is no automated test suite. `src/test.py` is a standalone script that lists available Groq models (`client.models.list()`), not a pytest test.
- `validation.py` and `llm_errors.py` are internal library modules (Pydantic schemas, shared validation functions, fatal-error classification) — they aren't run directly, only imported by the scripts above.
- `judge_pair()` in `dataset_generator.py` remains an unimplemented stub (`judge_score` in every pair is always `null`) — the real quality judges for this project are `analysis.py` (rule-based by default, or `--llm-judge`) and `api.py`'s `/review-resume`.
