# Session 5 — Cloud Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Data lives in MongoDB Atlas, built by a GitHub Actions pipeline with no local machine. The served ALS model is trained on all data and stored on the HF Hub. CI runs Mongo integration tests. A minimal FastAPI `/health` Space is live, auto-deployed, and kept awake.

**Architecture:**
- **Pipeline:** a manual `pipeline.yaml` chains the existing CLIs (ingest → evaluate → train → embed → index) against Atlas, then uploads `model.pkl` + `manifest.json` to an HF Hub model repo and restarts the API Space.
- **API:** the API Space is a Docker Space built from the repo. At startup it downloads the manifest and serves `/health`.
- **Embedding:** the GPU embedding microservice is deleted, and torch switches to CPU wheels everywhere.

**Tech Stack:** Python 3.12, uv, FastAPI, pymongo, implicit, sentence-transformers (CPU torch), `huggingface_hub`, GitHub Actions, MongoDB Atlas M0, `mongodb/mongodb-atlas-local:8.0` (CI service).

**Spec:** `docs/superpowers/specs/2026-11-25-cloud-hosted-replan-design.md`, plus `plan.md` → Session 5.

## Global Constraints

- $0: free tiers only. Nothing may require running anything on the owner's machine.
- Python 3.12; `uv sync --frozen` everywhere; `ruff check`, strict `mypy`, and `pytest` must be clean at every commit.
- torch comes from the CPU wheel index `https://download.pytorch.org/whl/cpu` (no CUDA wheels anywhere).
- Text model `BAAI/bge-small-en-v1.5` (384-d); `TEXT_EMBED_DIM` stays 384.
- Secrets (`TMDB_API_KEY`, `MONGODB_URI`, `HF_TOKEN`, `API_KEY`, `LLM_API_KEY`) are never committed, printed, or logged.
- Atlas gets no `tmdb_cache` collection from the pipeline: TMDB payloads live in a file cache persisted with `actions/cache`.
- HF Space app port is `7860`.
- Keep-alive cadence is every 12 h.
- Integration tests use uuid-suffixed DB names when they touch vector indexes (Risk #6).

## Review Focus

1. **HF Hub unreachable or artifact missing at API startup.** `/health` still returns 200 with `"status": "degraded"` and `"artifact": null`, with no crash loop. *(Task 7 test.)*
2. **Mongo unreachable from the Space.** `/health` answers within about 3 s with `"mongo": "down"`, not a hang or a 500. *(Task 7 test; `serverSelectionTimeoutMS=2000` in `get_client`.)*
3. **Corrupt or partial TMDB cache file restored from `actions/cache`.** It's treated as a miss and refetched, not raised. *(Task 2 test.)*
4. **Pipeline re-run on already-populated Atlas.** Ingest counts are unchanged, embed processes 0, the index already exists, and the artifact is overwritten. *(Task 3 integration test + Task 9 DoD.)*
5. **Secrets in logs.** The HF token and Atlas URI never appear in pipeline logs, and the `httpx` logger stays at WARNING in every new CLI. *(Task 6 test asserts that `manifest.json` contains no `mongodb` URI or token. Task 8 workflows pass secrets only via `env:`.)*

---

### Task 0: Accounts and secrets (owner, in a browser — no code)

- [ ] **Atlas:** create a free M0 cluster. Add a database user `movie_recs_app` (readWrite on `movie_recs` only) with a generated password. Set Network Access to `0.0.0.0/0`. Copy the `mongodb+srv://` URI.
- [ ] **Hugging Face:**
  - Create a Docker Space `<user>/movie-recs-api` (blank, CPU basic) and a model repo `<user>/movie-recs-artifacts` (public is fine).
  - Create a **fine-grained** token with write access to those two repos only.
  - In the Space settings, add the secrets `MONGODB_URI` and `HF_TOKEN`, and the variables `HF_ARTIFACT_REPO=<user>/movie-recs-artifacts` and `MONGODB_DB=movie_recs`.
- [ ] **GitHub repo → Settings → Secrets and variables → Actions:**
  - Secrets: `TMDB_API_KEY`, `MONGODB_URI`, `HF_TOKEN`.
  - Variables: `HF_API_SPACE=<user>/movie-recs-api`, `HF_ARTIFACT_REPO=<user>/movie-recs-artifacts`, `API_HEALTH_URL=https://<user>-movie-recs-api.hf.space/health`.

---

### Task 1: CPU torch + retire the GPU embedding service + config cleanup

**Files:**
- Modify: `pyproject.toml` (ml group + uv index), `uv.lock`, `src/movie_recs/config.py`, `.env.example`, `compose.yaml`, `Makefile`, `src/movie_recs/embeddings/run.py` (docstring), `src/movie_recs/db/client.py`
- Delete: `src/movie_recs/embeddings/service.py`, `tests/unit/test_embeddings_service.py`, `docker/embedding.Dockerfile`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `Settings` fields:
  - `llm_base_url: AnyUrl = AnyUrl("https://api.groq.com/openai/v1")`
  - `llm_model: str = "llama-3.1-8b-instant"`
  - `llm_api_key: str | None = None`
  - `hf_token: str | None = None`
  - `hf_artifact_repo: str | None = None`
  - `embed_device: str | None = "cpu"`
- Removed fields: `vllm_base_url`, `vllm_model`, `embed_base_url`, `cloudflare_tunnel_token`.
- `get_client()` now passes `serverSelectionTimeoutMS=2000`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_config.py`:

```python
def test_llm_and_hf_settings_have_cloud_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert str(s.llm_base_url) == "https://api.groq.com/openai/v1"
    assert s.hf_artifact_repo is None and s.hf_token is None
    assert s.embed_device == "cpu"

def test_removed_self_hosting_settings_are_gone():
    fields = Settings.model_fields
    for name in ("vllm_base_url", "vllm_model", "embed_base_url", "cloudflare_tunnel_token"):
        assert name not in fields
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_config.py -v`. Expected: FAIL (missing `llm_base_url`). Also repoint the two existing assertions on `settings.vllm_model` (lines 24 and 37) to `llm_model`, keeping each test's env/`.env` override pattern.
- [ ] **Step 3: Implement.**
  - Update `Settings` as listed in Interfaces, and update `.env.example` to match (remove the VLLM, EMBED_BASE_URL, and CLOUDFLARE lines; add documented `LLM_*`, `HF_TOKEN`, `HF_ARTIFACT_REPO`).
  - Add `serverSelectionTimeoutMS=2000` in `get_client`.
  - Delete the three files listed under Delete. Remove the `embedding` service and `hf_cache` volume from `compose.yaml`. Remove the GPU wording from the Makefile and the `embeddings/run.py` docstring.
- [ ] **Step 4: Switch torch to CPU wheels.** In `pyproject.toml`:

```toml
ml = ["sentence-transformers>=5.7.0", "torch>=2.9"]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

  Run `uv lock`. Verify the lock has no `nvidia-*` packages: `grep -c 'name = "nvidia-' uv.lock` should print `0`.
- [ ] **Step 5: Verify.** `uv sync --frozen --group ml && uv run pytest -m "not integration" -q && uv run ruff check . && uv run mypy`. Expected: all green, with `test_embeddings_service` no longer collected.
- [ ] **Step 6: Commit** `refactor(embeddings): retire GPU service, CPU torch, cloud config`.

---

### Task 2: File-backed TMDB cache + MovieLens download

**Files:**
- Create: `src/movie_recs/ingest/file_cache.py`, `src/movie_recs/ingest/download.py`
- Test: `tests/unit/test_file_cache.py`, `tests/unit/test_download.py`

**Interfaces:**
- Produces: `class FileTmdbCache(root: Path)`, which satisfies the `ingest.tmdb.TmdbCache` protocol (`get(tmdb_id) -> dict | None`, `set(tmdb_id, payload) -> None`). It stores `root / f"{tmdb_id}.json"`.
- Produces: `MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"` and `download_movielens(dest: Path, *, url: str = MOVIELENS_URL) -> Path`. It returns `dest / "ml-latest-small"`, which contains `movies.csv`, `ratings.csv`, `links.csv`, `tags.csv`.

- [ ] **Step 1: Write failing tests.**

```python
def test_file_cache_round_trips(tmp_path):
    c = FileTmdbCache(tmp_path); c.set(862, {"overview": "x"})
    assert c.get(862) == {"overview": "x"}

def test_file_cache_miss_returns_none(tmp_path):
    assert FileTmdbCache(tmp_path).get(1) is None

def test_file_cache_corrupt_file_is_a_miss(tmp_path):
    (tmp_path / "5.json").write_text("{not json")
    assert FileTmdbCache(tmp_path).get(5) is None

def test_download_extracts_csvs(tmp_path, respx_mock):
    # build an in-memory zip with ml-latest-small/{movies,ratings,links,tags}.csv
    respx_mock.get(MOVIELENS_URL).respond(content=zip_bytes)
    out = download_movielens(tmp_path)
    assert {p.name for p in out.iterdir()} >= {"movies.csv", "ratings.csv", "links.csv", "tags.csv"}

def test_download_skips_when_already_present(tmp_path, respx_mock):
    # pre-create the four CSVs; no route registered → any request would error
    assert download_movielens(tmp_path) == tmp_path / "ml-latest-small"
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_file_cache.py tests/unit/test_download.py -v`. Expected: FAIL (import errors).
- [ ] **Step 3: Implement both modules.**
  - Cache writes are atomic: write `*.json.tmp`, then `os.replace`. `get` catches `json.JSONDecodeError`/`OSError` and returns `None`.
  - Download uses `httpx` streaming into `zipfile`. It skips when all four CSVs exist.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ingest): file TMDB cache and MovieLens download`.

---

### Task 3: Bulk writes + cache injection in ingest

**Files:**
- Modify: `src/movie_recs/ingest/run.py`
- Test: `tests/integration/test_ingest_idempotency.py` (existing, must still pass), `tests/unit/test_ingest_run.py` (new)

**Interfaces:**
- Consumes: `FileTmdbCache`, `download_movielens` (Task 2).
- Produces:
  - `run(data_dir: Path, *, sample: bool, fetch_tmdb: bool = True, tmdb_cache: TmdbCache | None = None) -> dict[str, int]` (defaults to `MongoTmdbCache()` when `None`).
  - `BULK_BATCH = 1000`.
  - New CLI flags: `--tmdb-cache-dir PATH` (uses `FileTmdbCache`) and `--download` (calls `download_movielens(data_dir.parent)` first).

- [ ] **Step 1: Write a failing unit test.** `test_upsert_batches_chunks_replace_ones`: a fake collection records `bulk_write` calls. Upserting 2,500 docs gives 3 calls of sizes [1000, 1000, 500], all `ReplaceOne` with `upsert=True`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_ingest_run.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `_upsert_batches(collection, ops: list[ReplaceOne], batch: int = BULK_BATCH) -> None`, using `bulk_write(..., ordered=False)`. Use it for movies, ratings, and users. Wire `tmdb_cache` and the two CLI flags.
- [ ] **Step 4: Verify.** `docker compose up -d mongodb && uv run pytest tests/unit/test_ingest_run.py tests/integration/test_ingest_idempotency.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `perf(ingest): bulk upserts, injectable TMDB cache, --download`.

---

### Task 4: Evaluation reframing (leave-one-out primary) + JSON report

**Files:**
- Modify: `src/movie_recs/recsys/evaluate.py`
- Test: `tests/unit/test_recsys_evaluate.py`

**Interfaces:**
- Produces:
  - `@dataclass class EvaluationReport: primary: list[MetricRow]; secondary: list[MetricRow]; primary_label: str; secondary_label: str`
  - `def report_to_dict(report: EvaluationReport) -> dict[str, Any]`
  - `run() -> EvaluationReport`. It **no longer saves an artifact**; that moves to Task 5.
  - CLI `--report-json PATH`.
  - Label constants: `PRIMARY_LABEL = "Primary — per-user leave-one-out"`, `SECONDARY_LABEL = "Secondary — global temporal cutoff (small-n)"`.

- [ ] **Step 1: Write failing tests.**
  - `test_report_to_dict_labels_loo_primary`: `report_to_dict(...)["primary"]["label"] == PRIMARY_LABEL`, and each row serializes to `{model, metric: value...}` with float values.
  - `test_report_to_dict_is_json_serializable`: `json.dumps(report_to_dict(report))` doesn't raise.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Reorder `run()`: leave-one-out first, printed under `PRIMARY_LABEL`, then global under `SECONDARY_LABEL` with the evaluable-user count in the label line. Drop the artifact save and the `--artifact-path` flag.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_recsys_evaluate.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(recsys): leave-one-out as primary split, JSON report`.

---

### Task 5: Serving artifact trained on all data

**Files:**
- Create: `src/movie_recs/recsys/train.py`
- Modify: `src/movie_recs/recsys/als.py` (`RecsysArtifact` gains `popularity: list[int] = field(default_factory=list)`)
- Test: `tests/unit/test_recsys_train.py`

**Interfaces:**
- Consumes: `build_matrix`, `train_als`, `train_item_item`, `item_item_neighbors`, `IdMap`, `LIKE_THRESHOLD`, `CONFIDENCE_ALPHA` (`recsys/als.py`); `load_ratings` (`recsys/evaluate.py`).
- Produces: `train_serving_artifact(ratings: pd.DataFrame) -> RecsysArtifact`. It trains on **all** rows. `train_cutoff_ts` is the max timestamp. `popularity` holds liked movieIds sorted by like count, descending.
- CLI: `python -m movie_recs.recsys.train --out artifacts/recsys/model.pkl`.

- [ ] **Step 1: Write failing tests** on a synthetic ratings frame (reuse the block-clustered fixture pattern from `test_recsys_als.py`):

```python
def test_every_liked_movie_gets_a_factor():
    art = train_serving_artifact(ratings)
    liked = set(ratings.loc[ratings.rating >= LIKE_THRESHOLD, "movieId"])
    assert liked <= set(art.item_map.idx_to_id)
    assert art.item_factors.shape[0] == len(art.item_map.idx_to_id)

def test_popularity_is_sorted_by_like_count_desc():
    art = train_serving_artifact(ratings)
    counts = ratings[ratings.rating >= 4].movieId.value_counts()
    assert art.popularity[:3] == list(counts.index[:3])

def test_cutoff_is_max_timestamp():
    assert train_serving_artifact(ratings).train_cutoff_ts == int(ratings.timestamp.max())
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `train_serving_artifact` and the CLI (`save_artifact` to `--out`).
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_recsys_train.py tests/unit/test_recsys_als.py -v`. Expected: PASS. The old pickles without `popularity` still load, via the default.
- [ ] **Step 5: Commit** `feat(recsys): serving artifact trained on all ratings`.

---

### Task 6: Artifact store (HF Hub) + manifest + catalog version

**Files:**
- Create: `src/movie_recs/recsys/artifact_store.py`, `src/movie_recs/db/meta.py`
- Modify: `pyproject.toml` (add `huggingface_hub` to core `dependencies`), `uv.lock`, `src/movie_recs/db/collections.py` (`meta_collection()`)
- Test: `tests/unit/test_artifact_store.py`, `tests/integration/test_meta.py`

**Interfaces:**
- Produces:
  - `@dataclass class Manifest: git_sha: str; trained_at: str; catalog_version: str; n_items_with_factors: int; metrics: dict[str, Any]`, with `to_json() -> str` and `@classmethod from_json(s) -> Manifest`.
  - `MODEL_FILE = "model.pkl"`, `MANIFEST_FILE = "manifest.json"`.
  - `upload_artifact(model_path: Path, manifest: Manifest, *, repo_id: str, token: str | None, api: HfApi | None = None) -> None`, which calls `api.upload_file` twice, with `repo_type="model"`.
  - `download_artifact(*, repo_id: str, token: str | None, dest: Path, download: Callable[..., str] = hf_hub_download) -> tuple[Path, Manifest]`.
  - `db/meta.py`: `set_catalog_version(version: str) -> None` and `get_catalog_version() -> str | None` (doc `_id="catalog"` in `meta`).
- CLI `python -m movie_recs.recsys.artifact_store upload --model PATH --report-json PATH --catalog-version V`. It reads `GITHUB_SHA`, `HF_TOKEN`, and `HF_ARTIFACT_REPO` from the env. Catalog version format: `f"{utc_now:%Y%m%dT%H%M%SZ}-{git_sha[:7]}"`, generated in the workflow and passed in.

- [ ] **Step 1: Write failing tests.**
  - `test_manifest_round_trips`.
  - `test_upload_calls_hub_with_both_files`: a fake `api` records `(path_in_repo, repo_id, repo_type)` → `{("model.pkl", R, "model"), ("manifest.json", R, "model")}`.
  - `test_download_returns_parsed_manifest`: a fake `download` copies fixture files.
  - `test_manifest_contains_no_secrets`: with `MONGODB_URI=mongodb+srv://u:p@x` and `HF_TOKEN=hf_abc` set, `manifest.to_json()` contains neither `"mongodb"` nor `"hf_abc"`.
  - Integration `test_catalog_version_round_trips`, using the `mongo_db` fixture.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Use `uv add huggingface_hub`.
- [ ] **Step 4: Run** the unit tests, plus `uv run pytest -m integration tests/integration/test_meta.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(recsys): HF Hub artifact store, manifest, catalog version`.

---

### Task 7: API `/health` + Space packaging

**Files:**
- Create: `src/movie_recs/api/__init__.py`, `src/movie_recs/api/main.py`, `src/movie_recs/api/schemas.py`, `deploy/api/Dockerfile`, `deploy/api/README.md`, `scripts/hf_space.py`
- Test: `tests/unit/test_api_health.py`, `tests/unit/test_hf_space.py`

**Interfaces:**
- Consumes: `download_artifact`, `Manifest` (Task 6); `get_db` (`db/client.py`).
- Produces:
  - `api.main.create_app(*, load_artifact: bool = True) -> FastAPI`. The lifespan downloads the artifact into `app.state.manifest` and `app.state.artifact_path`, and on any exception logs a warning and sets them to `None`.
  - `api.main.app = create_app()`.
  - Dependency `get_mongo_ping() -> Callable[[], bool]`, overridable in tests.
  - `schemas.HealthResponse(status: Literal["ok","degraded"], mongo: Literal["ok","down"], artifact: Manifest | None, version: str)`.
- Produces: `scripts/hf_space.py` with subcommands `deploy --space ID --kind api` and `restart --space ID`. The pure helper `stage_space(kind: str, repo_root: Path, out: Path) -> Path` copies `deploy/<kind>/Dockerfile` → `out/Dockerfile`, `deploy/<kind>/README.md` → `out/README.md`, plus `pyproject.toml`, `uv.lock`, `.python-version`, and `src/`. Deploy uses `HfApi().upload_folder(repo_type="space")`; restart uses `HfApi().restart_space`.

- [ ] **Step 1: Write failing tests.**

```python
# `client_with(mongo: bool, manifest: Manifest | None) -> TestClient` is a fixture factory:
# create_app(load_artifact=False), set app.state.manifest, override get_mongo_ping.

def test_health_ok(client_with):
    r = client_with(mongo=True, manifest=M).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["mongo"] == "ok"
    assert r.json()["artifact"]["git_sha"] == M.git_sha

def test_health_degraded_when_artifact_missing(client_with):
    assert client_with(mongo=True, manifest=None).get("/health").json()["status"] == "degraded"

def test_health_reports_mongo_down_without_500(client_with):
    r = client_with(mongo=False, manifest=M).get("/health")
    assert r.status_code == 200 and r.json()["mongo"] == "down"

def test_startup_survives_artifact_download_failure(monkeypatch):
    # patch download_artifact to raise; TestClient(create_app()) starts; /health → degraded

def test_stage_space_layout(tmp_path):
    out = stage_space("api", REPO_ROOT, tmp_path)
    assert {"Dockerfile", "README.md", "pyproject.toml", "uv.lock", "src"} <= {p.name for p in out.iterdir()}
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** the app, schemas, and script.
  - The Mongo ping is `get_db().command("ping")`, wrapped so any `PyMongoError` → `False`.
  - `deploy/api/README.md` is HF Space front matter: `sdk: docker`, `app_port: 7860`, `title: movie-recs-api`.
  - `deploy/api/Dockerfile`: `python:3.12-slim` + `uv` copied from `ghcr.io/astral-sh/uv:0.11.13`. It runs `uv sync --frozen --no-default-groups --group serving --group ml` (CPU torch via the Task 1 index), creates a non-root user with UID 1000 (an HF Spaces requirement), sets `HF_HOME=/tmp/hf`, and starts with `CMD ["uv","run","--no-sync","uvicorn","movie_recs.api.main:app","--host","0.0.0.0","--port","7860"]`.
- [ ] **Step 4: Verify.** `uv run pytest tests/unit/test_api_health.py tests/unit/test_hf_space.py -v` should PASS. Then run `docker build -f deploy/api/Dockerfile .` in any Docker-capable dev environment, or skip it and rely on the Space build in Task 9. Expected: the image builds with no `nvidia` packages.
- [ ] **Step 5: Commit** `feat(api): /health endpoint and HF Space packaging`.

---

### Task 8: Workflows — CI integration, pipeline, deploy, keep-alive

**Files:**
- Modify: `.github/workflows/ci.yaml`, `pyproject.toml` (marker text)
- Create: `.github/workflows/pipeline.yaml`, `.github/workflows/deploy.yaml`, `.github/workflows/keepalive.yaml`
- Test: `tests/unit/test_workflows.py`

**Interfaces:**
- Consumes: the CLIs from Tasks 3–7, and the secrets and variables from Task 0.

- [ ] **Step 1: Write failing tests** that parse the YAML with `yaml.safe_load` (add `pyyaml` to the dev group). Note that PyYAML parses the `on:` key as the boolean `True`.
  - `ci.yaml` has a `services.mongodb.image == "mongodb/mongodb-atlas-local:8.0"` and a step running `pytest -m integration`.
  - `pipeline.yaml` is triggered only by `workflow_dispatch`, and its step commands appear in this order: `ingest`, `recsys.evaluate`, `recsys.train`, `movie_recs.embeddings`, `artifact_store upload`, `hf_space.py restart`.
  - `deploy.yaml` triggers on `workflow_run` of `CI` (types `completed`, branches `[main]`), its job is gated on `github.event.workflow_run.conclusion == 'success'`, and it runs `hf_space.py deploy --space`.
  - `keepalive.yaml` has `schedule[0].cron == "17 */12 * * *"`, `permissions.actions == "write"`, and a step calling `actions/workflows/keepalive.yaml/enable`.
  - No workflow file contains the literal strings `mongodb+srv://` or `hf_`, and secrets appear only as `${{ secrets.* }}`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **ci.yaml.** Keep the existing steps. Add the `mongodb` service (port `27017:27017`) and the steps `uv sync --frozen --group ml`, `actions/cache` for `~/.cache/huggingface`, and `uv run pytest -m integration`. Update the marker description in `pyproject.toml` to "requires Mongo (atlas-local)".
  - **pipeline.yaml.** One job, `timeout-minutes: 120`, with `actions/cache` keyed `tmdb-cache-v1` on `data/tmdb_cache/`. The `env` block maps the `MONGODB_URI`, `TMDB_API_KEY`, and `HF_TOKEN` secrets and the `HF_ARTIFACT_REPO`/`HF_API_SPACE` variables. Steps:
    1. `uv sync --frozen --group ml`
    2. `uv run python -m movie_recs.ingest --download --tmdb-cache-dir data/tmdb_cache`
    3. `uv run python -m movie_recs.recsys.evaluate --report-json artifacts/report.json`
    4. `uv run python -m movie_recs.recsys.train --out artifacts/recsys/model.pkl`
    5. `uv run python -m movie_recs.embeddings`
    6. compute `CATALOG_VERSION` and call `set_catalog_version` via `uv run python -c`
    7. `uv run python -m movie_recs.recsys.artifact_store upload --model artifacts/recsys/model.pkl --report-json artifacts/report.json --catalog-version "$CATALOG_VERSION"`
    8. `uv run python scripts/hf_space.py restart --space "$HF_API_SPACE"`
  - **deploy.yaml.** Triggered by `workflow_run` on CI completion (branch `main`, gated on `conclusion == 'success'`), so a red build never deploys; check out `github.event.workflow_run.head_sha`, then run `uv run python scripts/hf_space.py deploy --space "$HF_API_SPACE" --kind api`.
  - **keepalive.yaml.** Runs on the cron above plus `workflow_dispatch`. Steps: `curl -fsS --retry 5 --retry-delay 30 --max-time 120 "$API_HEALTH_URL"` (the first hit may wake the Space), then `gh api -X PUT repos/${{ github.repository }}/actions/workflows/keepalive.yaml/enable` with `GH_TOKEN: ${{ github.token }}`.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_workflows.py -v`. Expected: PASS. Push the branch and confirm that CI is green **with the integration step running**.
- [ ] **Step 5: Commit** `ci: mongo integration in CI; pipeline, deploy, keepalive workflows`.

---

### Task 9: Go live + README draft + record findings

**Files:**
- Modify: `README.md`, `plan.md` (Session 5 status + Decision Log), `Makefile` (`pipeline-local` is **not** added: there are no local runs)

- [ ] **Step 1: Merge the PR** to `main` and confirm `deploy.yaml` pushes the Space and the Space build succeeds.
- [ ] **Step 2: Run `pipeline.yaml`** from the Actions tab. Expected log lines:
  - ingest `{'movies': 9742, 'ratings': 100836, 'users': 610}`
  - embed `with_embedding: 9742`
  - `text_vec` READY
  - upload succeeded
  - Space restarted
- [ ] **Step 3: Verify live.** `curl -fsS $API_HEALTH_URL` should return `status: "ok"`, `mongo: "ok"`, and an `artifact` whose `catalog_version` equals the pipeline's value and whose `n_items_with_factors` equals the number of distinct movies with ≥1 rating ≥ 4.0. Record that number.
- [ ] **Step 4: Verify idempotency.** Re-run `pipeline.yaml`. The ingest counts should be identical, embed should report `embedded: 0`, and the run should finish faster thanks to the TMDB cache hit.
- [ ] **Step 5: Run keep-alive** via `workflow_dispatch`. Expected: green, and the enable call returns 204.
- [ ] **Step 6: README draft.** Include the live `/health` link, a one-paragraph description, the architecture mermaid from `plan.md`, a **"Skills demonstrated"** list (Python, FastAPI, collaborative filtering (ALS/item-item), offline evaluation, vector search (MongoDB `$vectorSearch`), embeddings, CLIP (S6), LLM integration (S9), Streamlit (S10), Docker, GitHub Actions CI/CD, data pipelines, pytest/mypy/ruff), and the runbook (Task 0 + "run the pipeline"). No employer or resume details.
- [ ] **Step 7: Update `plan.md`.**
  - Mark the Session 5 DoD boxes and add a **Status** line with the measured numbers (pipeline duration, factor coverage, image size).
  - Add Decision Log entries for anything that deviated from this plan.
  - Record the free-tier policies actually observed (Space sleep, Atlas M0 search-index limit and idle pause, keep-alive enable response) under Risk #1.
- [ ] **Step 8: Commit** `docs: session 5 live; README draft; record free-tier findings`.
