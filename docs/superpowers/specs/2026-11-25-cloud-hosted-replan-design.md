# Movie Recs — Cloud-Hosted Re-plan (Design)

**Date:** 2026-11-25
**Status:** draft, pending approval
**Replaces:** the hosting, serving, and remaining-session parts of `plan.md` (Sessions 5–10, Architecture, Local Dev Setup, Non-Goals, Risks #1/7/8/14/15). Sessions 1–4 stay as-is and keep their history.

## Intent and why

This is a portfolio project meant to show off applied-AI / ML-engineering skills on a resume: a Netflix-style movie recommender with collaborative filtering, vector search, real-time session personalization, and an LLM layer.

The current plan serves everything from the owner's home PC (GPU, vLLM, Cloudflare Tunnel). That makes the live demo depend on the PC being on, so a recruiter's click can land on a dead link. **The re-plan makes the demo always live, at $0/month, with nothing running on the owner's machine, ever.** That covers serving, batch jobs, and model training. Local Docker remains only as a development and test convenience.

The resume story moves from "self-hosted GPU inference" to "a cloud-deployed ML system with an automated data/training pipeline, CI/CD, and LLM integration." It keeps the same product.

## Goals

- A public URL that works whenever someone clicks it. A wake-up delay of up to a couple of minutes after long idle is acceptable but should be rare.
- $0/month: only free tiers.
- No dependency on the owner's machine for serving, batch jobs, training, or deployment.
- Keep the product features: two-stage retrieve → rerank, text + CLIP-poster vector search, anonymous sessions with real-time 👍/👎 re-personalization (ALS fold-in), grounded LLM explanations, natural-language search, and a Streamlit UI.
- Keep a visible FastAPI ↔ Streamlit service boundary with API-key auth.
- Show engineering rigor: tests, CI that includes Mongo integration tests, automated deploys, and an automated data pipeline.

## Non-goals

- No self-hosted or GPU mode. vLLM, CUDA serving, and Podman are dropped from the project.
- No paid services and no cards on file.
- No user accounts. Sessions stay anonymous.
- No streaming data pipeline. Catalog and training work is batch.
- No bigger dataset: stays on `ml-latest-small`.
- No fine-tuning.

## Anti-goals (failure even if it "works")

- The demo link is dead or errors when nobody has touched it for a while.
- A free-tier limit (LLM rate limit, Atlas storage) turns into a 500 or a broken page instead of graceful degradation.
- Any step needs the owner to run something on their own PC.
- Secrets (TMDB key, Groq key, Atlas URI, API key, HF token) end up in the repo or in the browser.

## Constraints

- **MongoDB Atlas M0 (free):** 512 MB storage, `$vectorSearch` supported, with a cap on the number of search indexes (only two are needed: `text_vec`, `poster_vec`). Network access must allow dynamic IPs (HF Spaces, GitHub runners), so it uses `0.0.0.0/0` plus a strong, scoped database user.
- **Hugging Face Spaces (free CPU Docker):** about 2 vCPU / 16 GB RAM, sleeps after about 48 h without traffic, secrets set in Space settings.
- **Groq free tier:** OpenAI-compatible API with per-minute and per-day request/token limits.
- **GitHub Actions:** free CPU runners (about 7 GB RAM, no GPU), 6 h job limit.
- Python 3.12, `uv`, `ruff`, strict `mypy`, `pytest`, as established in Sessions 1–4.

## Approaches considered

| | Approach | Trade-off | Outcome |
|---|---|---|---|
| A | **Two HF Spaces** (FastAPI Space + Streamlit Space) + Atlas M0 + Groq | Keeps a real service boundary and API key; 16 GB RAM fits the torch CPU embedder; ~48 h sleep, mitigated by a keep-alive ping | **Chosen** |
| B | Streamlit Community Cloud + Render free + Atlas M0 | Each piece on its "native" host; but Render has 512 MB RAM (forces ONNX instead of torch), sleeps after 15 min, and two cold starts can stack | Rejected |
| C | One Space running both apps | Simplest to run; loses the visible service boundary and API-key gate | Rejected |

LLM serving was also considered:
- **Free Groq API: chosen.** It's the simplest option, and "LLM usage" is the point, not self-hosting.
- vLLM on serverless GPU (Modal): rejected as unnecessary.
- Keeping a local vLLM mode: rejected by the owner.

On data, the options were keeping `ml-latest-small` and fixing the evaluation framing, or moving to `ml-32m` filtered to the top movies. **Keeping ml-small was chosen:** the resume value is in the system design, and staying avoids reworking Sessions 2–3.

## Decided design

### Architecture

```
Browser ──HTTPS──> [HF Space: Streamlit]  (holds API_KEY server-side)
                         │ HTTPS + X-API-Key
                         ▼
                   [HF Space: FastAPI]
                    ├─ ALS item factors + item-item neighbors (loaded at startup from HF Hub)
                    ├─ bge-small text embedder on CPU (query-time only)
                    ├─ fold-in, rerank (numpy)
                    ├──> MongoDB Atlas M0  (movies + vectors, sessions, ratings)
                    └──> Groq API  (explanations, NL-search parsing; graceful fallback)

GitHub Actions
  ci.yaml        : ruff, mypy, unit tests, Mongo integration tests (atlas-local service container)
  pipeline.yaml  : manual/scheduled — ingest → evaluate → train-on-all → embed text + posters → build indexes → upload artifact to HF Hub
  deploy.yaml    : on merge to main — push both apps to their Spaces
  keepalive.yaml : scheduled ping of both Spaces' /health
```

### Decided changes to existing work

1. **Serve a model trained on all data.** Evaluation still trains on the split. The pipeline then retrains ALS and item-item on the full dataset and that model is the one served. This makes "no CF factor" rare: it only applies to movies nobody liked. Fall-through to content/visual still exists, but it's no longer the common case (Risk #16 shrinks).
2. **Evaluation framing.** The per-user leave-one-out split (573 users) becomes the **primary** reported result. The global temporal split is reported secondarily, labeled small-n (about 27 users). The "good" bar is: ALS beats popularity and random on NDCG@10/Recall@10. The 0.10–0.35 NDCG target is removed.
3. **The embedding microservice is retired.**
   - `TextEmbedder` (CPU) is used in-process by the API for query embeddings and by the pipeline for catalog embeddings.
   - `docker/embedding.Dockerfile`, the `embedding` compose service, and `EMBED_BASE_URL` go away. `embeddings/service.py` and its tests are deleted.
   - The API image installs torch from the **CPU-only** wheel index to keep it small.
4. **CLIP poster embeddings run on CPU in the pipeline.** They use open_clip ViT-B-32 (512-dim). The ~124 movies with no poster are skipped. There is no online image embedding.
5. **CI runs Mongo integration tests.** An atlas-local service container is added to `ci.yaml`. The `integration` marker splits so nothing GPU-only remains, and since there is no GPU anymore, every integration test is Mongo-only.
6. **GitHub only.** All GitLab references and `.gitlab-ci.yml` are dropped.
7. **Removed from the plan:** vLLM, `compose.vllm.yaml`, GPU reservations, the Cloudflare Tunnel (`deploy/cloudflared`), the home-PC runbook, the reference-machine table, and Risks #1/7/8/14/15.
8. **`compose.yaml` keeps only `mongodb` (atlas-local)** for local dev and tests.

### LLM layer

- **Client:** an OpenAI-compatible client pointed at Groq via `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`, so a provider swap is config-only. It has timeouts, one retry, and on a 429 or timeout it returns `None`.
- **Explanations** are grounded only in the item's real metadata and the session's upvoted titles. `/recommend?explain=true` fills `reason`, or leaves it `None` on fallback.
- **Explanation caching:** explanations are cached in Mongo keyed by (movieId, sorted upvoted-seed set hash). This saves free-tier quota and keeps explanations stable.
- **NL search:** the LLM parses a query into a structured intent (seed title, genre/mood modifiers) using JSON-mode / structured output, validated with pydantic. On fallback, the raw query is embedded and plain vector search runs.
- **Rate limits:** the API applies its own per-session caps on LLM-backed calls (`/search`, explain) so one visitor can't drain the daily quota.

### Storage budget (Atlas 512 MB)

Movies with two vectors come to roughly 9.7k × (384+512) doubles, about 70 MB plus metadata. Ratings are about 100k small docs. The `sessions` and explanation caches carry a TTL index (e.g. 14 days) so they can't grow without bound. The `tmdb_cache` collection does **not** go to Atlas: the pipeline keeps it as a GitHub Actions cache or artifact. Raw payloads are not needed online.

### Model artifact

The pipeline pickles/npz-saves the artifact (item factors, regularization, id maps, item-item neighbors, popularity prior) and uploads it to a free HF Hub model repo. The API Space downloads it at startup with `HF_TOKEN` if the repo is private. A small JSON manifest (train timestamp, git SHA, metrics) is stored with the artifact, and `/health` reports it.

### Session order (proposed rewrite of plan.md Sessions 5+)

The deploy path is built **early** as a walking skeleton, so every later session ships straight to the live URL.

- **S5: Cloud foundation.** Set up Atlas M0, the pipeline workflow (ingest → evaluate → train-on-all → text embed → index → HF Hub upload), the embedding-service retirement, Mongo integration tests in CI, the evaluation re-framing, and a minimal FastAPI Space (`/health`) with deploy and keep-alive workflows.
- **S6: CLIP posters.** Add CPU poster embedding to the pipeline, the `poster_vec` index, and visual + hybrid retrieval.
- **S7: Recommendation API.** Two-stage retrieve → rerank, the endpoints, API key + CORS, and cold-start routing.
- **S8: Real-time sessions.** Fold-in, `/session`, `/feedback`, session-aware recs, downvote suppression, and the session TTL.
- **S9: LLM layer.** Groq client, grounded explanations + cache, NL search, fallbacks, and per-session caps.
- **S10: Streamlit UI.** Netflix rows, 👍/👎 loop, search bar, and the second Space plus its deploy.
- **S11: Polish.** End-to-end smoke test against the live URLs (a post-deploy workflow), README with architecture and screenshots, and the abuse/rate-limit review.

Each session keeps the existing conventions: its own branch, mergeable to `main`, real tests, a definition of done, and a demo. The demo is now on the live URL wherever possible.

### Calls settled after the builder check (Claude's calls, owner to confirm)

- **The pipeline does not train on live votes.** It retrains only on `ml-latest-small` and runs on manual `workflow_dispatch`, with no schedule. "Automated" means reproducible and hands-off, not continuous learning. Feeding votes back into training is a possible later stretch, out of scope here.
- **Surviving GitHub's 60-day rule.** The repo is public, so Actions minutes are unlimited, but GitHub disables scheduled workflows after 60 days with no repo activity.
  - `keepalive.yaml` re-enables itself through the GitHub API (the `keepalive-workflow` API mode, so no heartbeat commits).
  - Its pings hit API `/health`, which also runs a trivial Mongo query, so Atlas never looks idle long enough to pause.
  - No outside uptime service is needed.
- **Pipeline runs rewrite in place.** Upserts are idempotent and a run is rare and manual, so a few minutes of slightly stale or mixed results during a run is acceptable.
  - The pipeline restarts the API Space at the end (HF API) so it loads the new artifact.
  - The API checks that the artifact's `catalog_version` matches the one stored in Mongo and logs a warning on mismatch. It does not error.
  - No staging-collection swap, since doubling the vector collection doesn't fit comfortably in 512 MB.
- **LLM quota protection has two layers:** per-session caps, plus a **global daily LLM-call budget** counter in Mongo, set below Groq's daily limit. Once either is exhausted, responses fall back gracefully (`reason=None`, plain vector search). Neither layer depends on IP tracking.

## Left to the builder

- Exact Groq model (a current small Llama-family instruct model, picked at S9 against the then-current free limits).
- HF Hub repo names and Space names; whether the artifact repo is public or private.
- Keep-alive cadence, which must be inside the ~48 h sleep window (e.g. every 12 h).
- How CPU-only torch is pinned in `uv` (a separate index scoped to the API image's group).
- Pipeline trigger: manual `workflow_dispatch` at minimum; whether to also schedule it.
- Exact TTL durations and per-session cap numbers.
- Confirm current Atlas M0 search-index limits and Spaces sleep policy when implementing S5; record what was found in the Decision Log.
