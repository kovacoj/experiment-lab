# AGENTS.md

## Workspace Shape
- This repo is a portable labs-first backend template for Signal Foundry.
- Keep deterministic backend code under `app/labs/` and `app/text_engine/`, demo fixtures under `app/demo_data/`, and verification in `tests/`.
- `app/labs/schemas.py` defines the typed contracts, `app/labs/registry.py` defines the scenario registries, `app/labs/runner.py` is the main local entrypoint, and `app/labs/critic.py` is the deterministic validation layer.
- `app/text_engine/` owns raw-text normalization, cleaning, deduplication, entity linking, rule extraction, model-adapter stubs, and shared aggregation helpers.
- Scenario-specific labs live under `app/labs/reputation/` and `app/labs/supply_chain/`.
- `README.md` is a short operator guide for running the demo CLI and tests, not a product spec.

## Environment
- This repo is self-contained and should be installable on another machine with `pip install -e .` or by building the local `Dockerfile`.
- The project requires Python `>=3.13`.
- Prefer Polars for analytical processing and Pydantic for contracts. Do not introduce pandas unless the human explicitly asks.

## Working Conventions
- Treat the labs layer as deterministic, independently testable backend code. The labs should be callable from plain Python tests without frontend, n8n, database, auth, or live Apify dependencies.
- Route raw external text through the shared text engine once, then let labs consume `ExtractedTextSignal`-style records. Do not let each lab parse raw text separately.
- Prefer Polars lazy execution. Push filters and projections early, avoid row-wise Python loops, and keep intermediate computation columnar.
- Use `app/demo_data/*_external_raw.ndjson` as the canonical raw external demo fixtures and `app/demo_data/*_internal.ndjson` as the canonical internal fixtures. Keep them small, deterministic, and aligned with the expected scenario outputs.
- Keep scenario behavior explicit in the lab modules and registries rather than hiding it behind generic abstractions.
- Keep orchestration thin. Heavy analytical work belongs inside the labs, not in future LLM/orchestrator code.
- When adding or changing a lab, update or add tests in `tests/` in the same change.
- There is no repo-configured lint, formatter, task runner, or CI workflow for this template. Do not claim those checks ran unless you add them.

## Agent Rules
- Default autonomous changes to `app/labs/`, `app/text_engine/`, `app/demo_data/`, and `tests/`.
- Prefer the stable wrapper `python -m app.agent_skills.experiment_lab_cli ...` when acting as an agent consumer of the lab engine. Do not import random lab modules directly unless you are editing the implementation.
- Keep the current scope narrow: labs package, fixtures, runner, critic, and tests. Do not add frontend, n8n workflows, live Apify connectors, authentication, database code, or arbitrary orchestration layers unless the human explicitly asks.
- Preserve the shared concepts in the lab contracts: lab identity, scenario identity, hypothesis, status, score, confidence, summary, evidence, recommended actions, limitations, and monitoring rules.
- Keep the two current scenarios wired through `LAB_REGISTRY`: `reputation_monitor` and `supply_chain_risk`.
- Keep the text engine shared across both scenarios. Scenario differences should live in taxonomy, source adaptation, extraction rules, and lab bundles, not in duplicated pipelines.
- Prefer adding scenario-specific logic in the corresponding lab file over introducing a new cross-cutting abstraction unless at least two labs clearly need the same behavior.
- Do not silently weaken privacy and safety constraints. Reputation outputs should stay shift-level rather than person-blaming, and cached competitor claims should remain framed as cached demo findings.

## Git Gotchas
- The local `.gitignore` excludes Python cache files, virtualenvs, and common OS junk.
