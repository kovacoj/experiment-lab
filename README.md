# Signal Foundry Labs Template

Labs-first Python template for Signal Foundry.

## Scope

- Shared lab schemas and interfaces
- Shared text signal engine
- Scenario lab registry
- Deterministic demo fixtures
- Runner, critic, and ranking layer
- Plain Python tests

## Backend Flow

```text
Internal business data
+
External raw text/web data
        ↓
Source adapters
        ↓
TextDocumentSignal[]
        ↓
Cleaning + dedup + entity linking
        ↓
ExtractedTextSignal[]
        ↓
Scenario-specific labs
        ↓
Selected findings + evidence + actions
```

Both demo stories use the same text engine. Only the scenario taxonomy and lab registry change.

## Local Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Run The Demo

From this directory:

```bash
python -m app.labs.runner --scenario reputation_monitor
python -m app.labs.runner --scenario supply_chain_risk
```

## Local Sentiment Model (XLM-RoBERTa)

Use the local sentiment model so teammates can run sentiment inference without Hugging Face MCP at runtime.

Model used:

- `cardiffnlp/twitter-xlm-roberta-base-sentiment`

### 1) Install dependencies

From `experiment-lab/`:

```bash
uv sync
```

### 2) Download model files locally

```bash
uv run python scripts/download_sentiment_model.py
```

This downloads model assets to:

- `.models/sentiment/cardiffnlp-twitter-xlm-roberta-base-sentiment`

`.models/` is gitignored and must not be committed.

### 3) Verify inference

```bash
uv run python scripts/smoke_test_sentiment.py
```

### 4) Verify offline inference

```bash
HF_HUB_OFFLINE=1 uv run python scripts/smoke_test_sentiment.py
```

If this passes, sentiment inference runs from local model files and does not require network access.

### Optional env defaults

If you manage env vars locally, these defaults are expected:

```bash
SENTIMENT_MODE=local
SENTIMENT_MODEL_ID=cardiffnlp/twitter-xlm-roberta-base-sentiment
SENTIMENT_MODEL_DIR=.models/sentiment/cardiffnlp-twitter-xlm-roberta-base-sentiment
SENTIMENT_DEVICE=cpu
SENTIMENT_BATCH_SIZE=4
SENTIMENT_MAX_CHARS=800
```

## Experiment Lab Skill

Use the stable local skill wrapper when an agent or developer needs structured machine output instead of importing internal modules directly.

Core commands:

```bash
uv run python -m app.agent_skills.experiment_lab_cli list-scenarios
uv run python -m app.agent_skills.experiment_lab_cli list-labs --scenario reputation_monitor
uv run python -m app.agent_skills.experiment_lab_cli run-analysis --scenario reputation_monitor --format json
uv run python -m app.agent_skills.experiment_lab_cli refresh-compare --scenario supply_chain_risk --format json
```

External integrations such as Apify, Hugging Face, and n8n are intentionally out of scope for this skill and should be handled by MCP or other orchestration layers separately.

List the available labs for one scenario:

```bash
python -m app.labs.runner --scenario reputation_monitor --list-labs
```

Run just one lab manually as a developer:

```bash
python -m app.labs.runner --scenario reputation_monitor --lab-id location_sentiment
```

Get JSON output instead of text:

```bash
python -m app.labs.runner --scenario reputation_monitor --format json
```

## Run Tests

```bash
python -m unittest discover -s tests
```

## Docker

Build the image:

```bash
docker build -t experiment-lab .
```

Or with Compose:

```bash
docker compose build
```

Run the default scenario:

```bash
docker run --rm experiment-lab
```

Run a specific scenario:

```bash
docker run --rm experiment-lab python -m app.labs.runner --scenario supply_chain_risk
```

Run tests in Docker:

```bash
docker compose run --rm experiment-lab-tests
```

## Make Targets

```bash
make test
make run
make run-supply-chain
make docker-build
make docker-run
make docker-run-supply-chain
make docker-test
make download-sentiment-model
make smoke-sentiment
make smoke-sentiment-offline
```
