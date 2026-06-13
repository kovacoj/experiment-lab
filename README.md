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
```
