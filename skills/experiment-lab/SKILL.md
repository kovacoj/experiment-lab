# Signal Foundry Experiment Lab Skill

## Purpose

Use this skill when the agent needs to run the local Signal Foundry experiment-lab engine.

The experiment-lab engine is responsible for deterministic business-signal analysis over cached/demo data. It runs scenario-specific research labs, validates outputs, applies selection/critic logic, builds ensemble findings, and compiles decision-card payloads.

This skill is local and deterministic. It should not call external MCP servers, Hugging Face, Apify, n8n, web search, or live APIs.

## When to use this skill

Use this skill when the user asks to:

* run a Signal Foundry scenario;
* inspect available research labs;
* run the reputation monitor demo;
* run the supply-chain risk demo;
* generate lab reports;
* generate decision cards;
* simulate a refresh comparison;
* debug lab selection, hidden labs, discarded labs, or ensemble findings.

## Available scenarios

Supported scenario IDs:

* `reputation_monitor`
* `supply_chain_risk`

## Core commands

Run commands from the repository root.

### List scenarios

```bash
uv run python -m app.agent_skills.experiment_lab_cli list-scenarios
```

### List labs for a scenario

```bash
uv run python -m app.agent_skills.experiment_lab_cli list-labs --scenario reputation_monitor
```

```bash
uv run python -m app.agent_skills.experiment_lab_cli list-labs --scenario supply_chain_risk
```

### Run full scenario analysis as JSON

```bash
uv run python -m app.agent_skills.experiment_lab_cli run-analysis --scenario reputation_monitor --format json
```

```bash
uv run python -m app.agent_skills.experiment_lab_cli run-analysis --scenario supply_chain_risk --format json
```

### Run full scenario analysis as readable text

```bash
uv run python -m app.agent_skills.experiment_lab_cli run-analysis --scenario reputation_monitor --format text
```

```bash
uv run python -m app.agent_skills.experiment_lab_cli run-analysis --scenario supply_chain_risk --format text
```

### Simulate refresh comparison

```bash
uv run python -m app.agent_skills.experiment_lab_cli refresh-compare --scenario reputation_monitor --format json
```

```bash
uv run python -m app.agent_skills.experiment_lab_cli refresh-compare --scenario supply_chain_risk --format json
```

## Expected behavior

The skill should return typed, structured output.

A successful JSON response should include:

* `scenario`
* `status`
* `analysis_contract`, if available
* `lab_run_report`
* `decision_cards`
* `alert_candidates`
* `metadata`

A refresh comparison response should include:

* `scenario`
* `status`
* `prediction_changed`
* `old_headline`
* `new_headline`
* `old_confidence`
* `new_confidence`
* `alert`, if changed

## Output categories

Lab results may be classified as:

* `selected`: strong, actionable, sufficiently supported;
* `warning`: relevant but limited by confidence or data quality;
* `hidden`: valid but weak or not immediately actionable;
* `discarded`: unsafe, irrelevant, unsupported, or invalid;
* `failed`: crashed or invalid execution.

Hidden and discarded labs must remain visible in machine output for transparency and debugging.

## Safety and quality rules

The skill must not:

* call Apify;
* call Hugging Face;
* call n8n;
* browse the web;
* install packages;
* mutate source fixtures;
* create external side effects;
* infer person-level blame;
* overclaim causality from weak signals.

The skill may:

* load local demo fixtures;
* run local text-engine normalization;
* run local research labs;
* run critic/selection logic;
* build ensemble findings;
* compile decision cards;
* return JSON or text summaries.

## Important design rule

The experiment-lab skill is the deterministic local analysis engine.

External capabilities should be routed through MCP servers separately:

* Apify MCP for external data collection;
* Hugging Face MCP for optional model enrichment;
* n8n workflow or webhook for monitoring automation.

The experiment-lab skill consumes prepared data and returns structured findings. It should not directly own external integrations.
