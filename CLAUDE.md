# CLAUDE.md

## Purpose
- This repo is the local deterministic Signal Foundry analysis engine.
- When using Claude from the parent orchestrator workspace, prefer the stable CLI wrapper under `app.agent_skills.experiment_lab_cli`.

## Stable Interface
- Use:

```bash
uv run python -m app.agent_skills.experiment_lab_cli list-scenarios
uv run python -m app.agent_skills.experiment_lab_cli list-labs --scenario reputation_monitor
uv run python -m app.agent_skills.experiment_lab_cli run-analysis --scenario reputation_monitor --format json
uv run python -m app.agent_skills.experiment_lab_cli refresh-compare --scenario reputation_monitor --format json
```

- Do not import random internal lab modules directly unless you are editing implementation code.

## Safety
- This repo must remain local and deterministic.
- Do not add Apify, Hugging Face, n8n, or live web-service calls into the skill wrapper.

## Skill Doc
- Full skill contract:

`skills/experiment-lab/SKILL.md`
