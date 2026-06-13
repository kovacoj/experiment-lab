PYTHON ?= python
SCENARIO ?= reputation_monitor

.PHONY: test run run-supply-chain docker-build docker-run docker-run-supply-chain docker-test download-sentiment-model smoke-sentiment smoke-sentiment-offline

test:
	$(PYTHON) -m unittest discover -s tests

run:
	$(PYTHON) -m app.labs.runner --scenario $(SCENARIO)

run-supply-chain:
	$(PYTHON) -m app.labs.runner --scenario supply_chain_risk

docker-build:
	docker build -t experiment-lab .

docker-run:
	docker compose run --rm experiment-lab

docker-run-supply-chain:
	docker compose run --rm experiment-lab-supply-chain

docker-test:
	docker compose run --rm experiment-lab-tests

download-sentiment-model:
	uv run python scripts/download_sentiment_model.py

smoke-sentiment:
	uv run python scripts/smoke_test_sentiment.py

smoke-sentiment-offline:
	HF_HUB_OFFLINE=1 uv run python scripts/smoke_test_sentiment.py
