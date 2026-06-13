PYTHON ?= python
SCENARIO ?= reputation_monitor

.PHONY: test run run-supply-chain docker-build docker-run docker-run-supply-chain docker-test

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
