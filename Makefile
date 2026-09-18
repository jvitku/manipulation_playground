COMPOSE := docker compose -f docker/compose.yaml
RUN     := $(COMPOSE) run --rm -T -u $(shell id -u):$(shell id -g) dev
ARGS    ?=

.PHONY: build shell test lint fmt run m0 m1 m2 m3 m4 m5 m6 all

build:
	$(COMPOSE) build dev

shell:
	$(COMPOSE) run --rm -u $(shell id -u):$(shell id -g) dev bash

test:
	$(RUN) pytest -q

lint:
	$(RUN) sh -c "ruff check . && ruff format --check ."

fmt:
	$(RUN) sh -c "ruff format . && ruff check --fix ."

# usage: make run S=scripts/01_gantry_insert.py ARGS="--x-offset-mm 1"
run:
	$(RUN) python $(S) $(ARGS)

m0:
	$(RUN) python scripts/00_probe_api.py --out outputs/m0

m1:
	$(RUN) python scripts/01_gantry_insert.py --out outputs/m1

m2:
	$(RUN) python scripts/02_contamination.py --out outputs/m2

m3:
	$(RUN) python scripts/03_solver_sweep.py --out outputs/m3

m4:
	$(RUN) python scripts/04_robosuite_wipe_ft.py --out outputs/m4

m5:
	$(RUN) python scripts/05_robosuite_gain_sweep.py --out outputs/m5

m6:
	$(RUN) python scripts/06_robosuite_nut_round.py --out outputs/m6
	$(RUN) python scripts/07_record_episodes.py --track A --n 50 --out data/trackA
	$(RUN) python scripts/07_record_episodes.py --track B --task Wipe --n 20 --out data/trackB

all: m0 m1 m2 m3 m4 m5 m6
