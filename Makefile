COMPOSE := docker compose -f docker/compose.yaml
RUN     := $(COMPOSE) run --rm -T -u $(shell id -u):$(shell id -g) dev
RUNT    := $(COMPOSE) run --rm -T -u $(shell id -u):$(shell id -g) train
ARGS    ?=

.PHONY: build build-train shell test lint fmt run m0 m1 m2 m3 m4 m5 m6 all s1 report

build:
	$(COMPOSE) build dev

build-train: build
	$(COMPOSE) build train

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
	$(RUN) python scripts/07_record_episodes.py --track A --n 50 --noise-mm 0.5 --out data/trackA
	$(RUN) python scripts/07_record_episodes.py --track B --task Wipe --n 20 --out data/trackB_Wipe
	$(RUN) python scripts/07_record_episodes.py --track B --task NutAssemblyRound --n 10 --out data/trackB_NutAssemblyRound
	$(RUN) python scripts/08_robosuite_peg_in_hole.py --out outputs/m6_pih
	$(RUN) python scripts/07_record_episodes.py --track B --task PegInHole --n 20 --noise-mm 0.5 --out data/trackB_PegInHole

all: m0 m1 m2 m3 m4 m5 m6

# Stage 1 (Tier 1 BC experiment): expert data, train force / no-force, closed-loop eval
s1:
	$(RUN) python scripts/09_collect_expert.py --n 300 --out data/expert_trackA
	$(RUNT) python scripts/10_train_bc.py --data data/expert_trackA --out outputs/s1/force --force 1
	$(RUNT) python scripts/10_train_bc.py --data data/expert_trackA --out outputs/s1/noforce --force 0
	$(RUN) python scripts/11_eval_bc.py --ckpt outputs/s1/force/best.pt outputs/s1/noforce/best.pt --n 50 --out outputs/s1/eval

# HTML report (docs/report/index.html): needs outputs from `make all` and `make s1` (+ noise sweep)
report:
	$(RUNT) python scripts/13_policy_videos.py --seeds 50003 50008 50022
	$(RUN) python scripts/14_report_data.py
	$(RUN) python docs/report/build.py
