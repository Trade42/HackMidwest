.PHONY: install test test-all smoke reproduce demo physics calibrate

install:
	pip install -e ".[dev]"

test:            ## fast tests (what CI runs)
	python -m pytest -q -m "not slow"

test-all:        ## including exhaustive and density-matrix checks
	python -m pytest -q

smoke:           ## full pipeline on 3 questions with the stub reader
	python reproduce.py --config configs/smoke.yaml --quick

reproduce:       ## every number and figure of the offline run
	python scripts/fetch_offline_models.py
	python reproduce.py --config configs/offline.yaml

demo:
	python -m qctx.demo --config configs/offline.yaml

calibrate:
	python scripts/calibrate.py --questions data/calibration.jsonl --scores scores

physics:
	python physics/make_all.py
