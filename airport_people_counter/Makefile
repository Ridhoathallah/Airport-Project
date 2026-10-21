# Airport X-Ray Queue — Camera-Only Prototype
# Convenience targets. Run with `make <target>` from this directory.
# Equivalent to using ./run.sh with the matching flags.

PY := venv/bin/python

.PHONY: run standalone headless define-roi preview record test smoke test-camera list-cameras api help

help:
	@echo "Targets:"
	@echo "  make run           run the camera-only prototype (standalone GUI)"
	@echo "  make standalone    run with OpenCV operator GUI window"
	@echo "  make headless      run headless with FastAPI server for React dashboard"
	@echo "  make define-roi    interactively set the queue ROI + counting line"
	@echo "  make preview       preview a saved layout"
	@echo "  make record        record analytics events to output/events.jsonl"
	@echo "  make test          run the unit test suite"
	@echo "  make smoke         quick offline smoke test on a sample image"
	@echo "  make test-camera   test camera without YOLO (default: index 0)"
	@echo "  make list-cameras  list available cameras"
	@echo "  make api           run with FastAPI server enabled (port 8080)"
	@echo "  make ml-help          ML dataset tooling (validate/build/split/train/eval)"
	@echo "  make dataset-validate validate a dataset root, write a JSON report"
	@echo "  make dataset-build    validate -> split -> manifest -> catalog"

run:
	$(PY) app/detect_people.py --mode standalone

standalone:
	$(PY) app/detect_people.py --mode standalone

headless:
	$(PY) app/detect_people.py --mode headless

define-roi:
	$(PY) app/detect_people.py --define-roi

preview:
	$(PY) app/detect_people.py --roi-preview

record:
	$(PY) app/detect_people.py --record output/events.jsonl

test:
	$(PY) -m unittest discover -s tests

smoke:
	$(PY) app/detect_people.py --source output/smoke_test/bus.jpg

test-camera:
	$(PY) app/detect_people.py --test-camera $(or $(INDEX),0)

list-cameras:
	$(PY) app/detect_people.py --list-cameras

api:
	$(PY) app/detect_people.py --api

# ─────────────────────────────────────────────────────────────
#  ML DATASET TOOLING  (data-engineering workstream)
#  Independent of the runtime application and its monitoring database.
# ─────────────────────────────────────────────────────────────
ML_PY        := venv/bin/python
DATASET_ROOT ?= datasets/people_v1
DATASET_NAME ?= people_v1
CATALOG      ?= datasets/metadata/dataset_catalog.db
SEED         ?= 20261020
RATIOS       ?= 0.7/0.2/0.1
WEIGHTS      ?= models/yolov5s.pt

.PHONY: dataset-validate dataset-build dataset-preview dataset-check \
        catalog-init catalog-stats catalog-datasets catalog-experiments \
        collect-hard train-person evaluate-person ml-help

ml-help:
	@echo "ML dataset tooling:"
	@echo "  make catalog-init         create the metadata catalog (idempotent)"
	@echo "  make dataset-validate     validate a dataset root, write a JSON report"
	@echo "  make dataset-build        validate -> split -> manifest -> catalog"
	@echo "  make dataset-check        training preflight for a dataset (no training)"
	@echo "  make dataset-preview      render annotation previews for review"
	@echo "  make catalog-stats        row counts per catalog table"
	@echo "  make catalog-datasets     list registered dataset versions"
	@echo "  make catalog-experiments  list registered training experiments"
	@echo "  make collect-hard         mine hard examples for human review"
	@echo "  make train-person         fine-tune with full experiment provenance"
	@echo "  make evaluate-person      evaluate a checkpoint vs the baseline"

catalog-init:
	$(ML_PY) scripts/catalog.py --catalog $(CATALOG) init

dataset-validate:
	$(ML_PY) scripts/validate_dataset.py --root $(DATASET_ROOT) \
		--report datasets/reports/$(DATASET_NAME).json

dataset-check:
	$(ML_PY) scripts/train_person.py --data $(DATASET_ROOT)/data.yaml --weights $(WEIGHTS) \
		--dry-run

dataset-build:
	$(ML_PY) scripts/build_dataset.py --root $(DATASET_ROOT) --name $(DATASET_NAME) \
		--seed $(SEED) --ratios $(RATIOS) --catalog $(CATALOG)

dataset-preview:
	$(ML_PY) scripts/validate_dataset.py --root $(DATASET_ROOT) \
		--preview-out datasets/reports/preview --preview-limit 40

catalog-stats:
	$(ML_PY) scripts/catalog.py --catalog $(CATALOG) stats

catalog-datasets:
	$(ML_PY) scripts/catalog.py --catalog $(CATALOG) datasets

catalog-experiments:
	$(ML_PY) scripts/catalog.py --catalog $(CATALOG) experiments

collect-hard:
	$(ML_PY) scripts/collect_hard_examples.py --source $(SOURCE) --weights $(WEIGHTS) \
		--out datasets/collected/run_001

train-person:
	$(ML_PY) scripts/train_person.py --data $(DATASET_ROOT)/data.yaml --weights $(WEIGHTS) \
		--epochs $(EPOCHS) --imgsz $(IMGSZ) --batch-size $(BATCH) --device $(DEVICE) \
		--catalog $(CATALOG)

evaluate-person:
	$(ML_PY) scripts/evaluate_model.py --weights $(CANDIDATE) --baseline $(WEIGHTS) \
		--data $(DATASET_ROOT)/data.yaml --split test --catalog $(CATALOG)
