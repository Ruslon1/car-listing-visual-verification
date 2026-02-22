#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = car-listing-visual-verification
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
DROM_CLASSES = configs/classes.yaml
DROM_QPS = 1.5
DROM_CONCURRENCY = 12
DROM_LISTINGS_PER_CLASS = 150
DROM_CLI = $(PYTHON_INTERPRETER) -m car_listing_visual_verification.data.drom
HF_RELEASE_DIR = data/processed/hf_release
HF_DATASET_NAME = drom-car-listings-99-classes
HF_FILE_MODE = hardlink
HF_LICENSE = other

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: drom-discover
drom-discover:
	$(DROM_CLI) discover --classes $(DROM_CLASSES) --qps $(DROM_QPS) --concurrency $(DROM_CONCURRENCY) --max-listings-per-class $(DROM_LISTINGS_PER_CLASS)

.PHONY: drom-fetch-meta
drom-fetch-meta:
	$(DROM_CLI) fetch-meta --classes $(DROM_CLASSES) --qps $(DROM_QPS) --concurrency $(DROM_CONCURRENCY)

.PHONY: drom-fetch-images
drom-fetch-images:
	$(DROM_CLI) fetch-images --classes $(DROM_CLASSES) --qps $(DROM_QPS) --concurrency $(DROM_CONCURRENCY)

.PHONY: drom-validate
drom-validate:
	$(DROM_CLI) validate

.PHONY: drom-filter-content
drom-filter-content:
	$(DROM_CLI) filter-content

.PHONY: drom-dedup
drom-dedup:
	$(DROM_CLI) dedup

.PHONY: drom-manifest
drom-manifest:
	$(DROM_CLI) prepare-manifest

.PHONY: drom-split
drom-split:
	$(DROM_CLI) split --val-ratio 0.1 --test-ratio 0.1 --seed 42

.PHONY: drom-run-all
drom-run-all:
	$(DROM_CLI) run-all --classes $(DROM_CLASSES) --qps $(DROM_QPS) --concurrency $(DROM_CONCURRENCY) --max-listings-per-class $(DROM_LISTINGS_PER_CLASS) --no-cache

.PHONY: drom-prune-artifacts
drom-prune-artifacts:
	rm -rf data/raw/drom/pages data/interim/drom
	mkdir -p data/raw/drom/pages data/interim/drom

.PHONY: data
data: requirements drom-run-all drom-prune-artifacts

.PHONY: hf-release
hf-release:
	$(DROM_CLI) prepare-hf-release --manifest-path data/processed/manifest.parquet --class-mapping-path data/processed/class_mapping.parquet --output-dir $(HF_RELEASE_DIR) --dataset-name $(HF_DATASET_NAME) --file-mode $(HF_FILE_MODE) --license-id $(HF_LICENSE)

.PHONY: hf-upload
hf-upload: hf-release
	@if [ -z "$(DATASET_REPO)" ]; then echo "Usage: make hf-upload DATASET_REPO=<hf-user>/<dataset-name>"; exit 1; fi
	hf repo create $(DATASET_REPO) --repo-type dataset --private || true
	hf upload-large-folder --repo-type dataset $(DATASET_REPO) $(HF_RELEASE_DIR) --num-workers 16


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
