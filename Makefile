#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = car-listing-visual-verification
PYTHON_INTERPRETER = python3
PIPELINE_REPO_PATH = /Users/ruslan/Projects/car-listing-data-pipeline

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint source code
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code
.PHONY: format
format:
	ruff check --fix
	ruff format

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"

## Show where pipeline commands moved
.PHONY: pipeline-location
pipeline-location:
	@echo "Data pipeline moved to: $(PIPELINE_REPO_PATH)"
	@echo "Use that repository for scraping/filtering/HF upload commands."

## Compatibility target: pipeline was moved out of this repository
.PHONY: data
data:
	@echo "Target removed from this repository. Run pipeline in $(PIPELINE_REPO_PATH)."
	@exit 1

## Compatibility target: pipeline was moved out of this repository
.PHONY: hf-upload
hf-upload:
	@echo "Target removed from this repository. Run hf-upload in $(PIPELINE_REPO_PATH)."
	@exit 1

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
