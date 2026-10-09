#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = MLOps-Vera
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint the code: ruff (format + rules) and pylint (use `make format` to fix formatting)
.PHONY: lint
lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run pylint mlops_vera tests

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix .
	uv run ruff format .

## Lint the notebooks and the repository with Pynblint
# Isolated tool (uvx): it pins an old typer that conflicts with ours, needs click < 8.2
# and UTF-8 mode on Windows.
.PHONY: nblint
nblint:
	PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uvx --from pynblint --with "click<8.2" pynblint .





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Run the DVC data pipeline (download -> preprocess -> split)
.PHONY: data
data:
	uv run dvc repro split

## Run the DVC pipeline up to the trained model (... -> embed -> train)
.PHONY: train
train:
	uv run dvc repro train

## Run the leave-one-generator-out study (MR-4: logo@<generator> -> logo_summary)
.PHONY: logo
logo:
	uv run dvc repro logo_summary

## Validate the image metadata with Great Expectations (Data Docs in reports/data_docs)
.PHONY: validate
validate:
	uv run dvc repro validate_data

## Run all tests with coverage (`uv run pytest -m "not model"` skips the slower model tests)
.PHONY: test
test:
	uv run pytest --cov=mlops_vera --cov-report=term-missing

## Run every quality check: lint, notebook lint, data validation and tests
.PHONY: qa
qa: lint nblint validate test


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
