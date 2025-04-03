#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ai4health
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 ai4health
	isort --check --diff ai4health
	black --check ai4health

## Format source code with black
.PHONY: format
format:
	isort ai4health
	black ai4health

## Run tests
.PHONY: test
test:
	python -m pytest tests

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset (default config)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m ai4health.dataset

## Make dataset with custom options (e.g., make data-custom SR=22050 DURATION=3 MFCC=30 MIN_CONF=0.5)
.PHONY: data-custom
data-custom: requirements
	$(PYTHON_INTERPRETER) -m ai4health.dataset \
		--sr $(SR) \
		--duration $(DURATION) \
		--n-mfcc $(MFCC) \
		--min-confidence $(MIN_CONF)

## Train and register a CNN model
.PHONY: train-model
train-model: requirements
	$(PYTHON_INTERPRETER) -m ai4health.modeling.train \
		--model-name $(MODEL) \
		--version $(VERSION) \
		--epochs $(EPOCHS)

## Predict using a registered model
.PHONY: predict-model
predict-model: requirements
	$(PYTHON_INTERPRETER) -m ai4health.modeling.predict \
		--model-name $(MODEL) \
		--version $(VERSION)

## Manually register a model version
.PHONY: register-model
register-model: requirements
	$(PYTHON_INTERPRETER) -m ai4health.modeling.registry register-model \
		--model-name $(MODEL) \
		--version $(VERSION) \
		--config-path $(CONFIG) \
		--metrics-path $(METRICS) \
		--model-path $(MODEL_PATH)

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
