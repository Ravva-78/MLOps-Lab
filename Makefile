# Use bash so we can write cleaner shell logic
SHELL := /bin/bash

# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup run logs clean

# Create venv (if missing) and install requirements
setup:
	@echo "Checking virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Upgrading pip..."
	@$(PIP) install --upgrade pip

	@if [ -f requirements.txt ]; then \
		echo "Installing dependencies..."; \
		$(PIP) install -r requirements.txt; \
	else \
		echo "No requirements.txt found. Skipping dependency install."; \
	fi

	@if [ ! -d ".dvc" ]; then \
		echo "Initializing DVC..."; \
		$(VENV)/bin/dvc init --no-scm -f; \
	else \
		echo "DVC already initialized."; \
	fi

	@echo "Setup complete."

# Run the application
run:
	@export PYTHONPATH=$${PYTHONPATH}:$$(pwd) && \
	$(PYTHON) src/app.py

# View logs
logs:
	@if [ -f logs/pipeline.log ]; then \
		cat logs/pipeline.log; \
	else \
		echo "No logs found."; \
	fi

# Clean project artifacts (but keep venv)
clean:
	@echo "Cleaning logs and cache..."
	@rm -rf logs/
	@find . -type d -name "__pycache__" -exec rm -rf {} +