# Research Assistant Workspace Setup
VENV_NAME := ra-python
PYTHON := python3
PIP := pip3

.PHONY: setup-python activate install clean help

# Create Python virtual environment
setup-python:
	@echo "Setting up Research Assistant workspace..."
	@which $(PYTHON) > /dev/null || (echo "Python3 not found. Please install Python 3.8+"; exit 1)
	@$(PYTHON) -m venv $(VENV_NAME)
	@echo "✓ Virtual environment '$(VENV_NAME)' created"
	@echo "✓ Activate with: source $(VENV_NAME)/bin/activate"
	@echo "✓ Then run: make install"

# Activate virtual environment (info only)
activate:
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV_NAME)/bin/activate"
	@echo ""
	@echo "Or on Windows:"
	@echo "  $(VENV_NAME)\\Scripts\\activate"
	@echo ""
	@echo "After activation, run: make install"

# Install all dependencies in activated environment
install:
	@echo "Installing Research Assistant dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	@$(PIP) install transformers accelerate
	@$(PIP) install gliner spacy networkx matplotlib seaborn
	@$(PIP) install jupyter ipykernel
	@$(PIP) install requests beautifulsoup4 pdfplumber
	@echo "✓ Installing spaCy language model..."
	@$(PYTHON) -m spacy download en_core_web_sm
	@echo "✓ All dependencies installed"
	@echo "✓ You can now navigate to specific tool directories and run their Makefiles"

# Clean virtual environment
clean:
	@echo "Cleaning up workspace..."
	@rm -rf $(VENV_NAME)
	@echo "✓ Virtual environment removed"

# Setup complete research environment
setup-all: setup-python
	@echo ""
	@echo "Now activate the environment and install dependencies:"
	@echo "  source $(VENV_NAME)/bin/activate && make install"

# Show help
help:
	@echo "Research Assistant Workspace Management:"
	@echo "  make setup-python    - Create Python virtual environment"
	@echo "  make activate        - Show activation commands"
	@echo "  make install         - Install dependencies (run after activation)"
	@echo "  make setup-all       - Complete setup (env creation + shows next steps)"
	@echo "  make clean           - Remove virtual environment"
	@echo ""
	@echo "Workflow:"
	@echo "  1. make setup-python"
	@echo "  2. source $(VENV_NAME)/bin/activate"
	@echo "  3. make install"
	@echo "  4. cd literature_review && make setup-gliner"
	@echo ""
	@echo "Available tools:"
	@echo "  - literature_review/    : GLiNER entity extraction"
	@echo "  - data_analysis/        : Data processing scripts"
	@echo "  - visualization/        : Network and theme visualization"

# Check system status
status:
	@echo "Research Assistant Workspace Status:"
	@echo "Python: $$(which $(PYTHON))"
	if [ -d "$(VENV_NAME)" ]; then \
		echo "Virtual environment: ✓ Present ($(VENV_NAME))"; \
	else \
		echo "Virtual environment: ✗ Missing"; \
	fi
	@if command -v $(PIP) > /dev/null && $(PIP) list | grep -q gliner; then \
		echo "GLiNER: ✓ Installed"; \
	else \
		echo "GLiNER: ✗ Not installed"; \
	fi