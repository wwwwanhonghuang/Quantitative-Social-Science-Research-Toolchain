# Top-level Makefile for creating new projects

# Usage:
#   make new-project NAME=project1

TEMPLATE := template/project_makefile_template.mk

.PHONY: new-project

new-project:
	@echo "Creating new project: $(PROJECT_NAME)..."

	# Create project folders
	@mkdir -p projects/$(PROJECT_NAME)/data/raw
	@mkdir -p projects/$(PROJECT_NAME)/data/processed
	@mkdir -p projects/$(PROJECT_NAME)/results/tables
	@mkdir -p projects/$(PROJECT_NAME)/results/figures
	@mkdir -p projects/$(PROJECT_NAME)/results/models

	# Create an empty README.md
	@touch projects/$(PROJECT_NAME)/README.md

	# Create a project-level Makefile
	@echo "Creating Makefile for project $(NEW_PROJECT)..."
	@export DATA_DIR=$(realpath projects/$(PROJECT_NAME)/data) && \
	export RESULTS_DIR=$(realpath projects/$(PROJECT_NAME)/results) && \
	export PROJECT_NAME=$(PROJECT_NAME) && \
	envsubst < $(TEMPLATE) > projects/$(PROJECT_NAME)/Makefile

	@echo "✓ Project $(PROJECT_NAME) created successfully!"
