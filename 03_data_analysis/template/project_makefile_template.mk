# Project-level Makefile for $(PROJECT_NAME)

PROJECT := ${PROJECT_NAME}
DATA_DIR := data
RAW_DIR := ${DATA_DIR}/raw
PROCESSED_DIR := ${DATA_DIR}/processed
RESULTS_DIR := results
TABLES_DIR := ${RESULTS_DIR}/tables
FIGURES_DIR := ${RESULTS_DIR}/figures
MODELS_DIR := ${RESULTS_DIR}/models
PREPROCESSING_SCRIPT := ${PREPROCESSING_SCRIPT}
.PHONY: all preprocess statistics causal figures clean

all: preprocess statistics causal figures

preprocess:
	@echo 'Run preprocessing for project ${PROJECT_NAME}'
	python $(PREPROCESSING_SCRIPT) --project_path projects/$(PROJECT)

statistics:
	@echo 'Run statistics for project ${PROJECT_NAME}'

causal:
	@echo 'Run causal inference for project ${PROJECT_NAME}'

figures:
	@echo 'Generate figures for project ${PROJECT_NAME}'

clean:
	@echo 'Clean processed data and results for project ${PROJECT_NAME}'
