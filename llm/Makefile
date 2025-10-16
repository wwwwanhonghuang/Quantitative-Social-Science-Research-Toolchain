# GLiNER Information Extraction System with CPU Management
GLINER_MODEL := urchade/gliner_multi-v2.1
MODEL_PATH := models/gliner-multi

# Download and setup targets
.PHONY: download-model verify-model clean-models


# Download GLiNER model
download-nre-model:
	@mkdir -p models
	@echo "Downloading GLiNER multi-v2.1 model..."
	@python scripts/download_model.py \
		--model-name $(GLINER_MODEL) \
		--save-path $(MODEL_PATH)
	@echo "✓ GLiNER model downloaded and saved"

# Download T5-small summarization model
download-summarizer-model:
	@mkdir -p models
	@echo "Downloading T5-small summarization model..."
	@python -c "\
	from transformers import T5ForConditionalGeneration, T5Tokenizer;\
	model_name = 't5-small';\
	tokenizer = T5Tokenizer.from_pretrained(model_name);\
	model = T5ForConditionalGeneration.from_pretrained(model_name);\
	tokenizer.save_pretrained('models/t5-small');\
	model.save_pretrained('models/t5-small');\
	"
	@echo "✓ T5-small summarization model downloaded and saved"


# Verify model download
verify-model:
	@echo "Verifying GLiNER model..."
	@python scripts/verify_model.py --model-path models/gliner-multi

# Clean downloaded models
clean-models:
	@rm -rf models/*
	@echo "✓ All models cleaned up"

# Full setup pipeline
setup-gliner: setup-environment download-model verify-model
	@echo "✓ GLiNER setup complete - ready for extraction tasks!"

# CPU Thread Management
TOTAL_THREADS := $(shell nproc)
RESERVED_THREADS := 2
AVAILABLE_THREADS := $(shell echo $$(($(TOTAL_THREADS) - $(RESERVED_THREADS))))
GLINER_THREADS := $(shell echo $$(($(AVAILABLE_THREADS) * 3 / 4)))

# Memory constraints
MAX_MEMORY := 4G

# Sociology entity types
ENTITY_TYPES := ["research_method", "theoretical_framework", "social_concept", "demographic_group", "geographic_region", "temporal_period", "academic_institution", "key_contribution", "methodological_approach", "research_gap"]

.PHONY: show-cpu-info extract-entities-cpu-optimized

# Show CPU configuration
show-cpu-info:
	@echo "CPU Configuration:"
	@echo "  Total threads: $(TOTAL_THREADS)"
	@echo "  Reserved for system: $(RESERVED_THREADS)"
	@echo "  Available for GLiNER: $(AVAILABLE_THREADS)"
	@echo "  GLiNER threads (75%): $(GLINER_THREADS)"

# CPU-optimized entity extraction (with model verification)
extract-entities-cpu-optimized: verify-model
	@echo "Using $(GLINER_THREADS) threads for GLiNER (reserving $(RESERVED_THREADS) for system)"
	@python scripts/extract_entities.py \
		--model-path $(MODEL_PATH) \
		--input $(PROJECT_DIR)/main.tex \
		--entity-types $(ENTITY_TYPES) \
		--output $(PROJECT_DIR)/extracted_entities.json \
		--cpu-threads $(GLINER_THREADS) \
		--max-memory $(MAX_MEMORY)

# Memory and CPU monitoring during processing
extract-with-monitoring: verify-model
	@echo "Starting extraction with resource monitoring..."
	@python scripts/monitored_extraction.py \
		--model-path $(MODEL_PATH) \
		--input $(PROJECT_DIR)/main.tex \
		--cpu-threads $(GLINER_THREADS) \
		--reserved-threads $(RESERVED_THREADS) \
		--max-memory $(MAX_MEMORY)

# Quick test with sample text
test-model:
	@echo "Testing GLiNER model with sample text..."
	@python scripts/test_gliner.py \
		--model-path $(MODEL_PATH) \
		--cpu-threads $(GLINER_THREADS)

# Show model info
model-info:
	@echo "GLiNER Model Information:"
	@echo "  Model: $(GLINER_MODEL)"
	@echo "  Local path: $(MODEL_PATH)"
	@echo "  Entity types: $(ENTITY_TYPES)"
	@make show-cpu-info