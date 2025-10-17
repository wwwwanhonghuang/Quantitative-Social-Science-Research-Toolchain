# Model definitions
GLINER_MODEL = urchade/gliner_multi-v2.1
SENTENCE_TRANSFORMER_MODEL = sentence-transformers/all-MiniLM-L6-v2
SUMMARIZER_MODEL = facebook/bart-large-cnn-samsum
CLASSIFIER_MODEL = distilbert-base-uncased
TOPIC_MODEL_BACKEND = sentence-transformers/all-MiniLM-L6-v2

# Paths
MODEL_PATH = models

.PHONY: download-all-models download-nre-model download-summarizer-model \
        download-sentence-transformer download-classifier install-topic-modeling \
        test-models

# Download all essential models
download-all-models: download-sentence-transformer download-nre-model \
                    download-summarizer-model download-classifier install-topic-modeling
	@echo "✓ All models downloaded successfully"
	@echo ""
	@echo "Total disk space: ~1.5GB"
	@echo "Expected RAM usage: ~6-8GB for all models"
	@make model-summary

# 1. Sentence Transformer (FASTEST for embeddings/similarity)
# all-MiniLM-L6-v2: 80MB, very fast on CPU
download-sentence-transformer:
	@mkdir -p $(MODEL_PATH)
	@echo "Downloading Sentence Transformer (all-MiniLM-L6-v2)..."
	@echo "  Size: ~80MB | RAM: ~500MB | Speed: FAST"
	@python -c "\
	from sentence_transformers import SentenceTransformer;\
	model = SentenceTransformer('$(SENTENCE_TRANSFORMER_MODEL)');\
	model.save('$(MODEL_PATH)/sentence-transformer');\
	print('✓ Model saved to $(MODEL_PATH)/sentence-transformer');\
	"
	@echo "✓ Sentence Transformer downloaded"

# 2. GLiNER for NER (already planned, good CPU performance)
download-nre-model:
	@mkdir -p $(MODEL_PATH)
	@echo "Downloading GLiNER multi-v2.1 model..."
	@echo "  Size: ~500MB | RAM: ~2GB | Speed: MODERATE"
	@python scripts/download_model.py \
		--model-name $(GLINER_MODEL) \
		--save-path $(MODEL_PATH)
	@echo "✓ GLiNER model downloaded"

# 3. BART-CNN (better than T5-small, optimized for summaries)
# OR use distilbart-cnn (smaller, faster)
download-summarizer-model:
	@mkdir -p $(MODEL_PATH)
	@echo "Downloading BART-large-CNN summarization model..."
	@echo "  Size: ~1.6GB | RAM: ~3GB | Speed: MODERATE"
	@echo "  Note: For faster inference, use distilbart-cnn-12-6 (400MB)"
	@python -c "\
	from transformers import BartForConditionalGeneration, BartTokenizer;\
	model_name = '$(SUMMARIZER_MODEL)';\
	tokenizer = BartTokenizer.from_pretrained(model_name);\
	model = BartForConditionalGeneration.from_pretrained(model_name);\
	tokenizer.save_pretrained('$(MODEL_PATH)/bart-summarizer');\
	model.save_pretrained('$(MODEL_PATH)/bart-summarizer');\
	print('✓ Model saved to $(MODEL_PATH)/bart-summarizer');\
	"
	@echo "✓ BART summarizer downloaded"

# Alternative: Faster distilbart (uncomment if BART is too slow)
download-fast-summarizer:
	@mkdir -p $(MODEL_PATH)
	@echo "Downloading DistilBART-CNN (faster alternative)..."
	@echo "  Size: ~400MB | RAM: ~1.5GB | Speed: FAST"
	@python -c "\
	from transformers import BartForConditionalGeneration, BartTokenizer;\
	model_name = 'sshleifer/distilbart-cnn-12-6';\
	tokenizer = BartTokenizer.from_pretrained(model_name);\
	model = BartForConditionalGeneration.from_pretrained(model_name);\
	tokenizer.save_pretrained('$(MODEL_PATH)/distilbart-summarizer');\
	model.save_pretrained('$(MODEL_PATH)/distilbart-summarizer');\
	print('✓ Model saved to $(MODEL_PATH)/distilbart-summarizer');\
	"
	@echo "✓ DistilBART summarizer downloaded"

# 4. DistilBERT for classification (CPU-optimized)
download-classifier:
	@mkdir -p $(MODEL_PATH)
	@echo "Downloading DistilBERT classifier..."
	@echo "  Size: ~250MB | RAM: ~1GB | Speed: FAST"
	@python -c "\
	from transformers import DistilBertTokenizer, DistilBertForSequenceClassification;\
	model_name = '$(CLASSIFIER_MODEL)';\
	tokenizer = DistilBertTokenizer.from_pretrained(model_name);\
	model = DistilBertForSequenceClassification.from_pretrained(model_name);\
	tokenizer.save_pretrained('$(MODEL_PATH)/distilbert');\
	model.save_pretrained('$(MODEL_PATH)/distilbert');\
	print('✓ Model saved to $(MODEL_PATH)/distilbert');\
	"
	@echo "✓ DistilBERT classifier downloaded"

# 5. Topic Modeling (BERTopic - reuses sentence transformer)
install-topic-modeling:
	@echo "Installing topic modeling dependencies..."
	@pip install bertopic umap-learn hdbscan scikit-learn
	@echo "✓ Topic modeling dependencies installed (uses all-MiniLM-L6-v2)"

# Optional: Download quantized models for even faster inference
download-quantized-models:
	@echo "Installing ONNX Runtime for quantized models..."
	@pip install optimum[onnxruntime]
	@echo "✓ ONNX Runtime installed"
	@echo ""
	@echo "To convert models to ONNX (faster CPU inference):"
	@echo "  optimum-cli export onnx --model distilbert-base-uncased models/distilbert-onnx"


# Verify model download and test inference
verify-model:
	@echo "════════════════════════════════════════════════════"
	@echo "  Verifying Downloaded Models"
	@echo "════════════════════════════════════════════════════"
	@echo ""
	@python scripts/verify_models.py

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