TEMPLATE_DIR := templates
OUTPUT_DIR := docs/latex
# Create questionnaire LaTeX document
create-questionnaire:
	@python scripts/generate_questionnaire_latex.py \
		--questions config/questionnaire_items.json \
		--template $(TEMPLATE_DIR)/questionnaire.tex.in \
		--output $(OUTPUT_DIR)/questionnaire.tex