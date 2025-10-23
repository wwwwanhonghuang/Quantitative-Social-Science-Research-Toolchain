# Literature Review Project Generator
# Using envsubst with existing templates

LITERATURE_TITLE ?= "Literature_Review"
AUTHOR ?= "Your Name \\\\ Your Institution"
NORMALIZED_TITLE := $(shell echo '$(LITERATURE_TITLE)' | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | sed 's/[^a-z0-9_]//g')
PROJECT_DIR := reviews/$(NORMALIZED_TITLE)
TEMPLATE_DIR := templates
CURRENT_DATE := $(shell date +'%Y-%m-%d')
PARTIALS_DIR := $(TEMPLATE_DIR)/partials
SUMMARIZER_PATH := $(realpath ../llm/scripts/app/summarize.py)

export LITERATURE_TITLE
export AUTHOR
export NORMALIZED_TITLE
export CURRENT_DATE

.PHONY: all project pdf watch clean list open remove help

all: project pdf

# Create project from templates

project: $(PROJECT_DIR)/main.tex $(PROJECT_DIR)/references.bib $(PROJECT_DIR)/sections/abstract.tex $(PROJECT_DIR)/sections/introduction.tex $(PROJECT_DIR)/sections/methodology.tex $(PROJECT_DIR)/sections/results.tex $(PROJECT_DIR)/sections/conclusion.tex $(PROJECT_DIR)/sections/discussion.tex $(PROJECT_DIR)/sections/appendix.tex $(PROJECT_DIR)/Makefile $(PROJECT_DIR)/README.md $(PROJECT_DIR)/summary.md $(PROJECT_DIR)/literature_list.list
	@echo "✓ Literature review project created: $(PROJECT_DIR)"

# Create project directory structure
$(PROJECT_DIR)/.dir:
	@mkdir -p $(PROJECT_DIR)/sections
	@mkdir -p $(PROJECT_DIR)/figures
	@mkdir -p $(PROJECT_DIR)/tables
	@touch $@

# Generate main.tex from template
$(PROJECT_DIR)/main.tex: $(TEMPLATE_DIR)/main.tex.in $(PROJECT_DIR)/.dir
	@envsubst '$$LITERATURE_TITLE $$AUTHOR $$NORMALIZED_TITLE $$CURRENT_DATE' < $< > $@
	@echo "✓ Generated: $@"

# Generate references.bib from template
$(PROJECT_DIR)/references.bib: $(TEMPLATE_DIR)/references.bib.in $(PROJECT_DIR)/.dir
	@envsubst '$$LITERATURE_TITLE $$AUTHOR $$NORMALIZED_TITLE $$CURRENT_DATE' < $< > $@
	@echo "✓ Generated: $@"

# Generate empty summary.md
$(PROJECT_DIR)/summary.md: $(PROJECT_DIR)/.dir
	@touch $@
	@echo "✓ Created empty summary: $@"


$(PROJECT_DIR)/literature_list.list: $(PROJECT_DIR)/.dir
	@touch $@
	@echo "✓ Created Literature List: $@"

# Generate section files from individual templates
$(PROJECT_DIR)/sections/abstract.tex: $(PARTIALS_DIR)/abstract.tex.in $(PROJECT_DIR)/.dir
	envsubst < $< > $@

$(PROJECT_DIR)/sections/introduction.tex: $(PARTIALS_DIR)/introduction.tex.in $(PROJECT_DIR)/.dir
	envsubst < $< > $@

$(PROJECT_DIR)/sections/methodology.tex: $(PARTIALS_DIR)/methodology.tex.in $(PROJECT_DIR)/.dir
	envsubst < $< > $@

$(PROJECT_DIR)/sections/results.tex: $(PARTIALS_DIR)/results.tex.in $(PROJECT_DIR)/.dir
	envsubst < $< > $@

$(PROJECT_DIR)/sections/discussion.tex: $(PARTIALS_DIR)/discussion.tex.in $(PROJECT_DIR)/.dir
	envsubst < $< > $@

$(PROJECT_DIR)/sections/conclusion.tex: $(PARTIALS_DIR)/conclusion.tex.in $(PROJECT_DIR)/.dir
	envsubst < $< > $@

$(PROJECT_DIR)/sections/appendix.tex: $(PARTIALS_DIR)/appendix.tex.in $(PROJECT_DIR)/.dir
	envsubst < $< > $@
	@echo "✓ Generated all section files"

# Generate project-specific Makefile
$(PROJECT_DIR)/Makefile: $(TEMPLATE_DIR)/project.Makefile.in $(PROJECT_DIR)/.dir
	@envsubst '$$LITERATURE_TITLE $$NORMALIZED_TITLE' < $< > $@
	@echo "✓ Generated: $@"

# Generate README
$(PROJECT_DIR)/README.md: $(TEMPLATE_DIR)/README.md.in $(PROJECT_DIR)/.dir
	@envsubst '$$LITERATURE_TITLE $$NORMALIZED_TITLE $$CURRENT_DATE' < $< > $@
	@echo "✓ Generated: $@"

# Compile LaTeX to PDF
pdf: project
	@cd $(PROJECT_DIR) && \
	$(MAKE) pdf
	@echo "✓ PDF generated: $(PROJECT_DIR)/main.pdf"

# Quick compile (existing project)
quickpdf:
	@if [ -f "$(PROJECT_DIR)/main.tex" ]; then \
		cd $(PROJECT_DIR) && \
		$(MAKE) quickpdf; \
	else \
		echo "❌ Project not found: $(PROJECT_DIR)"; \
	fi

# Watch for changes and auto-compile
watch:
	@if [ -f "$(PROJECT_DIR)/main.tex" ]; then \
		echo "Watching for changes in $(PROJECT_DIR)..."; \
		cd $(PROJECT_DIR) && \
		$(MAKE) watch; \
	else \
		echo "❌ Project not found: $(PROJECT_DIR)"; \
	fi

# List all literature review projects
list:
	@echo "Literature Review Projects:"
	@if [ -d "reviews" ]; then \
		for dir in reviews/*; do \
			if [ -d "$$dir" ] && [ -f "$$dir/main.tex" ]; then \
				echo "  - $$(basename $$dir)"; \
			fi; \
		done; \
	else \
		echo "  No projects found"; \
	fi

# Open project PDF
open:
	@if [ -f "$(PROJECT_DIR)/main.pdf" ]; then \
		xdg-open "$(PROJECT_DIR)/main.pdf" 2>/dev/null || \
		open "$(PROJECT_DIR)/main.pdf" 2>/dev/null || \
		echo "Please open manually: $(PROJECT_DIR)/main.pdf"; \
	else \
		echo "❌ PDF not found. Run 'make pdf' first"; \
	fi

# Clean build artifacts
clean:
	@if [ -d "$(PROJECT_DIR)" ]; then \
		cd $(PROJECT_DIR) && \
		$(MAKE) clean; \
	fi

# Remove entire project
remove:
	@if [ -d "$(PROJECT_DIR)" ]; then \
		rm -rf "$(PROJECT_DIR)"; \
		echo "✓ Project removed: $(NORMALIZED_TITLE)"; \
	else \
		echo "❌ Project not found: $(NORMALIZED_TITLE)"; \
	fi

# Help target
help:
	@echo "Literature Review Management System"
	@echo ""
	@echo "Usage:"
	@echo "  make project LITERATURE_TITLE='Your Review Title' [AUTHOR='Your Name']"
	@echo "  make pdf LITERATURE_TITLE='Your Title'"
	@echo "  make quickpdf LITERATURE_TITLE='Your Title'"
	@echo "  make watch LITERATURE_TITLE='Your Title'"
	@echo "  make open LITERATURE_TITLE='Your Title'"
	@echo "  make list"
	@echo "  make clean LITERATURE_TITLE='Your Title'"
	@echo "  make remove LITERATURE_TITLE='Your Title'"
	@echo "  make integrate_summary"
	@echo "  make summary-status"
	@echo ""
	@echo "Examples:"
	@echo "  make project LITERATURE_TITLE='Machine Learning Survey 2024'"
	@echo "  make project LITERATURE_TITLE='Deep Learning in Healthcare' AUTHOR='John Doe \\\\ University of Research'"
	@echo "  make pdf LITERATURE_TITLE='machine_learning_survey_2024'"

_summary-by-llm:
	@echo "Summarizing input file $(INPUT) use $(SUMMARIZER_PATH)..."
	@cd ../llm && python $(SUMMARIZER_PATH) $(INPUT) $(OUTPUT)
	@echo "✓ Done summarizing $(INPUT) -> $(OUTPUT)"

# Expand and summarize a single project
summarize-project-by-llm:
	@echo "Expanding LaTeX sections for $(PROJECT)..."
	@python scripts/expand_tex.py $(PROJECT)/main.tex $(PROJECT)/main_expanded.txt
	@echo "✓ Done expanding sections."
	@$(MAKE) _summary-by-llm INPUT=$(PROJECT)/main_expanded.txt OUTPUT=$(PROJECT)/llm_summary.txt
	@echo "✓ Summary saved to $(PROJECT)/llm_summary.txt"

# Expand and summarize all projects listed in PROJECTS
summarize-all-projects-by-llm:
	@for proj in ./reviews; do \
		$(MAKE) summarize-project-by-llm PROJECT=$$proj; \
	done

# Integrate all project summaries
integrate_summary:
	@echo "Integrating all project summaries..."
	@echo "# Integrated Research Summary - $(shell date +'%Y-%m-%d %H:%M')" > integrated_summary.md
	@echo "" >> integrated_summary.md
	@for project_path in reviews/*; do \
		if [ -f "$$project_path/summary.md" ]; then \
			echo "## Project: $$(basename $$project_path)" >> integrated_summary.md; \
			echo "" >> integrated_summary.md; \
			cat "$$project_path/summary.md" >> integrated_summary.md; \
			echo "" >> integrated_summary.md; \
			echo "---" >> integrated_summary.md; \
			echo "" >> integrated_summary.md; \
		fi; \
	done
	@echo "✓ Integrated summary created: integrated_summary.md"

# Show summary status
summary-status:
	@echo "Project Summary Status:"
	@for project_path in reviews/*; do \
		if [ -f "$$project_path/summary.md" ]; then \
			lines=$$(wc -l < "$$project_path/summary.md"); \
			if [ $$lines -eq 0 ]; then \
				echo "  $$(basename $$project_path): ⚪ Empty"; \
			else \
				echo "  $$(basename $$project_path): ✅ Has content ($$lines lines)"; \
			fi; \
		else \
			echo "  $$(basename $$project_path): ❌ No summary.md"; \
		fi; \
	done