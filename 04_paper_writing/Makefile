# Top-level Makefile for paper projects
# Usage: make new-paper NAME=paper1

TEMPLATE_DIR := ./templates/paper
MAIN_TEX := main.tex
.PHONY: new-paper

new-paper:
	@echo "Creating new paper project: $(NAME)..."

	# Create folders
	@mkdir -p papers/$(NAME)/sections
	@mkdir -p papers/$(NAME)/figures
	@mkdir -p papers/$(NAME)/bibliography

	# Copy required template files
	@echo "Copying template files..."
	@cp $(TEMPLATE_DIR)/sn-article.tex papers/$(NAME)/$(MAIN_TEX)
	@cp $(TEMPLATE_DIR)/sn-*.bst papers/$(NAME)/
	@cp $(TEMPLATE_DIR)/sn-jnl.cls papers/$(NAME)/
	@cp $(TEMPLATE_DIR)/empty.eps papers/$(NAME)/
	@cp $(TEMPLATE_DIR)/sn-bibliography.bib papers/$(NAME)/$(NAME).bib

	# Create empty section files
	@touch papers/$(NAME)/sections/introduction.tex
	@touch papers/$(NAME)/sections/methods.tex
	@touch papers/$(NAME)/sections/results.tex
	@touch papers/$(NAME)/sections/discussion.tex

	# Create project-level Makefile for LaTeX compilation
	@echo "\
# Project-level Makefile for paper $(NAME)\n\
\n\
PROJECT := $(NAME)\n\
MAIN_TEX := main.tex\n\
SECTIONS_DIR := sections\n\
FIGURES_DIR := figures\n\
BIB_DIR := bibliography\n\
BIB_FILE := $(NAME).bib\n\
\n\
.PHONY: all compile clean\n\
\n\
all: compile\n\
\n\
compile:\n\
\t@echo 'Compile LaTeX paper for $(NAME)'\n\
\t@pdflatex $(MAIN_TEX)\n\
\t@bibtex $(basename $(MAIN_TEX)) || true\n\
\t@pdflatex $(MAIN_TEX)\n\
\t@pdflatex $(MAIN_TEX)\n\
\n\
clean:\n\
\trm -f *.aux *.log *.bbl *.blg *.toc *.out *.lof *.lot\n\
" > papers/$(NAME)/Makefile

	@echo "✓ Paper project $(NAME) created successfully!"
