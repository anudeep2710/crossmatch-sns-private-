# Makefile for IEEE Research Paper Compilation

# Main paper file
MAIN = enhanced_cross_platform_user_identification
OUTDIR = output

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean paper help

# Default target
all: paper

# Compile the research paper
paper: $(OUTDIR)/$(MAIN).pdf

$(OUTDIR)/$(MAIN).pdf: $(MAIN).tex references.bib
	@mkdir -p $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(BIBTEX) $(OUTDIR)/$(MAIN)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	@echo "Paper compiled successfully: $(OUTDIR)/$(MAIN).pdf"

# Quick compile (single pass)
quick:
	@mkdir -p $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	@echo "Quick compile completed: $(OUTDIR)/$(MAIN).pdf"

# Clean auxiliary files
clean:
	rm -rf $(OUTDIR)/*.aux $(OUTDIR)/*.log $(OUTDIR)/*.bbl $(OUTDIR)/*.blg
	rm -rf $(OUTDIR)/*.toc $(OUTDIR)/*.out $(OUTDIR)/*.fls $(OUTDIR)/*.fdb_latexmk
	@echo "Auxiliary files cleaned"

# Clean all generated files
cleanall:
	rm -rf $(OUTDIR)
	@echo "All generated files cleaned"

# Help
help:
	@echo "Available targets:"
	@echo "  all    - Compile the research paper (default)"
	@echo "  paper  - Compile the research paper"
	@echo "  quick  - Quick compile (single pass, no bibliography)"
	@echo "  clean  - Remove auxiliary files"
	@echo "  cleanall - Remove all generated files"
	@echo "  help   - Show this help message"
