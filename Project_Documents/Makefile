# Makefile for LaTeX IEEE paper compilation

# Main paper
MAIN = paper
SUPP = supplementary
FIGS = figures

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Directories
OUTDIR = output
FIGDIR = figures

.PHONY: all clean main supplementary figures help

# Default target
all: main supplementary figures

# Compile main paper
main: $(OUTDIR)/$(MAIN).pdf

$(OUTDIR)/$(MAIN).pdf: $(MAIN).tex references.bib
	@mkdir -p $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(BIBTEX) $(OUTDIR)/$(MAIN)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	@echo "Main paper compiled successfully: $(OUTDIR)/$(MAIN).pdf"

# Compile supplementary material
supplementary: $(OUTDIR)/$(SUPP).pdf

$(OUTDIR)/$(SUPP).pdf: $(SUPP).tex
	@mkdir -p $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(SUPP).tex
	$(LATEX) -output-directory=$(OUTDIR) $(SUPP).tex
	@echo "Supplementary material compiled successfully: $(OUTDIR)/$(SUPP).pdf"

# Compile figures
figures: $(OUTDIR)/$(FIGS).pdf

$(OUTDIR)/$(FIGS).pdf: $(FIGS).tex
	@mkdir -p $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(FIGS).tex
	@echo "Figures compiled successfully: $(OUTDIR)/$(FIGS).pdf"

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

# Check LaTeX installation
check:
	@echo "Checking LaTeX installation..."
	@which $(LATEX) > /dev/null && echo "✓ pdflatex found" || echo "✗ pdflatex not found"
	@which $(BIBTEX) > /dev/null && echo "✓ bibtex found" || echo "✗ bibtex not found"
	@echo "LaTeX check completed"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Compile main paper, supplementary, and figures"
	@echo "  main         - Compile main paper only"
	@echo "  supplementary - Compile supplementary material only"
	@echo "  figures      - Compile figures only"
	@echo "  quick        - Quick compile (single pass, no bibliography)"
	@echo "  clean        - Remove auxiliary files"
	@echo "  cleanall     - Remove all generated files"
	@echo "  check        - Check LaTeX installation"
	@echo "  help         - Show this help message"

# Archive for submission
archive: all
	@mkdir -p submission
	cp $(OUTDIR)/$(MAIN).pdf submission/
	cp $(OUTDIR)/$(SUPP).pdf submission/
	cp $(OUTDIR)/$(FIGS).pdf submission/
	cp $(MAIN).tex submission/
	cp $(SUPP).tex submission/
	cp $(FIGS).tex submission/
	cp references.bib submission/
	cp README_SUBMISSION.md submission/
	tar -czf submission.tar.gz submission/
	@echo "Submission archive created: submission.tar.gz"
