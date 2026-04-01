# Paper build system
# Usage:
#   make          - Build PDF
#   make clean    - Remove build artifacts
#   make figures  - Collect figures from benchmark results
#   make all      - Figures + PDF

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all pdf figures clean

all: figures pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN) || true
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

figures:
	bash generate_figures.sh

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).fls \
	      $(MAIN).fdb_latexmk $(MAIN).synctex.gz
