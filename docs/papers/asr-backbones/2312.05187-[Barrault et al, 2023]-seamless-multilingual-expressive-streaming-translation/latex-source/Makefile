
pdf: *.tex bibliography.bib
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

all: pdf 

clean:
	rm -f *~ *.aux *.log *.bbl *.blg main.pdf

