
all: paper.pdf
	echo cp paper.pdf /mnt/c/Users/happyjimmy/Desktop/

paper.pdf: IEEEtran.bst paper.tex mybib.bib
	pdflatex paper
	bibtex paper || true
	pdflatex paper
	pdflatex paper

clean:
	@rm -f paper.aux paper.bbl paper.blg paper.dvi \
	paper.log paper.pdf paper.ps paper.out
