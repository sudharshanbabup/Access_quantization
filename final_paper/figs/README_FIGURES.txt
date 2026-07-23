FIGURES
=======
All data plots are the real figures supplied by the authors.
overview.pdf            Review-requested TikZ redesign of Figure 1 (source: ../figs-src/overview.tex)
overview_useroriginal.pdf  The original overview you provided (kept in case you prefer it)
concept.pdf             Causal-chain conceptual figure (source: ../figs-src/concept.tex)
All other *.pdf         Your experimental plots (synthetic + CIFAR-10/ImageNette).

To regenerate the two TikZ diagrams:
  cd figs-src && pdflatex overview.tex && pdflatex concept.tex && cp overview.pdf concept.pdf ../figs/
