# Method note

`method.tex` is the LaTeX source; `method.pdf` is the compiled note. The TikZ
figures are original explanatory diagrams, with editable sources in `figures/`.
The citations in `references.bib` link to primary papers or the official
challenge page. No author or affiliation has been assumed.

Build from this directory with a standard TeX Live installation:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build method.tex
cp build/method.pdf method.pdf
```

The build uses pdfLaTeX, BibTeX, TikZ, Latin Modern, and standard LaTeX packages.
There are no shell-escape or network requirements. To regenerate page images
for visual checking (Poppler required):

```sh
pdftoppm -r 120 -png method.pdf build/page
```

The public calibration table describes the named shared profile, rather than
secret-model performance. Keep `selection.tex` synchronized with the production
selector when changing a submission policy. Build intermediates stay in
`build/`; distribute the PDF, LaTeX sources, bibliography, and figure sources.
