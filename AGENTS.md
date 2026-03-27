## Accuracy standard
- Do not leave undefined symbols in equations. Define symbols at first use whenever reasonably possible.
- If a symbol was introduced much earlier and is easy to forget, briefly remind the reader locally instead of forcing backtracking.
- Do not use vague words such as "schematic" unless the note explicitly explains what is being suppressed and why.
- If a formula is imported from continuum lightcone string theory rather than derived in the note, say so explicitly.
- If a quantity is not fully fixed because of missing conventions or an unresolved calculation, say that plainly.
- Distinguish carefully between:
  - lattice-defined quantities and continuum target quantities,
  - local vertex data and global sewn-diagram data,
  - vertex labels and transverse/spinor indices,
  - exact formulas, convention-dependent formulas, and deferred formulas.

## Reader burden
- Optimize for low cognitive load.
- Every displayed equation should be understandable from nearby text.
- Prefer explicit sentences like "Here X means ..." or "This is not yet ..." over relying on expert inference.
- When there is a nontrivial basis choice, state the basis, the physical subspace, and what is actually inverted or solved.
- When the note makes an approximation, interpolation, refinement, or continuum-limit claim, state the precise status and limitations.

## Figures
- Add figures for geometric constructions, bookkeeping maps, and local interaction-region data when they materially help.
- Keep figures visually clean. Do not place long equations inside drawings if they cause overlap or clutter.
- Use captions to explain the mathematical role of the figure, not just its visual content.

## House style for this note
- Prefer precision over brevity.
- State conventions explicitly.
- Use consistent notation across sections once a symbol is chosen.
- If a notation change is made in one section, check downstream sections for consistency.
- When the note claims a formula is "exact," ensure the exact domain of validity is stated.

## TeX file organization
- Keep one folder per TeX note at repo root, named after the note basename (for example, `/notes/NCOS/` for `NCOS.tex`).
- Store the note source and all generated artifacts in that folder (`.tex`, `.pdf`, `.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.out`, `.synctex.gz`, `.bbl`, `.blg`, `.run.xml`, `-blx.bib`).
- When creating a new note, create its folder first inside /notes/ and place the TeX source inside it instead of writing note files at repo root.
