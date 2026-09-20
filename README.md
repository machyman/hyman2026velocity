# hyman2026velocity

**Velocity sorting biases the equilibrium in Array-RQMC for binary-collision models**

James M. Hyman, Department of Mathematics, Tulane University
`mhyman@tulane.edu` · ORCID [0000-0001-5247-5794](https://orcid.org/0000-0001-5247-5794)

Companion code, data and manuscript source. Manuscript version **v1_68_0**.
arXiv identifier: **TBD**.

---

## What this is about

Array-RQMC sorts the states of many trajectories at each step and aligns the
sorted states with a low-discrepancy point set. This repository accompanies a
study of one way of bringing that idea to a binary-collision model of the Kac
type: ordering the particles of a single system by their own state, and letting
that order choose which particles collide.

The sort then enters the dynamics rather than the assignment of randomness, and
the equilibrium moves. The paper proves this for one collision step and
measures the consequences for the chain.

The obstruction is to that adaptation and not to Array-RQMC itself. A
construction that sorts independent copies of a chain, leaving each copy's
transition law untouched, is unaffected.

---

## Repository structure

```
manuscript/     the compiled paper
  hyman2026velocity_v1_68_0.pdf     manuscript, 29 pages

code/           every generator behind a figure, table or reported number
  kac_fig1_spectrum_v1_0_0.py       Figure 1, the mechanism
  kac_fig2_finiteN_v1_0_0.py        Figure 2, plotting
  kac_manuscript_data_v1_0_0.py     Figure 2 data; Section 5 counterexample
  kac_fig3_tradeoff_v1_0_0.py       Figure 3, plotting
  kac_phasef_v2_1_0.py              Tables 3 and 4 data
  kac_remaining_v1_0_0.py           Tables 2 and 5 data
  kac_onestep_test_v1_0_0.py        Table 1, the one-step test
  kac_onestep_curve_v1_0_0.py       Table 1, curve rule on the applied load
  kac_cascade_monotone_v1_0_0.py    Theorem 9.1, the monotone cascade
  kac_fig4_condensate_v1_0_0.py     Figure 4, the condensate
  kac_general_law_v1_0_0.py         Section 9 scope remark; necessity controls
  build_colab_wrappers.py           builds the notebooks below
  kac_theory_*, kac_reduction_*,    exploratory and cross-checking scripts
  kac_scope_*, kac_hierarchy_*,     retained for provenance; not load-bearing
  kac_compliance_*, kac_tradeoff_*,
  kac_n4_*, w2_*,                   the equation (11) certification attempt
  artifact_sweep_*, generator_template_*

data/           generated CSVs and the campaign archive
  onestep_test.csv                  Table 1
  remaining_results.csv             Tables 2 and 5
  phasef_part1.csv, phasef_part2.csv            Tables 3 and 4, full precision
  phasef_part1_rounded.csv, phasef_part2_rounded.csv   presentational, authored
  manuscript_ks_sweep.csv           Figure 2
  spectrum_N6.csv                   Figure 1
  cascade_monotone.csv              Theorem 9.1
  fig04_condensate.csv              Figure 4
  general_law.csv                   Section 9 scope remark
  discrepancy_campaign_R4_FULL_results_2026-08-12.zip   3.3 MB; the registered
                                    t=0 particle loads used by Sections 7 and 8

notebooks/      Colab wrappers, one per generator; see the table below
                built by code/build_colab_wrappers.py, not hand-written

campaign/       the code behind the particle loads used in Sections 7 and 8
  discrepancy_campaign_v1_2_0.py    the campaign; contains the projection
                                    defined in Section 6
  pic1d1v_gate_v1_4_0.py            the PIC gate
  analyze_campaign_r4_2026-08-12.py analysis of the R4 results
  WoS_PIC_Discrepancy_Campaign_Colab_2026-08-12_v4_executed.ipynb
                                    the executed Colab run that produced
                                    data/discrepancy_campaign_R4_FULL_results_2026-08-12.zip
  Superseded versions v1_0_0 to v1_1_1 of the campaign and gate are not
  included; only the versions that produced the deposited results are.

figures/        fig01_spectrum.pdf, fig02_finite_N.pdf, fig03_tradeoff.pdf

tests/          the checkers described below

MANUSCRIPT_STATUS.yaml   the reproducibility registry: every artifact, its
                         generator, inputs, outputs and SHA-256
```

---

## Run it in Colab

Every generator has a wrapper notebook. Each fetches this repository, checks the
environment, prints the generator, runs it unmodified, and offers the outputs
for download. No local install.

| Notebook | Wraps | Reproduces | Runtime |
|---|---|---|---|
| [`01_one_step_test.ipynb`](notebooks/01_one_step_test.ipynb) | `kac_onestep_test_v1_0_0.py` | Lemma 4.2, Table 1 | ~1 min |
| [`02_one_step_curve.ipynb`](notebooks/02_one_step_curve.ipynb) | `kac_onestep_curve_v1_0_0.py` | Lemma 4.2, Section 7 | ~1 min |
| [`03_figure1_mechanism.ipynb`](notebooks/03_figure1_mechanism.ipynb) | `kac_fig1_spectrum_v1_0_0.py` | Lemmas 4.1, 4.2, 4.4; Figure 1 | ~10 s |
| [`04_figure2_data.ipynb`](notebooks/04_figure2_data.ipynb) | `kac_manuscript_data_v1_0_0.py` | Figure 2, Section 5 | ~1 min |
| [`05_figure2_plot.ipynb`](notebooks/05_figure2_plot.ipynb) | `kac_fig2_finiteN_v1_0_0.py` | Figure 2 | ~10 s |
| [`06_figure3_plot.ipynb`](notebooks/06_figure3_plot.ipynb) | `kac_fig3_tradeoff_v1_0_0.py` | Figure 3, Section 6 | ~10 s |
| [`07_tables3_4.ipynb`](notebooks/07_tables3_4.ipynb) | `kac_phasef_v2_1_0.py` | Tables 3 and 4 | ~3 min |
| [`08_tables2_5.ipynb`](notebooks/08_tables2_5.ipynb) | `kac_remaining_v1_0_0.py` | Tables 2 and 5, Sections 7 and 8 | ~30 s |
| [`09_cascade_monotone.ipynb`](notebooks/09_cascade_monotone.ipynb) | `kac_cascade_monotone_v1_0_0.py` | Theorem 9.1 | ~1 min |
| [`10_figure4_condensate.ipynb`](notebooks/10_figure4_condensate.ipynb) | `kac_fig4_condensate_v1_0_0.py` | Theorem 9.2, Figure 4 | ~2 min |
| [`11_general_law_controls.ipynb`](notebooks/11_general_law_controls.ipynb) | `kac_general_law_v1_0_0.py` | Theorem 9.2 and the scope remark, Section 9 | ~3 min |

The notebooks are generated by `code/build_colab_wrappers.py` rather than
maintained by hand, so that eleven near-identical files cannot drift apart.
Regenerate them after changing a generator.

## Reproducing the results locally

```bash
pip install -r requirements.txt
cd code
python3 kac_onestep_test_v1_0_0.py        # Table 1        ~ 1 min
python3 kac_remaining_v1_0_0.py           # Tables 2 and 5 ~ 30 s
python3 kac_phasef_v2_1_0.py              # Tables 3 and 4 ~ 3 min
python3 kac_manuscript_data_v1_0_0.py     # Figure 2 data  ~ 1 min
SOURCE_DATE_EPOCH=1786822750 python3 kac_fig1_spectrum_v1_0_0.py
SOURCE_DATE_EPOCH=1786822750 python3 kac_fig2_finiteN_v1_0_0.py
SOURCE_DATE_EPOCH=1786822750 python3 kac_fig3_tradeoff_v1_0_0.py
```

`kac_remaining_v1_0_0.py` and `kac_onestep_curve_v1_0_0.py` extract the particle
loads they need from `data/discrepancy_campaign_R4_FULL_results_2026-08-12.zip`;
copy that archive alongside them, or run them from `data/`.

`SOURCE_DATE_EPOCH` pins the timestamp matplotlib writes into a PDF. Without it
the figures are identical in content and differ in that one field.

The LaTeX source is not in this repository. It is distributed separately as an
Overleaf package, `hyman2026velocity_COR_<date>.zip`, which carries the source,
the bibliography, the `\input` file and the three figures and compiles to the
PDF here without modification. This repository carries the compiled paper and
everything needed to reproduce the numbers and figures in it.

---

## What is checked, and what is not

The `tests/` directory holds the checkers used during preparation. Each one
refuses or fails on input it cannot actually verify, and each ships with a
negative control that plants the defect it looks for and requires it to be
caught.

```bash
cd tests
python3 cleanroom_check.py --run     # do the generators regenerate their outputs?
python3 verify_rounded.py            # do the presentational CSVs match their sources?
python3 integrity_check.py           # do all registered files match their hashes?
```

`verify_rounded.py` runs against this layout unchanged, resolving its inputs
from `data/`. `verify_manuscript.py` and `scope_scan.py` operate on LaTeX source
and so apply to the Overleaf package rather than to this repository. Two do
not run here at all. `integrity_check.py` and
`cleanroom_check.py` read `MANUSCRIPT_STATUS.yaml`, whose paths record the flat
working directory the manuscript was prepared in rather than this grouped tree,
so they report the grouped files as missing. Reconstructing that flat layout is
left undocumented here because the recipe was not tested, and an untested
recipe in a reproducibility README is worse than none.

They are shipped because they are the instruments the results were checked
with, not because they run unmodified against this layout.

**Three artifacts are authored, not generated,** and are marked so in
`MANUSCRIPT_STATUS.yaml`: `phasef_part1_rounded.csv`,
`phasef_part2_rounded.csv` and `phasef_tables.tex`. They are not roundings of
the generated files — they select columns, fix digits and add a replicate
count — and the derivation was never recorded, so no script here reproduces
them. `verify_rounded.py` checks instead that every value they carry agrees
with the generated source to the digits shown. `cleanroom_check.py` therefore
reports **4 of 7** generators writing all declared outputs, permanently; that
is the honest figure and is not engineered upward.

**Two figures have reconstructed plotting scripts.** The originals were never
saved. `kac_fig2_finiteN_v1_0_0.py` and `kac_fig3_tradeoff_v1_0_0.py` were
rebuilt from the registered data and reproduce the content, the series and the
axes, but not the exact legend and annotation placement of the deposited PDFs.
The deposited PDFs are the figures in the manuscript.

---

## Data

The particle loads in `data/discrepancy_campaign_R4_FULL_results_2026-08-12.zip`
are 3.3 MB and are included in the repository rather than externally archived.
Every other dataset here is procedurally generated: the generation script with
its seed is the dataset, and the CSVs are convenience copies.

---

## Development Tools

This research used AI-assisted workflows for manuscript preparation and review.
See the manuscript's Acknowledgments section for the full disclosure.
All mathematical content, proofs, code, and computations are the sole work of
the author(s).

---

## How to cite

See `CITATION.cff`. The arXiv identifier is not yet assigned; `TBD` in that file
and in this README is replaced once the preprint is posted.

---

## Updating this repository through the GitHub web interface

The intended workflow. Git on the command line is the alternative, not the
default.

1. Go to `https://github.com/machyman/hyman2026velocity`.
2. **Add files → Upload files**, then drag the contents of the release folder
   in. Drag the *contents*, not the folder itself: dropping a folder creates a
   nested directory and every path in this README and in the manuscript breaks.
3. Dotfiles are hidden by default in macOS Finder. Press `Cmd-Shift-.` to show
   them before selecting, or files such as `.gitignore` are silently omitted.
4. Uploading a file that already exists replaces it. Uploading does **not**
   delete files that are absent from the upload; removals must be done
   explicitly through the web interface.
5. Commit directly to `main` with a message naming the manuscript version, for
   example `v1_68_0: section 9 generators, Figure 4, Figure 3 fix`.
6. After the push, re-download the repository as a zip and compare it against
   what you uploaded. A push that reports success and a repository that matches
   what you meant to send are different claims.
