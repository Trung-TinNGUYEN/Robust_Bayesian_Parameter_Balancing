# Reproducing the article

This guide lists the commands behind every supporting table and figure of the article. Run
them from the repository root, after installing the environment as the README describes.

## Before you start

- Every script reads the record-level inputs in `output/<pathway>/data/` and writes its
  results under `output/` (most to `output/<pathway>/results/revision/`), or under `figures/`
  for the figure scripts. A script that reads another script's output says so below.
- Most scripts finish within minutes with 8 threads. The longest are the cross-validation of
  the error-scale constructions (`run_revision_hetero.jl`) and the misspecification grid
  (`run_revision_misspecification.jl`), which can take an hour or more each.
- Run one multi-threaded script at a time. Run the timing scripts alone on an idle machine,
  because they measure wall-clock time.
- Many scripts accept the environment variable `PB_REV_PATHWAYS`, a comma-separated list of
  pathways (`glycolysis`, `tca`, `pentose_phosphate`, `shikimate`). Its default differs between
  scripts, and the commands below set it where the article needs other pathways than the
  default. `PB_REV_SUFFIX` appends a label to the output file names, so that a run with other
  settings does not overwrite the published one.

## Inputs

The record-level inputs are included. To rebuild them from `data/` and compare them with the
included files:

```bash
for pw in glycolysis tca pentose_phosphate shikimate; do
  julia --project=. scripts/build_pathway_inputs.jl $pw --outdir=/tmp/rebuild --verify
done
```

## The released parameter sets (S1, S2, S26 and S27 Tables)

```bash
julia --project=. scripts/run_revision_paramset.jl                            # output/paramset/
PB_REV_EXCLUDE_MUTANTS=1 julia --project=. scripts/run_revision_paramset.jl   # output/paramset_wildtype/
```

S2 and S27 Tables are `balanced_parameters_by_organism.csv` of the two folders. S1 and S26
Tables are their `balanced_parameters.csv` with three more columns, the range of each
quantity's own records and a flag for a consensus value outside that range, which
`run_revision_record_range.jl` adds to the tables in `supporting_tables/`:

```bash
cp output/paramset/balanced_parameters.csv supporting_tables/S1_Table.csv
cp output/paramset_wildtype/balanced_parameters.csv supporting_tables/S26_Table.csv
julia --project=. scripts/run_revision_record_range.jl
```

## Held-out evaluation on the curated records

```bash
# Real-data cells of the leave-one-source-out evaluation. Run the three-cell fit and the
# bootstrap first: both fits write the same files for glycolysis and the TCA cycle.
PB_REV_CELLS=gaussian_Fix,gaussian_RE_org,gaussian_RE_orgtype \
  julia --project=. --threads=8 scripts/run_revision_realdata.jl
julia --project=. scripts/run_revision_inference.jl                  # S3 and S4 Tables
julia --project=. --threads=8 scripts/run_revision_realdata.jl       # S10 Table, and S7 for these pathways
PB_REV_PATHWAYS=pentose_phosphate julia --project=. --threads=8 scripts/run_revision_realdata.jl   # S12 Table
PB_REV_PATHWAYS=shikimate julia --project=. --threads=8 scripts/run_revision_realdata.jl           # S13 Table

# The recalibrated record error scale
julia --project=. --threads=8 scripts/run_revision_sigmacal.jl       # S11 Table
PB_REV_PATHWAYS=pentose_phosphate julia --project=. --threads=8 scripts/run_revision_sigmacal.jl   # S14 Table
PB_REV_PATHWAYS=shikimate julia --project=. --threads=8 scripts/run_revision_sigmacal.jl           # S15 Table

# Error-scale constructions and offset transfer (four pathways)
julia --project=. --threads=8 scripts/run_revision_hetero.jl         # S16, S17 and S18 Tables
julia --project=. scripts/run_revision_hetero_inference.jl           # S5 Table (reads the run above)
julia --project=. --threads=8 scripts/run_revision_transfer.jl       # S19 Table
julia --project=. --threads=8 scripts/run_revision_transfer_diag.jl  # S20 Table
julia --project=. --threads=8 scripts/run_revision_conformal.jl      # organism-averaged coverage plotted in Fig 4
```

## Misspecification and the records

```bash
julia --project=. --threads=8 scripts/run_revision_misspecification.jl   # S8 and S9 Tables
julia --project=. scripts/run_revision_mutant.jl                         # S6 Table
```

## Downstream use and the external check

```bash
PB_EB_KAPPA0=1e-9 PB_REV_SUFFIX=_priorheld julia --project=. --threads=8 scripts/run_revision_downstream.jl   # S22 and S24 Tables
PB_EB_KAPPA0=1e-9 PB_REV_SUFFIX=_priorheld julia --project=. --threads=4 scripts/run_revision_dynamic.jl      # S23 Table
julia --project=. scripts/run_revision_ecoli_holdout.jl            # S21 Table: E. coli held out
julia --project=. scripts/run_revision_invivo_check.jl             # S21 Table: in vivo check of those predictions
```

`run_revision_ecoli_holdout.jl` checks that its *E. coli* fold reproduces the per-record files
of `run_revision_realdata.jl` and `run_revision_hetero.jl` on all four pathways, so run those
first (above).

`data/external/davidi2016_kmax_matched.csv` holds the in vivo turnover numbers of the check.
`scripts/build_davidi_kmax_match.py` rebuilds it from the source workbook, which is not
redistributed (see `data/external/README.md`).

## Checks of predicted rates (S29 to S31 Tables)

```bash
julia --project=. --threads=8 scripts/run_revision_rate_holdout.jl      # held-out organisms
julia --project=. --threads=8 scripts/run_revision_rate_simulation.jl   # simulation with known parameters
PB_RSIM_OUTER_MULT=4 PB_REV_SUFFIX=_em4 julia --project=. --threads=8 scripts/run_revision_rate_simulation.jl
PB_RSIM_TRUTH=draw PB_REV_SUFFIX=_truthdraw julia --project=. --threads=8 scripts/run_revision_rate_simulation.jl
julia --project=. scripts/run_revision_rate_invivo.jl                    # in vivo E. coli rates
PB_RIV_CONC_SD=0.2 PB_RIV_OUT_SUFFIX=_concsd02 julia --project=. scripts/run_revision_rate_invivo.jl
PB_RIV_CONC_SD=0.5 PB_RIV_OUT_SUFFIX=_concsd05 julia --project=. scripts/run_revision_rate_invivo.jl
python3 -I scripts/build_rate_check_tables.py --out supporting_tables   # S29, S30 and S31 Tables
```

The three scripts write to `output/rate_checks/`. The simulation reads the released set in
`output/paramset/` and the record scale and between-organism variance that
`run_revision_downstream.jl` estimates with its default settings (the prior mean updated, as in
the cross-validation), so run that script first, once without the settings above. The in vivo check reads the
draws of `run_revision_ecoli_holdout.jl`. The held-out check compares each refitted fold with the
per-record files of `run_revision_realdata.jl` and `run_revision_hetero.jl` when they exist, and
skips that comparison otherwise. Its inputs `data/external/ecoli_invivo_kapp.csv` and
`data/external/ecoli_invivo_concentrations.csv` are rebuilt from two MIT-licensed repositories by
`scripts/build_invivo_rate_inputs.py` (see `data/external/README.md`).

## Computation

```bash
julia --project=. scripts/run_revision_sparse_scaling.jl                   # S28 Table
julia --project=. scripts/run_revision_genomescale.jl                      # S25 Table (reads data/yeast-GEM.xml)
julia --project=. scripts/run_revision_engine_timing.jl                    # fit times of S1 Text, Section I
julia --project=. scripts/run_computational_scaling.jl                     # scaling tables and S1 Text Fig E
for t in 1 2 4; do julia --project=. --threads=$t scripts/run_parallel_scaling_benchmark.jl; done   # thread scaling
```

The timing scripts write to `output/scaling/`, where the tables measured for the article are
included. S1 Text Fig E can be redrawn from the included tables without timing anything:

```bash
julia --project=. -e 'include("scripts/run_computational_scaling.jl"); regenerate_scaling_plots_from_csv(method=:collapsed)'
```

## Figures

```bash
PB_FIG1_CONCISE=1 julia --project=. scripts/make_fig1_overview.jl   # Fig 1
PB_FIG_ONLY=Fig2,Fig3,Fig4,Fig5 julia --project=. scripts/make_revision_figures.jl   # Figs 2 and 3, S1 Text Figs A and B
PB_FIG_ONLY=Fig7 PB_FIG_DOWNSTREAM_SUFFIX=_priorheld julia --project=. scripts/make_revision_figures.jl   # Fig 5
julia --project=. scripts/make_fig8_recalibration.jl  # Fig 4
julia --project=. scripts/make_concise_figures.jl     # S1 Text Figs C and D
```

`make_revision_figures.jl` reads the outputs of the misspecification, real-data and
downstream scripts (for Fig 5, the run with the prior mean held), and `make_fig8_recalibration.jl` those of the real-data, recalibration and
conformal scripts on all four pathways. `make_fig1_overview.jl` reads the real-data and
recalibration summaries of all four pathways for its two coverage ranges.
`make_concise_figures.jl` reads S17 and S18 Tables.
The file names follow the numbering of an earlier version of the manuscript:

| Figure | File |
| --- | --- |
| Fig 1 | `figures/figure/Fig1_overview.png` |
| Fig 2 | `figures/figure/Fig2_residual_diagnostics.pdf` |
| Fig 3 | `figures/figure/Fig3_misspecification.pdf` |
| Fig 4 | `figures/figure/Fig7_recalibration.pdf` |
| Fig 5 | `figures/figure/Fig8_downstream.pdf` |
| S1 Text Fig A | `figures/figure/Fig4_calibration_curves.pdf` |
| S1 Text Fig B | `figures/figure/Fig5_loso_calibration.pdf` |
| S1 Text Fig C | `figures/figure/Fig5_conditional_repair.pdf` |
| S1 Text Fig D | `figures/figure/Fig6_calibration_curves.pdf` |
| S1 Text Fig E | `output/scaling/plots/scaling_d.pdf`, `scaling_n.pdf`, `scaling_G.pdf` |

## From outputs to supporting tables

Where a script writes one file per pathway, its table stacks the files in the order
glycolysis, TCA cycle, pentose phosphate pathway, shikimate pathway (for the pathways the
script was run on). `docs/supporting_tables.md` describes the columns.
`python3 -I scripts/build_short_tables.py --out supporting_tables` makes every table that
comes from real-data fits (S1 to S5, S7, S10 to S21, S26 and S27) from the outputs in this way,
in the published byte format. Run the record-range step for S1 and S26 after it. The tables
that need more than stacking are:

- **S1 and S26**: the record-range step above.
- **S7**: the files `realdata_residuals_<pathway>.csv`, with a pathway column added in front.
- **S22 to S24**: `python3 -I scripts/build_propagation_tables.py --out supporting_tables --suffix _priorheld`,
  which stacks the outputs with the prior mean held in the order of the tables, those under the
  half-decade convention for glycolysis, then for the TCA cycle, then those of the pre-specified
  default for glycolysis and for the TCA cycle.
- **S21**: one row per matched reaction and prediction, joining
  `data/external/davidi2016_kmax_matched.csv`, the files `ecoli_holdout_quantities_<pathway>.csv`
  and `output/invivo/invivo_check_rows.csv`, with values given to four significant digits.
- **S28**: the three files `output/scaling/sparse_scaling*.csv`, with their size columns
  renamed `D_or_m`, `n_records` and `G_or_k` and their time column renamed `time_s`.
- **S29 to S31**: `scripts/build_rate_check_tables.py`, above. S29 is
  `output/rate_checks/rate_holdout_summary.csv`, S30 stacks the three simulation summaries with
  the medians of the fitted hyperparameters joined, and S31 stacks the three in vivo runs.

## Other scripts

- Validation with synthetic ground truth: `validate_hetero.jl` (S1 Text, Section D.4) and
  `validate_transfer.jl` (Section E.5).
- `run_all_paper_experiments.jl` runs the leave-one-record-out evaluation quoted in S1 Text,
  Section G.1, together with the earlier experiments on glycolysis and the TCA cycle.
