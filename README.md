# RobustBayesianParameterBalancing

Code and data for the article

> TrungTin Nguyen, Binh H. Ho, Michael Pan, Jennifer A. Flegg, Michael J. McDonald and
> Christopher Drovandi. *Hierarchical Bayesian modelling for robust parameter balancing in
> kinetic models of metabolism.*

Parameter balancing combines measured kinetic and thermodynamic constants into one parameter
set that satisfies the thermodynamic (Haldane) relationships of a reaction network. Standard
parameter balancing treats every record as an independent measurement with an assumed error.
This repository provides a hierarchical version, in which the records from one source
organism share an offset for each quantity type and the record error scale is estimated from
the data, together with everything needed to reproduce the article:

- the Julia package `RobustBayesianParameterBalancing` and a command-line tool;
- the curated inputs for four pathways of the Yeast9 reconstruction of *Saccharomyces
  cerevisiae* (glycolysis, the tricarboxylic acid cycle, the pentose phosphate pathway and the
  shikimate pathway), built from BRENDA records and eQuilibrator formation energies;
- the released balanced parameter sets for these pathways, with consensus intervals,
  intervals for an organism without records, organism-specific intervals and the joint
  posterior of the basis vector;
- the scripts behind the results of the article, and its supporting tables S1 to S28.

The version analysed in the article is archived on Zenodo (DOI to be added on acceptance).

## Contents

| Path | Contents |
| --- | --- |
| `src/` | The package: the module `RobustBayesianParameterBalancing.jl` and the five files it includes (`parameter_balancing.jl`, `parameter_balancing_core.jl`, `prediction_utils.jl`, `hierarchy_utils.jl`, `workflow_interop.jl`). The other files of `src/` are shared by the scripts: `metabolic_reconstruction.jl` builds the pathway systems from the reconstruction, `experiment_runners.jl` runs the cross-validation and simulation experiments, `mutant_flags.jl` flags records of site-directed mutants, and the remaining files are plotting helpers. |
| `bin/parameter-balance.jl` | The command-line tool (see [Balancing your own data](#balancing-your-own-data)). |
| `examples/` | A minimal use of the package, and a script that runs the released specification of the article with the command-line tool. |
| `scripts/` | The scripts behind the results of the article (see [Reproducing the article](#reproducing-the-article)). |
| `data/` | Inputs: kinetic records derived from BRENDA (`data/brenda/`), formation energies from eQuilibrator (`data/equilibrator/`), the Yeast9 reconstruction (`data/yeast-GEM.xml`), the prior and option tables of standard parameter balancing (`data/pb_prior.tsv`, `data/pb_options.tsv`) and in vivo turnover numbers, apparent rates and metabolite concentrations of *E. coli* for the external checks (`data/external/`, with its own README). |
| `output/<pathway>/data/` | The record-level inputs of each pathway, built from `data/` by `scripts/build_pathway_inputs.jl`. All analysis scripts read them. |
| `output/paramset/` | The released balanced parameter set, fitted to all records. |
| `output/paramset_wildtype/` | The same set fitted to the records that are not flagged as site-directed mutants. |
| `output/scaling/` | Timing and scaling tables quoted in S1 Text. |
| `supporting_tables/` | S1 to S28 Tables of the article, as comma-separated files. |
| `docs/supporting_tables.md` | What each supporting table holds and what its columns mean. |
| `docs/software_notes.md` | The package entry points and command-line options, the use of the released parameter set with a worked example, and the scripts behind passages of S1 Text. |
| `docs/reproducing.md` | The commands behind every supporting table and figure of the article. |
| `Project.toml`, `Manifest.toml` | The Julia environment, recorded with Julia 1.12.5. |
| `CITATION.cff`, `.zenodo.json`, `LICENSE` | Citation and archive metadata, and the licence. |

## Installation

Install Julia 1.12.5 (for example with [juliaup](https://github.com/JuliaLang/juliaup):
`juliaup add 1.12.5`), clone the repository, and instantiate the environment from its root:

```bash
git clone https://github.com/Trung-TinNGUYEN/Robust_Bayesian_Parameter_Balancing.git
cd Robust_Bayesian_Parameter_Balancing
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Check that the package loads and finds its bundled prior table:

```bash
julia --project=. -e 'using RobustBayesianParameterBalancing; println(length(load_parameter_balancing_prior().entries))'
```

All commands below are run from the repository root.

## Quick start

```bash
julia --project=. examples/basic_package_usage.jl          # balance two quantities with the package
julia --project=. bin/parameter-balance.jl --help          # commands of the command-line tool
bash examples/released_specification.sh shikimate          # the released specification on one pathway
```

The last command exports the record-level inputs of a pathway as CSV files and fits the
released specification of the article with the command-line tool. It writes the posterior of
the basis vector, its covariance and the intervals for a previously unmeasured organism to
`pb_example_shikimate/`. Set `JULIA` to the Julia executable if `julia` is not on the path.

## Using the released parameter set

`output/paramset/` holds the balanced parameter set for the four pathways (S1 and S2 Tables):

- `balanced_parameters.csv`: for each curated quantity (148 in all), the posterior median and
  95% interval of the consensus value and of the value for an organism without records, in
  natural units (s^-1 for turnover numbers, mM for Michaelis constants), with the number of
  records and organisms behind it and the enzyme form of those records;
- `balanced_parameters_by_organism.csv`: the posterior for each organism with records;
- per pathway, the joint posterior of the basis vector (`basis_posterior_mean_<pathway>.csv`,
  `basis_posterior_cov_<pathway>.csv`, `basis_posterior_draws_<pathway>.csv`) and the map from
  basis coordinates to quantities (`basis_design_<pathway>.csv`). Any derived quantity, or
  any function of several, can be computed from the draws with the Haldane relationships
  holding exactly.

`output/paramset_wildtype/` holds the same files fitted to wild-type records only (S26 and
S27 Tables), the set to use for kinetic models of wild-type enzymes. `docs/software_notes.md`
works through an example (hexokinase in glycolysis) and explains when to use the consensus,
the unmeasured-organism and the organism-specific values. `docs/supporting_tables.md`
describes every column.

## Balancing your own data

The command-line tool reads numeric matrices and label files. The released specification of
the article is one command:

```bash
julia --project=. bin/parameter-balance.jl balance-hierarchical \
  --Q Q.csv --x x.csv --Cx Cx.csv --q-prior q_prior.csv --C-prior C_prior.csv \
  --group sources.csv --replicate-group replicates.csv --type types.csv \
  --error-scale replication --output balanced_parameters.csv
```

`--error-scale replication` estimates the record error scale from replicate records without
fitting a model. `--type` (one quantity-type label per record) estimates the between-source
variance for each quantity type, with the pooled value as the fallback. The prior mean is held
at the thermodynamic prior by default, and `--predict FILE` writes intervals for a previously
unmeasured source. The released specification has a random intercept for each
organism-by-type cell, so `sources.csv` should join the organism and the quantity type, for
example `escherichia coli|kcat`. `examples/export_cli_inputs.jl` writes these files for any of
the four pathways, and `docs/software_notes.md` lists all options.

The tool also exports the prior table (`prior-summary`), validates SBtab-like quantity tables
(`validate-table`), adds prior pseudo-observations (`pseudo-fill`), runs the closed-form
Gaussian update of standard parameter balancing (`balance-gaussian`) and writes a
balanced-parameter sidecar for an SBML model (`kineticize`). From Julia, the main entry points
are `load_parameter_balancing_prior`, `validate_pb_inputs`, `balance_parameters`,
`export_balanced_table` and `kineticize_sbml`, and analyses with an explicit fold structure use
`fit_fold_mcem`.

## Reproducing the article

### Inputs

The record-level inputs of the four pathways are included in `output/<pathway>/data/`, so no
script needs the raw BRENDA release. `scripts/build_pathway_inputs.jl` rebuilds them from
`data/` and, with `--verify`, compares the result with the included files:

```bash
julia --project=. scripts/build_pathway_inputs.jl glycolysis --outdir=/tmp/rebuild --verify
```

The Python scripts in `data/brenda/` and `data/equilibrator/` document how the CSV and JSON
inputs were extracted from BRENDA release 2023.1 and from eQuilibrator. The BRENDA scripts
need the BRENDA flat file `brenda_2023_1.txt`, which is not redistributed here (download it
from https://www.brenda-enzymes.org).

### Running a script

Run every script from the repository root, for example

```bash
julia --project=. --threads=8 scripts/run_revision_paramset.jl
```

Each script reads its inputs and writes its results under `output/` (or `figures/` for the
figure scripts), creating the folders it needs. The header comment of a script describes what
it computes, what it reads and writes, and the environment variables it accepts. Scripts that
fit models use fixed random seeds, and their threaded loops seed each fold or replicate
separately, so the order in which threads run does not change the random draws. Run one
multi-threaded script at a time: several of them use all threads, and the timing scripts
measure wall-clock time.

### Supporting tables and figures

`docs/reproducing.md` gives the command, and any settings, behind every supporting table and
figure, and says how the outputs of the scripts make up each table. In short:

| Supporting table | Script in `scripts/` |
| --- | --- |
| S1, S2, S26, S27 | `run_revision_paramset.jl`, then `run_revision_record_range.jl` for S1 and S26 |
| S3, S4 | `run_revision_inference.jl` |
| S5 | `run_revision_hetero_inference.jl` |
| S6 | `run_revision_mutant.jl` |
| S7, S10, S12, S13 | `run_revision_realdata.jl` |
| S8, S9 | `run_revision_misspecification.jl` |
| S11, S14, S15 | `run_revision_sigmacal.jl` |
| S16, S17, S18 | `run_revision_hetero.jl` |
| S19 | `run_revision_transfer.jl` |
| S20 | `run_revision_transfer_diag.jl` |
| S21 | `run_revision_ecoli_holdout.jl` and `run_revision_invivo_check.jl` |
| S22, S24 | `run_revision_downstream.jl` (prior mean held, `PB_EB_KAPPA0=1e-9`) |
| S23 | `run_revision_dynamic.jl` (prior mean held) |
| S25 | `run_revision_genomescale.jl` |
| S28 | `run_revision_sparse_scaling.jl` |
| S29 | `run_revision_rate_holdout.jl` |
| S30 | `run_revision_rate_simulation.jl` |
| S31 | `run_revision_rate_invivo.jl` (after `run_revision_ecoli_holdout.jl`) |

| Figure | Script in `scripts/` |
| --- | --- |
| Fig 1 | `make_fig1_overview.jl` (with `PB_FIG1_CONCISE=1`) |
| Figs 2, 3 and 5, S1 Text Figs A and B | `make_revision_figures.jl` (Fig 5 from the run with the prior mean held) |
| Fig 4 | `make_fig8_recalibration.jl` |
| S1 Text Figs C and D | `make_concise_figures.jl` |
| S1 Text Fig E | `run_computational_scaling.jl` |

### Settings

Each script's header lists the environment variables it accepts. The common ones are:

| Variable | Effect |
| --- | --- |
| `PB_REV_PATHWAYS` | The pathways to run, comma-separated (`glycolysis`, `tca`, `pentose_phosphate`, `shikimate`). The default depends on the script. |
| `PB_REV_SUFFIX` | A label appended to the output file names, so that a run with other settings keeps the published outputs. |
| `PB_REV_SEED` | The master seed (default 20260805). |
| `PB_REV_EXCLUDE_MUTANTS` | When set, leaves out the records flagged as site-directed mutants. |
| `PB_INFERENCE_METHOD` | `collapsed` (the default, the collapsed sampler) or `structured_vi` (the structured variational approximation), in the scripts that use `src/experiment_runners.jl`. |
| `PB_PARALLEL`, `PB_PARALLEL_CV`, `PB_PARALLEL_SWEEPS` | Set to `false` to run all threaded loops, the cross-validation folds or the simulation replicates one at a time. |

## Outputs

Scripts overwrite their own output files. Results go to `output/<pathway>/results/` (most
analysis scripts write to `output/<pathway>/results/revision/`), summaries across pathways to
`output/`, and figures to `figures/`. The `.gitignore` of the repository keeps these
regenerated files out of version control, apart from the released parameter sets and the
scaling tables that are part of the release.

## Data sources and licences

- `data/brenda/`: turnover numbers and Michaelis constants derived from BRENDA release 2023.1
  (Chang et al., Nucleic Acids Research 49:D498-D508, 2021), redistributed under CC BY 4.0.
- `data/equilibrator/`: standard formation energies from eQuilibrator (Beber et al., Nucleic
  Acids Research 50, 2022, doi:10.1093/nar/gkab1106) at pH 7.4, ionic strength 0.25 mol/L,
  pMg 3.0 and 298.15 K.
- `data/yeast-GEM.xml`: release 9.0.2 of yeast-GEM, the Yeast9 reconstruction (Zhang et al.,
  Molecular Systems Biology 20, 2024), redistributed unchanged under CC BY 4.0.
- `data/pb_prior.tsv`, `data/pb_options.tsv`: prior and option tables for standard parameter
  balancing, in the table format of the software of Lubitz and Liebermeister (Bioinformatics
  35, 2019).
- `data/external/davidi2016_kmax_matched.csv`: in vivo turnover numbers of *Escherichia coli*
  derived from Davidi et al. (PNAS 113:3401-3406, 2016), matched to the four pathways.
  `data/external/README.md` describes how, and why the original workbook is not included.
- `data/external/ecoli_invivo_kapp.csv` and `data/external/ecoli_invivo_concentrations.csv`:
  in vivo apparent rates of *E. coli* enzymes per growth condition and metabolite
  concentrations, derived from the repositories `dandanvidi/kvivo_max` (data of Davidi et al.
  2016 and Schmidt et al., Nature Biotechnology 34:104-110, 2016) and
  `eladnoor/small-molecule-regulation` (concentrations of Gerosa et al., Cell Systems
  1:270-282, 2015), both under the MIT licence, with concentrations from Bennett et al. (Nature
  Chemical Biology 5:593-599, 2009) for two reactions.

`LICENSE` gives the attributions that the CC BY 4.0 licences of BRENDA and yeast-GEM require.

## How to cite

Please cite the article above and the archived version of this software. `CITATION.cff` holds
the citation metadata, and GitHub shows it under "Cite this repository".

## Licence

The code is released under the MIT licence. The files derived from BRENDA and the yeast-GEM
reconstruction are redistributed under CC BY 4.0, as `LICENSE` sets out.
