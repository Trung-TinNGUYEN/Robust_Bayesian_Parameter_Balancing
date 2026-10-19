# Software notes

Notes on the package, the command-line tool and the released parameter set for the article
"Hierarchical Bayesian modelling for robust parameter balancing in kinetic models of metabolism". Until October 2026
they were part of S1 Text (Section I.4, the worked example of Section F, and the script
names and command-line options quoted in Sections B to E). They were moved here, unchanged
in content, to shorten that document. Section numbers below are those of S1 Text.

The repository `README.md` remains the place for installation, the quick start and the
full list of experiment drivers.

## Package and command-line interface

The code is the Julia package `RobustBayesianParameterBalancing`, with the helper
`bin/parameter-balance.jl` for file-based use. It follows the workflow of the existing
parameter balancing software of Lubitz and Liebermeister (2019): prior and option tables,
validation and export of SBtab-like quantity tables, and the closed-form Gaussian update or
the robust and hierarchical Monte Carlo expectation-maximisation. Analyses that need
explicit fold structure run through the Julia API and the experiment drivers, and
`examples/released_specification.sh` runs the released specification end to end.

Main entry points of the package. The SBML export writes a balanced-parameter sidecar
table, and kinetic-law synthesis is left to SBML tooling.

| Interface | Function or command | Purpose |
| --- | --- | --- |
| Julia API | `load_parameter_balancing_prior` | Load prior medians, spreads, bounds and quantity metadata |
| Julia API | `validate_pb_inputs` | Check prior tables, quantity tables and matrix dimensions |
| Julia API | `balance_parameters` | Closed-form Gaussian balancing, or robust fold-wise inference |
| Julia API | `export_balanced_table` | Export posterior summaries as DataFrame, CSV, TSV or SBtab-like text |
| Julia API | `kineticize_sbml` | Balanced-parameter sidecar for an SBML model |
| Command line | `validate-table`, `pseudo-fill` | Validate a quantity table, and add pseudo-observation rows |
| Command line | `balance-gaussian` | Gaussian balancing from numeric matrix files |
| Command line | `balance-hierarchical` | Hierarchy with a source-level intercept and an optional estimated record scale |

The hierarchical command takes the numeric inputs of Gaussian balancing plus a source label
and, for the replication scale, a replicate-group label per record:

```
julia --project=. bin/parameter-balance.jl balance-hierarchical \
  --Q Q.csv --x x.csv --Cx Cx.csv --q-prior q_prior.csv \
  --C-prior C_prior.csv --group sources.csv \
  --replicate-group replicates.csv --type types.csv \
  --error-scale replication --output balanced_parameters.csv
```

It reports the estimated between-organism variance with the balanced posterior.

| Option | Effect |
| --- | --- |
| no `--error-scale` | The record scale is fixed at one, which reproduces standard parameter balancing and its hierarchical extension (the package default). |
| `--error-scale auto` | The iterated within-cell estimator of S1 Text, Section E.1 (the pre-specified default). |
| `--error-scale replication` | The fit-free estimator of S1 Text, Section E.2. It also fits the two-class population of replicate-group scales and reports the mixing weight, the class scales and the log-likelihood gain. |
| `--type FILE` | Estimates the between-source variance per quantity type by the Paule-Mandel estimator, with the pooled value as fallback outside 0.2 to 5 times it. With `--error-scale replication` this is the released specification. |
| `--prior-mean fixed` | The default. Holds the prior mean at the thermodynamic prior (S1 Text, Section D.4). |
| `--outer N` | The number of EM iterations, a fixed budget and not a convergence criterion (S1 Text, Section B.6). |
| `--predict FILE` | Writes, for every row of the dependency matrix Q, the posterior of the derived quantity and the interval for a previously unmeasured source at the levels of `--predict-levels`, under the single scale and the mean-matched two-class population. |

On the glycolysis records the command reproduces exactly the replication scale (5.61), the
pooled between-organism variance (1.15) and the two-class population (pi = 0.61, c1 = 0.78,
c2 = 13.2) of S1 Text, Section E. These c1 and c2 are the mean-matched values: the fitted
0.51 and 8.6 rescaled by 1.53, the ratio of the pooled scale to the population mean. The
per-type variances (3.07 and 0.36, against 3.47 and 0.37 in the released set) agree within
the sampler's Monte Carlo variation.

## Using the released parameter set

### Files

`output/paramset/` holds the released set. Every derived quantity is a fixed linear function
of the basis vector, so the released posterior mean, covariance and draws of the basis
vector (`basis_posterior_mean_<pathway>.csv`, `basis_posterior_cov_<pathway>.csv` and
`basis_posterior_draws_<pathway>.csv`) let a modeller reconstruct any quantity, or any
function of several, with the Haldane relations holding exactly. The file
`basis_design_<pathway>.csv` gives each curated quantity's row of the dependency matrix with
its type, EC number and substrate, so its posterior draws are the dot products of that row
with the basis draws. Rows for quantities without records need the dependency matrix, which
the pipeline rebuilds.

### A worked example

The hexokinase (EC 2.7.1.1) turnover number with D-glucose in glycolysis rests on 76 records
from 9 organisms. The consensus is 22.8 s^-1 (10.7 to 54.0). For an unmeasured organism it
is 23.1 s^-1 (0.57 to 1037), because the turnover-number between-organism variance of
glycolysis, 3.47 on the natural-log scale, opens the interval to three orders of magnitude.
*S. cerevisiae*, with 2 direct records and 53 turnover-number records in the pathway, has a
log offset of -0.19 +/- 0.40 and a conditional posterior of 18.4 s^-1 (7.2 to 58.7).
*H. sapiens*, with 59 direct records, has 20.2 s^-1 (10.7 to 35.2). *E. coli*, with none,
borrows an offset of -1.03 +/- 0.59 from its 24 other glycolysis turnover-number records,
giving 8.1 s^-1 (2.3 to 30.6). For a Yeast9 model, the consensus row would understate the
width by the offset uncertainty, and the unmeasured-organism row would overstate it by
ignoring the 53 records that pin the yeast offset.

Organism-conditional rows are marginal. The offset is added to each derived quantity by
type, outside the basis vector, so a set of such rows does not satisfy the Haldane
relationships jointly, and for a reaction with unequal numbers of substrates and products
the implied equilibrium constant shifts with the Michaelis-constant offset. To keep the
relationships exact, apply the organism's turnover-number offset (columns
`log_offset_median` and `log_offset_sd` of S2 Table) to the reaction-velocity block of the
basis vector, as in S1 Text, Section H, and take the Michaelis constants from the consensus
rows.

The conditional row is also a type-level shrinkage of the consensus, not a fit to the
organism's own values. It uses the yeast offset estimated from 53 turnover-number records,
not the two yeast hexokinase records, whose median of 1.3 s^-1 lies outside the interval
(column `organism_own_median` of S2 Table). And 53 of the 76 records behind this
quantity are site-directed mutants, so the consensus being shrunk is a mixed-form value
(S1 Text, Section C.5).

## Scripts behind passages of S1 Text

Earlier versions of S1 Text named the script or option behind some passages. They are
listed here in place of that text.

| Passage of S1 Text | Script or option |
| --- | --- |
| C.4, the mutant flag | `src/mutant_flags.jl` |
| D.4, synthetic scenarios with organism effects | `scripts/validate_hetero.jl` |
| E, the estimators on each training fold | `scripts/run_revision_sigmacal.jl`, `scripts/run_revision_hetero.jl`, `scripts/validate_hetero.jl` |
| E.5, the synthetic checks of the transfer layer | `scripts/validate_transfer.jl` |
| F, the released set rebuilt from wild-type records | `scripts/run_revision_paramset.jl` with `PB_REV_EXCLUDE_MUTANTS=1` |
| The structured variational approximation | The keyword `method = :structured_vi` of `balance_parameters`, `leave_one_source_out_cv` and `fit_fold_mcem`, and `PB_INFERENCE_METHOD=structured_vi` in the scripts that use `src/experiment_runners.jl`. No result of the shorter article uses it. |
