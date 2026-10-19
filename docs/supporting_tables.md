# Guide to the supporting tables

S1 to S31 Tables of the article "Hierarchical Bayesian modelling for robust parameter
balancing in kinetic models of metabolism" are comma-separated files, kept in the folder
`supporting_tables/`. S1 Text gives the title of
each table in its Section K. This guide describes every table and its columns, in table order.

S1 Text describes rows and columns in words, and this guide gives the codes and
column names under which the files hold them. Section numbers below are those of S1 Text. Quantities
are written in plain text: `s^-1` is per second, `tau^2` the between-organism variance,
`lambda` the prior scale, `kcat` a turnover number and `KM` a Michaelis constant.

The half-decade convention, which several tables name by the code `convention`, is the
standard deviation assumed for every kinetic record: 0.5 on the log10 scale, that is
0.5 ln 10 or about 1.15 in natural-log units, so that one standard deviation is a factor
of about 3.2 in the value.

## S1 Table. The released parameter set for the curated quantities

File `S1_Table.csv`.

For every curated quantity of the four pathways, the table gives the quantity type, enzyme, substrate, and the number of records and source organisms behind it. It gives the posterior median with 95% interval in natural units (s^-1 for turnover numbers, mM for Michaelis constants), both for the consensus parameter and for a previously unmeasured organism. It also gives the estimated record error scale and the between-organism variance used. The column `tau2_source` records whether the per-type estimate or the pooled value was used for that quantity type. The columns `n_mutant_records` and `n_wildtype_labelled` record the enzyme form of the records behind each row. The columns `record_min`, `record_max` and `consensus_outside_records` give the range of each quantity's own records and flag the 25 rows whose consensus median lies outside it. Of these, 24 rest on at most three records, where the joint posterior rather than the few direct records sets the value. For example, succinyl-CoA ligase has a consensus turnover number of 3.1 s^-1 against a single record of 201. These are mixed-form consensus values, not wild-type parameters. Of the records behind the released rows, 31% are site-directed mutants. The median turnover-number row rests on 34% mutants, and five rows with ten or more records rest on 69 to 82%. Mutant records are lower in kcat by factors of 1.9 to 38 (S6 Table). The enzyme form also enters the widths. The per-type between-organism variance behind the unmeasured-organism turnover-number intervals of glycolysis and the TCA cycle is 3.47 and 6.39 in the column `tau2_type`. It falls to 0.92 and 4.97 when the set is rebuilt from wild-type records alone, while the Michaelis-constant variances move by under 8%. It is therefore the turnover-number widths for an unmeasured organism on those two pathways that depend on pooling enzyme forms (S1 Text, Section C.5). A user who wants wild-type values should re-fit on the records the flag keeps. The script `scripts/run_revision_paramset.jl` does this when run with `PB_REV_EXCLUDE_MUTANTS=1`. That wild-type set is released as S26 Table and S27 Table, and it is the one to use for kinetic models of wild-type enzymes. The table covers the quantities that carry records, not the whole basis. Quantities with no record and the thermodynamic block are part of the posterior but are not tabulated here. The intervals are posterior intervals for the parameter. They are built from ingredients that the held-out evaluation validates: the record error scale and the between-organism variance. They are not themselves the record-level predictive intervals whose coverage this paper measures, and they are not certified parameter intervals. Columns prefixed `log_` hold natural logarithms of the value in s^-1 or M, not mM. The columns `tau2_type`, `tau2_pooled` and `tau2_per_type_raw` are between-organism variances of the natural-log value. The column `c_record_scale` multiplies the record variance (0.5 ln 10)^2. See S1 Text, Section D.4.

## S2 Table. The released parameter set, organism-conditional

File `S2_Table.csv`.

There is a row for every curated quantity of the four pathways and every organism with at least 20 records in the pathway. *S. cerevisiae*, *E. coli* and *H. sapiens* are always included where present. Each row gives the posterior median and 95% interval in natural units, conditional on that organism's fitted offset for the quantity type. It also gives the number of that organism's records for the quantity and for the type, the fitted natural-log offset with its posterior standard deviation (columns `log_offset_median` and `log_offset_sd`), and the consensus median for comparison. The table has 2192 rows for 42 organisms. The organism-conditional rows have not been checked by holding out an organism's own records. They are therefore the best organism-specific statement these data allow and have not been validated. The conditional interval shrinks the consensus by the organism's offset for the whole quantity *type*, not by the organism's records for that quantity. This is because a full-rank organism-by-quantity effect is not identified in these data. The interval can therefore sit away from the organism's own values, which the column `organism_own_median` gives where they exist. For yeast hexokinase, for instance, the two curated yeast records have a median of 1.3 s^-1 against a conditional interval of 7.2 to 58.7 s^-1. That interval is drawn from the organism's 53 turnover-number records across the pathway. Of the 123 yeast rows, 76 have no direct yeast record at all. Where a direct record does exist, the yeast median lies outside the conditional interval in 14 of the 47 rows. That is fewer than the 25 expected under the fitted record error, because the median of one or two records carries several times the variance of the parameter interval. The organism's own values therefore neither contradict nor confirm the conditional intervals. Coverage of the yeast rows is also thin. The TCA-cycle and shikimate yeast rows are all Michaelis constants, sharing offsets estimated from 12 and 4 yeast records. Neither pathway has a yeast turnover-number row. The pentose phosphate yeast turnover numbers rest on 9 records, so for those quantities a Yeast9 model should use the consensus or unmeasured-organism rows. The column `n_mutant_records_organism_quantity` carries the enzyme form as in S1 Table.

## S3 Table. Cluster-bootstrap intervals for the leave-one-source-out summaries

File `S3_Table.csv`.

Point estimate, 95% percentile interval and bootstrap standard error for coverage, CRPS and predictive root mean squared error, obtained by resampling whole source organisms.

## S4 Table. Paired organism-level model comparisons

File `S4_Table.csv`.

Hodges-Lehmann estimates and Wilcoxon signed-rank p-values for every pair of model specifications, computed across held-out source organisms.

## S5 Table. Organism-level uncertainty for the error-scale constructions

File `S5_Table.csv`.

Organism-cluster bootstrap intervals for the coverage of the main constructions, and paired contrasts between constructions, on all records and on turnover-number records alone, for all four pathways. Contrasts are evaluated on identical records under common random numbers. Each gives the record-weighted difference with cluster-bootstrap interval and standard error, the organism-weighted difference with its interval, the Hodges-Lehmann median paired difference, and the organism-level Wilcoxon signed-rank p-value. The `level` column distinguishes the nominal 95% and 99% analyses. The record-scale population contrasts against the single replication scale appear at both levels. The `p_holm` column adjusts each contrast across the four pathways by Holm's step-down procedure. The `n_nonzero` column is the number of organisms whose coverage differs between the two constructions. These are the organisms the signed-rank statistic acts on. 10,000 bootstrap replicates.

## S6 Table. Enzyme form of the curated records, wild type and mutant

File `S6_Table.csv`.

Per pathway and quantity type, the number of records and the share flagged as site-directed mutants and as explicitly wild type. The fit-free replication record scale is estimated from all replicate groups, from groups containing no mutant record and from groups whose records are all mutants, with the number of groups behind each. The table also gives the median log difference between mutant and non-mutant records of the same derived quantity, with the number of quantities that carry at least three of each.

## S7 Table. Residual diagnostics for the curated data

File `S7_Table.csv`.

Count, mean, standard deviation, skewness, excess kurtosis, tail fraction and the moment-implied shape parameters of the standardised residuals, overall and stratified by quantity type and by source organism.

## S8 Table. Full misspecification grid

File `S8_Table.csv`.

Empirical coverage at the nominal 95% level, mean interval width, bias, root mean squared error, estimated shape parameters and the number of usable replicates. Rows cover every combination of pathway, observation design, dispersion convention, data-generating family, sweep value and fitted model (608 rows). The `n_reps` column is the number of usable replicates, those of the 100 whose posterior summaries are all finite.

## S9 Table. Misspecification coverage at all nominal levels

File `S9_Table.csv`.

The same grid expanded over 21 nominal levels from 0.01 to 0.99, giving empirical coverage and mean interval width at each level (12768 rows). It is the source for the calibration curves across all levels in S1 Text, Section D.

## S10 Table. Leave-one-source-out results for all model specifications

File `S10_Table.csv`.

CRPS, WIS, log predictive density, nominal 95% coverage, mean interval width, predictive root mean squared error and bias, fitted between-source variance and fitted shape parameters. Rows cover every model specification and both pathways. The `cell` column names the specification: `gaussian_Fix` is the Gaussian model without a source effect (standard parameter balancing applied to individual records), `gaussian_RE_org` and `gaussian_RE_orgtype` are the Gaussian model with an organism effect and with an organism-by-type effect, `t_fixed50_RE_org` is the Student-t model with 50 degrees of freedom and an organism effect, `t_moment_Fix`, `t_moment_RE_org` and `t_moment_RE_orgtype` are the Student-t model with estimated degrees of freedom, and `skew_moment_RE_orgtype` is the skew-normal model with an estimated shape, whose results the text withdraws. The columns `model` (`gaussian`, `t`, `skew`), `re_grouping` (`none`, `org` for an organism effect, `org_type` for an organism-by-type effect) and `shape_mode` (`fixed` for a shape that is not estimated, `moment` for an estimated shape) hold the same information, and together they identify one row per pathway. For the Student-t rows the column `nu` holds the fixed value 50 or the estimate. S12 and S13 Tables use the same codes.

## S11 Table. Recalibrated record-level error scale

File `S11_Table.csv`.

Leave-one-source-out performance of the hierarchical Gaussian model under the half-decade convention and with a record-level error scale estimated from the training fold, both globally and per quantity type. The table also gives the fitted scales and between-cell variances.

## S12 Table. Leave-one-source-out results on the pentose phosphate pathway

File `S12_Table.csv`.

All model cells of the real-data analysis repeated on the third pathway. The residual diagnostics appear as additional rows of S7 Table.

## S13 Table. Leave-one-source-out results on the shikimate pathway

File `S13_Table.csv`.

All model specifications of the real-data analysis on the fourth pathway, outside central carbon metabolism. The residual diagnostics appear as additional rows of S7 Table.

## S14 Table. Recalibrated error scale on the pentose phosphate pathway

File `S14_Table.csv`.

The error-scale analysis of S11 Table repeated on the third pathway.

## S15 Table. Recalibrated error scale on the shikimate pathway

File `S15_Table.csv`.

The error-scale analysis of S11 Table on the fourth pathway.

## S16 Table. Error-scale constructions and conditional calibration

File `S16_Table.csv`.

Record- and organism-averaged 95% coverage, 99% coverage, mean width, 95% interval score, point accuracy, per-quantity-type coverage, and the worst and mean absolute coverage deviation across enzyme-by-type strata. All four pathways are evaluated under common random numbers, with shared prediction draws. The constructions begin with the half-decade convention, the iterated within-cell scale, the replication scale, and its per-type, per-enzyme and per-quantity refinements. Next come the four constructions that stratify the between-organism variance, by the fit-free moment estimator and by the Paule-Mandel marginal estimator, alone and jointly with the per-type record scale. Then come the record-scale populations: the inverse-gamma population by log-scale moments at shape floors 1.3, 1.6 and 2.0, a floor-free log-normal population, the inverse-gamma population by maximum likelihood, and the two-component population by expectation-maximisation. Their fitted shapes, mixing weight, class scales and log-likelihood gains over a single scale are in the `a_mix`, `a_ml`, `pi_two`, `c1_two`, `c2_two` and `dloglik_vs_single` columns. A final construction is the single scale with the between-organism variance integrated over its posterior. The convention, the iterated within-cell scale and the replication scale were fixed before the third and fourth pathways were assembled. Every other row is post hoc in the sense of Methods. The construction `joint2m` runs the per-type between-organism variance and the mean-matched two-class population together. Both `tau_rel` and `joint2m_rel` apply the released fallback rule in every fold. `IS95_org` is the 95% interval score averaged per organism. The `method` column takes 21 values. In the order of the description above they are `convention`, `global_cell`, `global`, `type`, `enzyme`, `quantity`, then `tau_type` and `both` (the fit-free moment estimator of the per-type between-organism variance, alone and with the per-type record scale), `tau_eb` and `joint` (the Paule-Mandel estimator, alone and with the per-type record scale), `scale_mix13`, `scale_mix` and `scale_mix20` (shape floors 1.3, 1.6 and 2.0), `scale_lnmix`, `scale_mixml`, `scale_mix2` (the two-component population as fitted), `scale_mix2m` (the same population rescaled so that its mean equals the pooled scale) and `tau_post`. The other three, `joint2m`, `tau_rel` and `joint2m_rel`, are described above. `CondGapMax` and `CondGapMean` are the worst and the mean absolute coverage deviation across strata, and `IS95` is the 95% interval score per record.

## S17 Table. Per-stratum conditional calibration

File `S17_Table.csv`.

Coverage, mean width, estimated record scale and between-organism variance within each quantity type and each enzyme-by-type stratum with at least 20 records. Rows cover every construction of S16 Table on all four pathways.

## S18 Table. Coverage at all nominal levels for the error-scale constructions

File `S18_Table.csv`.

Record-averaged empirical coverage at 21 nominal levels for every construction of S16 Table.

## S19 Table. Offset-transfer constructions on the recalibrated hierarchy

File `S19_Table.csv`.

Record- and organism-averaged 95% coverage, record- and parameter-level interval widths, cell-mean prediction error, offset-prediction R^2, signal fractions, and the estimated genus variance share and cross-pathway coupling. Rows cover the baseline and the genus, cross-pathway and combined transfer predictors on all four pathways.

## S20 Table. Cross-pathway offset correlations

File `S20_Table.csv`.

Raw Pearson correlations between the same organism-by-type cell's fitted offsets in two full recalibrated pathway fits, with permutation p-values. Correlations are given overall and restricted to cells with at least three records on both sides, for all six pathway pairs and pooled.

## S21 Table. In vivo turnover estimates against the hold-out predictions, per reaction

File `S21_Table.csv`.

One row per matched reaction and predictive object, 33 reactions by four objects (S1 Text, Section G.3). The reaction columns give the pathway, our reaction identifier, name, design row, EC code and direction, the matched reaction of the `iJO1366` reconstruction, the quality of the match, the in vivo maximal apparent turnover number in s^-1, whether it was reported directly or converted from a specific activity, the molecular weight used in the conversion, and whether the reconstruction lists isozymes or a protein complex for that reaction. Further columns give the two subsets used in the check, the number of training records and source organisms behind the row once *E. coli* is removed, and the number of held-out *E. coli* records and how many of them are flagged as site-directed mutants. For each predictive object the table gives the 95% interval in s^-1 and its median, the interval width in decades, whether the in vivo value lies inside, above or below it, the log ratio of the value to the predictive median, the Gaussian log predictive density, the CRPS, the probability integral transform, and whether the value lies inside or above the interval under the 1.9-fold kcat proxy. The last column carries the matching notes, including the isozyme, complex, cofactor and lumping caveats. The four objects are the consensus and record-level intervals of the Gaussian fixed-effects fit and the unmeasured-organism and record-level intervals of the pre-specified default.

## S22 Table. Downstream propagation results by reaction

File `S22_Table.csv`.

Posterior median and 95% interval width for the log10 reaction rate and log10 Keq, and the posterior probability that the reaction proceeds forwards. Rows cover every reaction, fitted model and downstream target (consensus or new organism), the last model being the pre-specified default with the record scale estimated (231 rows). Every fit holds the prior mean at the thermodynamic prior (`PB_EB_KAPPA0=1e-9`).

## S23 Table. Dynamic propagation and rate-law sensitivity

File `S23_Table.csv`.

Posterior median and 95% width of the terminal-reaction flux, per model specification and downstream target, with convergence counts. The flux is computed under the static convenience evaluation, the static mass-action evaluation, and full ODE integration to steady state. Rows cover the Gaussian fixed-effects consensus and the consensus and new-organism targets of the Gaussian organism-by-type fit, under the half-decade convention and under the pre-specified default (10 rows). Every fit holds the prior mean at the thermodynamic prior (`PB_EB_KAPPA0=1e-9`).

## S24 Table. Downstream substrate elasticities

File `S24_Table.csv`.

Posterior median and 95% interval width of the scaled elasticity d ln v / d ln c for every reaction, substrate, fitted model and downstream target, the last model being the pre-specified default with the record scale estimated (396 rows). Every fit holds the prior mean at the thermodynamic prior (`PB_EB_KAPPA0=1e-9`).

## S25 Table. Genome-scale posterior solve on a real reconstruction

File `S25_Table.csv`.

Basis dimension, non-zeros, fill-in and wall-clock for building the dependency structure, assembling the precision, factorising, solving for the posterior mean and drawing an exact posterior sample. These are given on yeast-GEM and on subsampled reconstructions where the dense path remains feasible, with the dense memory and flop requirement at the same dimension. The table also gives the cost and fill of one hierarchical source refresh, which rebuilds the rank-300 formation-energy factor as well as the sparse factorisation. It also gives the relative error of the sparse posterior mean against the dense one. This error is undefined for the full reconstruction because the dense path is not run there. Timings are single-threaded medians of three repetitions after a warm-up. The reconstruction supplies the structure and dimension. The records placed on it are synthetic and cover every coordinate, which is the worst case for fill-in.

## S26 Table. The released parameter set rebuilt from wild-type records

File `S26_Table.csv`.

As S1 Table, built by the same specification from the records not flagged as site-directed mutants. It is intended for kinetic models of wild-type enzymes. Its consensus lies outside the range of its own records in 30 of 148 rows (column `consensus_outside_records`).

## S27 Table. The organism-conditional parameter set rebuilt from wild-type records

File `S27_Table.csv`.

As S2 Table, from the records not flagged as site-directed mutants.

## S28 Table. Sparse-precision scaling benchmark

File `S28_Table.csv`.

Assembly and factor-solve-sample timings and Cholesky fill for the sparse and dense posterior solves on structured problems of growing dimension, up to D = 8000. The benchmark also covers the hierarchical (collapsed-Schur) variant, its Woodbury treatment of pathway-wide sources, and the factor-model solve of the formation-energy block.

## S29 Table. Coverage of predicted rates for held-out organisms

File `S29_Table.csv`.

One row per pathway (and the four pathways pooled), predictive object, level and stratum (S1 Text, Section H.4). A unit is an organism and a reaction for which the organism has forward turnover-number records and Michaelis-constant records for every substrate of the forward direction, water excluded. Its observed rate is the forward rate per unit enzyme with every substrate at 1 mM and no product present, built from the means of its log records. Each organism is held out in its published leave-one-source-out fold, and both fits reproduce the per-record coverage of that fold. The column `variant` names the predictive object: `spb` standard parameter balancing (Gaussian fixed effects at the half-decade convention), `default` the pre-specified default with independent organism offsets on turnover numbers and on Michaelis constants, `velocity` the default with the turnover-number offset only, as in the main-text propagation, and `per_type` the default with the per-type between-organism variance of the released specification. The column `level` is `record` for the distribution of what the held-out records show, which is the calibrated comparison, or `organism` for the interval without record error, the one a modeller would use for that organism. The column `stratum` is `all`, `km_sensitive` (units whose rate depends on the Michaelis constants at 1 mM, with a derivative of the log rate with respect to a common shift of the log Michaelis constants of at least 0.5), `saturated` (below 0.5) or `no_mutant` (units without any record flagged as a site-directed mutant). Columns give the numbers of units and organisms, the coverage of central 50, 80, 90 and 95% intervals, an organism-clustered bootstrap 95% interval for the 95% coverage, the mean probability integral transform, the median width of the 95% interval in decades, the mean CRPS and the median error in log10 units, and the 95% coverage of the turnover-number mean and of the Michaelis-constant means of the same units.

## S30 Table. Coverage of predicted rates in simulation with known parameters

File `S30_Table.csv`.

One row per run, pathway, scenario, model and target (S1 Text, Section H.4). Records are redrawn on the real glycolysis and TCA-cycle structure about the posterior mean of the released set, with organism-by-type offsets and record errors at the values the pre-specified default estimated on the real records. Scenario `B` gives every record one scale. Scenario `D` draws one scale per replicate group (one organism, one quantity) from an inverse-gamma population with shape 3 and scale 8. The column `run` is `main`, `em_budget_x4` (four times the EM iterations of every fit) or `truth_drawn` (the true basis vector drawn from the released posterior in every replicate). The column `model` is `spb` (standard parameter balancing, prior mean updated), `spb_held` (the same with the prior mean held at the thermodynamic prior), `default` (the pre-specified default, prior mean updated, as in the cross-validation), `default_held` (the same with the prior mean held, as in the propagation runs of S22 to S24 Tables) or `released` (the configuration of the released parameter set: prior mean held, fit-free record scale and per-type between-organism variance). The column `target` is `consensus` (the posterior draws as they are), `new_vel` (plus a turnover-number offset of a new organism, as in the main-text propagation) or `new_full` (plus a Michaelis-constant offset as well). Columns give the numbers of replicates and reactions, the 95% coverage of the true consensus rate and of the true rates of 400 simulated new organisms per replicate, each with its standard error over replicates, and the median width of the 95% interval of the log10 rate in decades. The last six columns describe the fit of the model in that run, pathway and scenario, the same for every target: the medians over replicates of the record scale `c`, of the pooled between-organism variance, of the between-organism variances used for turnover numbers and for Michaelis constants (per type for the released configuration, the pooled value otherwise, zero for standard parameter balancing), and of the prior scale `lambda`, and the mean over replicates of the 95% coverage of the true values of the curated quantities by the consensus intervals.

## S31 Table. Predicted rates at measured E. coli concentrations against in vivo apparent rates

File `S31_Table.csv`.

One row per concentration-error setting, reaction, growth condition and predictive object (S1 Text, Section H.4). The in vivo apparent rate per enzyme chain, in s^-1 and signed in the forward direction of our model, is the flux divided by the proteomic abundance in batch growth on glucose, fructose or glycerol (`data/external/ecoli_invivo_kapp.csv`). The rate law is evaluated at the metabolite concentrations measured in the same condition (`data/external/ecoli_invivo_concentrations.csv`), with the posterior draws of the *E. coli* hold-out fits. The column `concentration_error_sd` is 0 for the measured concentrations as they are, and 0.2 or 0.5 for a log-normal error of that standard deviation on every concentration in every draw. The column `variant` is `spb` (standard parameter balancing), `velocity` (the pre-specified default with an *E. coli* turnover-number offset, as in the main-text propagation), `full` (with a Michaelis-constant offset as well), or `kcat_spb` and `kcat_full` (the turnover number alone, in the direction of the observed flux). Columns give the source of the concentrations, the in vivo rate, the 95% interval and median of the predicted signed rate, whether the interval contains the in vivo rate, its probability integral transform, the fraction of draws in the observed direction, the log10 error of the median where the median has the observed direction, the width of the 95% interval of log10 |rate| in decades, the median natural log of the mass-action ratio over the equilibrium constant, the number of species in the rate law, and the thermodynamic class (`near_equilibrium` when that log ratio lies within 1 of zero under the default).
