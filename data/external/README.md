# External data

## davidi2016_kmax_matched.csv

In vivo maximal apparent turnover numbers (kmax, 1/s) of *Escherichia coli* enzymes, matched to the forward turnover-number rows (kcat+) of the four pathway models in this repository (glycolysis, TCA cycle, pentose phosphate, shikimate). `scripts/run_revision_invivo_check.jl` uses the table as an independent check on the E. coli hold-out predictions of `scripts/run_revision_ecoli_holdout.jl`.

### Source

Davidi D, Noor E, Liebermeister W, Bar-Even A, Flamholz A, Tummler K, Barenholz U, Goldenfeld M, Shlomi T, Milo R. Global characterization of in vivo enzyme catalytic rates and their correspondence to in vitro kcat measurements. *Proc Natl Acad Sci USA* 113(12):3401-3406 (2016). doi:[10.1073/pnas.1514240113](https://doi.org/10.1073/pnas.1514240113). Supporting Information, Dataset S1 (`pnas.1514240113.sd01.xlsx`).

The original workbook is **not redistributed** here. The article was published under the PNAS open-access option (2016), which has no Creative Commons licence, so this file holds only derived per-reaction numbers with the citation above. To rebuild the table, download the workbook yourself, from the article page or from PubMed Central (PMC4812741):
`https://pmc.ncbi.nlm.nih.gov/articles/instance/4812741/bin/pnas.1514240113.sd01.xlsx`.
The copy used here had SHA-256 `34cc8b2891703938e8fbb2576b4328ad9249edbb360bbe8ce9321278b0124068`.

Two further inputs are needed; neither is redistributed either:
- The BiGG iJO1366 model in JSON (`http://bigg.ucsd.edu/static/models/iJO1366.json`), for gene rules, EC annotations and reaction directions.
- `model_genes.csv` from `https://github.com/dandanvidi/kvivo_max` (MIT licence), for protein molecular weights.

### How the table was built

```
python3 scripts/build_davidi_kmax_match.py /path/to/pnas.1514240113.sd01.xlsx \
    --ijo1366 /path/to/iJO1366.json --model-genes /path/to/model_genes.csv
```

The script needs `openpyxl`, because it reads the SI font colour that marks isozyme reactions. It also reads `supporting_tables/S1_Table.csv` (read-only) to flag which rows carry BRENDA kcat records. With `--full-out PATH` it also writes a working table with every intermediate column (Davidi growth conditions, specific activities, S1 Table intervals). That table quotes more of the source dataset, so keep it outside the repository.

1. **Reaction mapping.** Each yeast-GEM reaction of a pathway was mapped by hand to its iJO1366 counterpart (see `SPEC` in the script). Directions were taken from iJO1366: iJO1366 writes PGK, PGM, RPI and SUCOAS the opposite way to our models, so Davidi's `_reverse` entries are our forward direction. EC matching uses the iJO1366 annotations, because the EC column of the SI "kcat [1/s]" sheet is wrong for several rows (TPI, MDH, PGL, RPI, DHQS, CHORS).
2. **kmax.** Where the SI "kmax [1/s]" sheet gives the value, it is used per polypeptide chain (`kmax_source = direct_1s_per_chain`). Otherwise kmax is converted from the "kmax [umol/mg/min]" sheet (`kmax_source = converted_from_specific_activity`) as

   kmax [1/s] = SA [umol/mg/min] x MW [kDa] / 60.

   `mw_used_kDa` gives the molecular weight and `mw_rule` gives how it was chosen: the chain mass for a single enzyme, the mean chain mass for an isozyme reaction, the summed mass of one copy of each subunit for a complex, or a named isozyme. The specific activity can be recovered as SA = kmax_1s x 60 / mw_used_kDa. Applied to single enzymes, this formula reproduces the directly given 1/s values to within about 5%, except for PFK.
3. **Values given to 4 significant digits.** Directly given values are as printed in the SI.

### Columns

| column | meaning |
|---|---|
| pathway, reaction_id, reaction_name | our pathway and yeast-GEM reaction |
| direction, Q_row | always `forward`; row of the full observation matrix Q (kcat+ of the reaction) |
| ec_code, reference_substrate | our EC number and the kcat+ reference substrate |
| has_kcat_records | `true` if the row carries BRENDA kcat records in the curated data (an S1 Table row) |
| iJO1366_reaction, iJO1366_genes | matched iJO1366 reaction and its gene rule (gene names) |
| davidi_reaction_id | Davidi's reaction id for our forward direction (empty if unmatched) |
| match_quality | `exact`, `cofactor_differs` (ICDH: NADP+ in E. coli, NAD+ in yeast), `lumped_complex` (AKGDH covers the whole OGDH complex), or `none` |
| match_by_reaction, match_by_ec | details of the match |
| kmax_1s | in vivo kmax in 1/s (per chain for direct values) |
| kmax_source | `direct_1s_per_chain` or `converted_from_specific_activity` |
| mw_used_kDa, mw_rule | molecular weight and rule for converted values |
| kmax_per_active_site_1s | Davidi's per-active-site kmax where available (2 to 4 times the per-chain value for PFK, TPI, GAPD, ENO, CS and GND, because the SI records few active sites per complex) |
| isozymes_in_iJO1366, protein_complex | the iJO1366 gene rule has `or` (isozymes) or `and` (complex) |
| davidi_si_enzyme_class | class marked by the SI font colour in the specific-activity sheet |
| notes | match caveats |

### Counts

The table has 36 forward rows: 33 with a kmax and 3 without (hexokinase, pyruvate kinase and transaldolase have no Davidi kmax).

- 29 of the 33 matched rows carry BRENDA kcat records: glycolysis 8, TCA 8, pentose phosphate 6, shikimate 7.
- 15 of those 29 have kmax given directly in 1/s: glycolysis 6, TCA 3, pentose phosphate 2, shikimate 4.
- The other 4 matched rows have no records, so the model predicts them from the full Q only: TCA r_0280, r_0831 and r_0505, and pentose phosphate r_0091 (PGL, a direct value).

### Caveats

- **kmax is not kcat.** kmax is the largest apparent rate over 31 growth conditions (pFBA flux divided by proteomic abundance). It is a lower bound on kcat and typically about 2-fold below it. The median kcat/kmax ratio is 1.82 over the 132 enzyme-reaction pairs of the Dataset S1 "kcat vs. kmax" sheet, and 1.93 over the 14 matched rows here that have both an in vitro kcat and a direct per-chain kmax. `scripts/run_revision_invivo_check.jl` therefore uses log(1.9) as its sensitivity shift. A kmax exists only where pFBA puts flux through the reaction in that direction.
- **Isozyme values underestimate the fastest isozyme.** For isozyme reactions, specific activity is averaged over isozymes weighted by mass.
- **Complex values are nominal.** For complexes (AKGDH, SUCOAS, SUCDi), the conversion is per one set of subunits.
- **Different organism.** The values are for E. coli, while our pathway models are yeast networks populated with cross-organism records.

- `kcat_vitro_ecoli_davidi_1s`: the in vitro turnover number for E. coli reported in the same source, where the source gives one (16 of 36 rows). It is not used in the comparison, which uses the in vivo values only; it is kept so that statements comparing the two are reproducible.

## ecoli_invivo_kapp.csv and ecoli_invivo_concentrations.csv

Inputs of `scripts/run_revision_rate_invivo.jl`, which evaluates the convenience rate law of the Propagation section at measured *E. coli* metabolite concentrations, with the parameters of the *E. coli* hold-out fits, and compares the predicted rate per enzyme with the in vivo apparent rate per enzyme (kapp) in the same growth condition.

### Sources

- **kapp.** Davidi et al. (2016), cited above, through the repository [github.com/dandanvidi/kvivo_max](https://github.com/dandanvidi/kvivo_max) (MIT licence, copyright 2015 dandanvidi). kapp is the pFBA flux (`data/flux[mmol_gCDW_s].csv`) divided by the proteomic abundance per polypeptide chain (`data/abundance_by_reaction[mmol_gCDW].csv`), as that repository's `scripts/catalytic_rates.py` divides them. The abundances are those of Schmidt A, et al. The quantitative and condition-dependent *Escherichia coli* proteome. *Nat Biotechnol* 34:104-110 (2016).
- **Concentrations.** Gerosa L, Haverkorn van Rijsewijk BRB, Christodoulou D, Kochanowski K, Schmidt TSB, Noor E, Sauer U. Pseudo-transition analysis identifies the key regulators of dynamic metabolic adaptations from steady-state data. *Cell Syst* 1:270-282 (2015). doi:[10.1016/j.cels.2015.09.008](https://doi.org/10.1016/j.cels.2015.09.008). Taken from `data/ecoli_metabolites_gerosa2015.csv` of [github.com/eladnoor/small-molecule-regulation](https://github.com/eladnoor/small-molecule-regulation) (MIT licence, copyright 2016 Elad Noor), in mM. For citrate synthase and malate dehydrogenase, whose acetyl-CoA, CoA and oxaloacetate Gerosa et al. did not measure, all concentrations come from Bennett BD, Kimball EH, Gao M, Osterhout R, Van Dien SJ, Rabinowitz JD. Absolute metabolite concentrations and implied enzyme active site occupancy in *Escherichia coli*. *Nat Chem Biol* 5:593-599 (2009), through kvivo_max `data/bennett_metabolite_concentrations[mM].csv`, whose oxaloacetate value (1 µM) that file attributes to Zimmermann et al. (2014).

Only the derived numbers used here are written: one kapp per reaction and condition, and the concentrations of the metabolites the rate laws need. The two MIT licences permit redistribution with their copyright notices, which are given above and in `LICENSE`.

### How the tables were built

```
python3 -I scripts/build_invivo_rate_inputs.py --kvivo /path/to/kvivo_max/data \
    --gerosa /path/to/small-molecule-regulation/data/ecoli_metabolites_gerosa2015.csv
```

### What is included

- **Reactions.** The eight reactions of our models whose *E. coli* enzyme is one polypeptide (or a named isozyme in kvivo_max) and whose substrates and products were all measured: glucose-6-phosphate isomerase, 6-phosphofructokinase, phosphoglycerate kinase and enolase (glycolysis), ribose-5-phosphate isomerase and ribulose-5-phosphate 3-epimerase (pentose phosphate), citrate synthase and malate dehydrogenase (TCA cycle). iJO1366 writes PGK and RPI opposite to our models, so their flux is sign-flipped (`orientation = -1`). kapp is signed: positive in the forward direction of our model.
- **Conditions.** Batch growth on glucose (`GLC_BATCH_mu=0.58_S`), fructose (`FRU_BATCH_mu=0.65_S`) and glycerol (`GLYC_BATCH_mu=0.47_S`). The galactose, acetate and xylose columns of the kvivo_max flux table have no carbon uptake, so they are not used. Pyruvate and succinate are left out because their pFBA solutions run upper glycolysis in the glycolytic direction, which we could not check. Bennett et al. give oxaloacetate for glucose only, so citrate synthase and malate dehydrogenase are glucose only.
- **Counts.** 16 of the 24 reaction-condition pairs have a kapp: PGI and PFK carry no pFBA flux on glycerol, PFK none on fructose, RPE has no abundance on fructose, and the two TCA reactions lack oxaloacetate outside glucose (`exclusion_reason`).

### Caveats

- **kapp is a model-based estimate.** Its flux comes from pFBA, not from 13C labelling, and its abundance from proteomics. Neither carries an error here.
- **2- and 3-phosphoglycerate are one pool** in Gerosa et al. The driver splits the pool with each draw's own phosphoglycerate mutase equilibrium constant, so it assumes that reaction is at equilibrium.
- **Oxaloacetate is an estimate.** At 1 µM it limits citrate synthase and makes malate dehydrogenase look thermodynamically unfavourable in the forward direction, which is the known difficulty of this pair rather than a property of the fit.
- **Different organism and compartment.** The rate laws and thermodynamic data are those of the yeast models (the TCA cycle at mitochondrial pH), evaluated with *E. coli* concentrations.
