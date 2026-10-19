"""Build the inputs of run_revision_rate_invivo.jl.

Writes two derived tables for the in vivo rate check of the E. coli hold-out:

  data/external/ecoli_invivo_kapp.csv
      In vivo apparent turnover rates per polypeptide chain (kapp, 1/s) of
      E. coli, per growth condition, for the reactions of our pathway models
      whose E. coli enzyme is a single polypeptide (homomeric, or a named
      isozyme), signed in the forward direction of our models.
  data/external/ecoli_invivo_concentrations.csv
      The intracellular metabolite concentrations (mM) used to evaluate the
      rate law in the same conditions.

Sources
  kapp      Davidi et al. (2016, PNAS 113:3401), through the repository
            github.com/dandanvidi/kvivo_max (MIT licence, copyright 2015
            dandanvidi): kapp = pFBA flux / proteomic abundance per chain, with
            flux[mmol_gCDW_s].csv and abundance_by_reaction[mmol_gCDW].csv
            exactly as that repository's catalytic_rates.py divides them.  The
            abundances are those of Schmidt et al. (2016, Nat Biotechnol 34:104).
  conc      Gerosa et al. (2015, Cell Syst 1:270), through the repository
            github.com/eladnoor/small-molecule-regulation (MIT licence,
            copyright 2016 Elad Noor), file data/ecoli_metabolites_gerosa2015.csv,
            in mM; for citrate synthase and malate dehydrogenase, whose
            acetyl-CoA, CoA and oxaloacetate Gerosa et al. did not measure,
            Bennett et al. (2009, Nat Chem Biol 5:593) through kvivo_max file
            data/bennett_metabolite_concentrations[mM].csv (its oxaloacetate
            value is from Zimmermann et al. 2014, as that file notes).

Conditions are the three batch cultures where the pFBA solution takes up the
named carbon source and the concentrations were measured: glucose
(GLC_BATCH_mu=0.58_S), fructose (FRU_BATCH_mu=0.65_S) and glycerol
(GLYC_BATCH_mu=0.47_S).  The galactose, acetate and xylose columns of the flux
table have no carbon uptake at all, so they are not used; pyruvate and
succinate are left out because their pFBA solutions run upper glycolysis in the
glycolytic direction, which we could not check.  Bennett et al. give
oxaloacetate for glucose only, so the two TCA reactions are glucose only.

Directions: iJO1366 writes PGK and RPI opposite to our models (orientation -1).

Usage
  python3 -I scripts/build_invivo_rate_inputs.py --kvivo /path/to/kvivo_max/data \\
      --gerosa /path/to/small-molecule-regulation/data/ecoli_metabolites_gerosa2015.csv
Only the standard library is needed.
"""
import argparse
import csv
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (pathway, our reaction, name, iJO1366 reaction, orientation, concentration source)
REACTIONS = [
    ("glycolysis", "r_0467", "glucose-6-phosphate isomerase", "PGI", 1, "gerosa2015"),
    ("glycolysis", "r_0886", "6-phosphofructokinase", "PFK", 1, "gerosa2015"),
    ("glycolysis", "r_0892", "phosphoglycerate kinase", "PGK", -1, "gerosa2015"),
    ("glycolysis", "r_0366", "enolase", "ENO", 1, "gerosa2015"),
    ("pentose_phosphate", "r_0982", "ribose-5-phosphate isomerase", "RPI", -1, "gerosa2015"),
    ("pentose_phosphate", "r_0984", "ribulose-5-phosphate 3-epimerase", "RPE", 1, "gerosa2015"),
    ("tca", "r_0300", "citrate synthase", "CS", 1, "bennett2009"),
    ("tca", "r_0713", "malate dehydrogenase", "MDH", 1, "bennett2009"),
]

# our condition name -> (pFBA and proteomics column, Gerosa column)
CONDITIONS = [
    ("glucose", "GLC_BATCH_mu=0.58_S", "Glucose"),
    ("fructose", "FRU_BATCH_mu=0.65_S", "Fructose"),
    ("glycerol", "GLYC_BATCH_mu=0.47_S", "Glycerol"),
]
# Bennett et al. measured glucose, glycerol and acetate; oxaloacetate only on glucose.
BENNETT_CONDITIONS = {"glucose": "glc"}

# BiGG metabolites the rate laws need (water excluded), by concentration source.
GEROSA_METS = ["g6p", "f6p", "fdp", "atp", "adp", "13dpg", "2pg+3pg", "pep",
               "ru5p__D", "r5p", "xu5p__D"]
BENNETT_METS = {"accoa": "C00024", "oaa": "C00036", "cit": "C00158", "coa": "C00010",
                "mal__L": "C00149", "nad": "C00003", "nadh": "C00004"}


def read_table(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], {r[0]: r[1:] for r in rows[1:]}


def fnum(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return math.nan
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kvivo", required=True, help="the data folder of kvivo_max")
    ap.add_argument("--gerosa", required=True, help="ecoli_metabolites_gerosa2015.csv")
    ap.add_argument("--out-kapp", default=os.path.join(REPO, "data", "external", "ecoli_invivo_kapp.csv"))
    ap.add_argument("--out-conc", default=os.path.join(REPO, "data", "external",
                                                       "ecoli_invivo_concentrations.csv"))
    a = ap.parse_args()

    fh, flux = read_table(os.path.join(a.kvivo, "flux[mmol_gCDW_s].csv"))
    ah, abund = read_table(os.path.join(a.kvivo, "abundance_by_reaction[mmol_gCDW].csv"))
    bh, bennett = read_table(os.path.join(a.kvivo, "bennett_metabolite_concentrations[mM].csv"))
    with open(a.gerosa, newline="") as f:
        gr = list(csv.reader(f))
    gh = gr[0]
    gerosa = {r[0]: r for r in gr[1:]}

    kapp_rows = []
    for (pw, rid, name, ijo, orient, src) in REACTIONS:
        for (cond, ccol, _) in CONDITIONS:
            reason = ""
            if src == "bennett2009" and cond not in BENNETT_CONDITIONS:
                reason = "no oxaloacetate concentration for this condition"
            fi = fh.index(ccol) - 1
            ai = ah.index(ccol) - 1
            f_fwd = fnum(flux[ijo][fi]) if ijo in flux else math.nan
            f_rev = fnum(flux[ijo + "_reverse"][fi]) if ijo + "_reverse" in flux else 0.0
            net_ijo = (0.0 if math.isnan(f_fwd) else f_fwd) - (0.0 if math.isnan(f_rev) else f_rev)
            net = orient * net_ijo
            e = fnum(abund[ijo][ai]) if ijo in abund else math.nan
            if not reason and (math.isnan(e) or e <= 0):
                reason = "no proteomic abundance"
            if not reason and net == 0.0:
                reason = "no pFBA flux"
            kapp = net / e if (e == e and e > 0) else math.nan
            kapp_rows.append({
                "pathway": pw, "reaction_id": rid, "reaction_name": name,
                "iJO1366_reaction": ijo, "orientation": orient, "condition": cond,
                "flux_condition": ccol, "net_flux_mmol_gCDW_s": "%.6g" % net,
                "abundance_mmol_gCDW": "" if math.isnan(e) else "%.6g" % e,
                "kapp_signed_1s": "" if (math.isnan(kapp) or reason) else "%.6g" % kapp,
                "concentration_source": src, "included": "true" if not reason else "false",
                "exclusion_reason": reason})

    conc_rows = []
    for (cond, _, gcol) in CONDITIONS:
        mi = gh.index(gcol + " (mean)")
        si = gh.index(gcol + " (std)")
        for m in GEROSA_METS:
            key = "2pg_c+3pg_c" if m == "2pg+3pg" else m + "_c"
            r = gerosa[key]
            conc_rows.append({"condition": cond, "bigg_id": m, "conc_mM": r[mi],
                              "sd_mM": r[si], "source": "gerosa2015"})
    for cond, bcol in BENNETT_CONDITIONS.items():
        bi = bh.index(bcol) - 1
        for m, kegg in BENNETT_METS.items():
            conc_rows.append({"condition": cond, "bigg_id": m,
                              "conc_mM": bennett[kegg][bi], "sd_mM": "",
                              "source": "bennett2009" + ("_zimmermann2014" if m == "oaa" else "")})

    for path, rows in ((a.out_kapp, kapp_rows), (a.out_conc, conc_rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote %d rows to %s" % (len(rows), os.path.relpath(path, REPO)))
    inc = [r for r in kapp_rows if r["included"] == "true"]
    print("included reaction-condition pairs: %d of %d" % (len(inc), len(kapp_rows)))
    for r in kapp_rows:
        print("  %-18s %-7s %-9s kapp %10s  %s" % (r["pathway"], r["reaction_id"], r["condition"],
                                                r["kapp_signed_1s"] or "-", r["exclusion_reason"]))


if __name__ == "__main__":
    main()
