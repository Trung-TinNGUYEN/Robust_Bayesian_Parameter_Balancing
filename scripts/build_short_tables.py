"""Assemble the supporting tables of the shorter manuscript that come from real-data fits.

Writes into the folder given by --out the tables listed in TABLES below, numbered as in the
shorter manuscript, from the outputs of the analysis scripts under output/.  Each table is a
copy of one output file, or a stack of the per-pathway files of one script in the order
glycolysis, TCA cycle, pentose phosphate pathway, shikimate pathway (for the pathways the
script was run on), or, for S21 Table, a join described in invivo_join below.  Line endings and
number formats follow the published tables, so that the tables reproduce byte for byte from
unchanged outputs.

Other tables have their own helpers or need no real-data fit:
  S1, S26      a copy of the released set (done here), then run_revision_record_range.jl adds
               the range of each quantity's records
  S22 to S24   scripts/build_propagation_tables.py
  S29 to S31   scripts/build_rate_check_tables.py
  S6, S8, S9, S25, S28  counts of records, synthetic simulations and benchmarks, which do not
               depend on the thermodynamic information of the fits

Usage
  python3 -I scripts/build_short_tables.py --out <folder that holds the supporting tables>
Only the standard library is needed.
"""
import argparse
import csv
import math
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PW2 = ["glycolysis", "tca"]
PW4 = ["glycolysis", "tca", "pentose_phosphate", "shikimate"]
CRLF, LF = "\r\n", "\n"


def rev(pw, stem):
    return os.path.join(REPO, "output", pw, "results", "revision", "%s_%s.csv" % (stem, pw))


# table number -> how it is made
TABLES = {
    1: ("copy", os.path.join("output", "paramset", "balanced_parameters.csv")),
    2: ("copy", os.path.join("output", "paramset", "balanced_parameters_by_organism.csv")),
    3: ("stack", "inference_bootstrap", PW2, CRLF),
    4: ("stack", "inference_paired", PW2, CRLF),
    5: ("stack", "hetero_inference", PW4, CRLF),
    7: ("stack_with_pathway", "realdata_residuals", PW4, CRLF),
    10: ("stack", "realdata_summary", PW2, CRLF),
    11: ("stack", "sigmacal_summary", PW2, LF),
    12: ("stack", "realdata_summary", ["pentose_phosphate"], LF),
    13: ("stack", "realdata_summary", ["shikimate"], LF),
    14: ("stack", "sigmacal_summary", ["pentose_phosphate"], LF),
    15: ("stack", "sigmacal_summary", ["shikimate"], LF),
    16: ("stack", "hetero_summary", PW4, CRLF),
    17: ("stack", "hetero_strata", PW4, CRLF),
    18: ("stack", "hetero_coverage", PW4, CRLF),
    19: ("stack", "transfer_summary", PW4, CRLF),
    20: ("copy", os.path.join("output", "transfer_diagnostics", "transfer_diag_pairs.csv")),
    21: ("invivo_join",),
    26: ("copy", os.path.join("output", "paramset_wildtype", "balanced_parameters.csv")),
    27: ("copy", os.path.join("output", "paramset_wildtype", "balanced_parameters_by_organism.csv")),
}

MODEL_LABELS = {
    "fe_org": "fixed effects, organism level (consensus)",
    "fe_rec": "fixed effects, record level",
    "pd_org": "pre-specified default, organism level (previously unmeasured organism)",
    "pd_rec": "pre-specified default, record level",
}


def table_name(k):
    return "S%d_Table" % k + ".csv"


def read(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def write(path, rows, eol):
    with open(path, "w", newline="") as fh:
        csv.writer(fh, lineterminator=eol).writerows(rows)


def stack(stem, pathways, with_pathway=False):
    header, out = None, []
    for pw in pathways:
        r = read(rev(pw, stem))
        h = (["pathway"] + r[0]) if with_pathway else r[0]
        if header is None:
            header = h
        elif h != header:
            raise SystemExit("%s has other columns than the first file" % rev(pw, stem))
        out += [([pw] + row) if with_pathway else row for row in r[1:]]
    return [header] + out


def g4(x):
    return "%.4g" % x


def invivo_join():
    """One row per matched forward reaction (with a kmax) and predictive object."""
    davidi = list(csv.DictReader(open(os.path.join(REPO, "data", "external", "davidi2016_kmax_matched.csv"),
                                      newline="")))
    rows_iv = list(csv.DictReader(open(os.path.join(REPO, "output", "invivo", "invivo_check_rows.csv"),
                                       newline="")))
    iv = {(r["shift"], r["pathway"], r["Q_row"], r["model"]): r for r in rows_iv}
    shift_x = sorted({r["shift"] for r in rows_iv if r["shift"] != "kmax"})
    if len(shift_x) != 1:
        raise SystemExit("expected one shifted target in invivo_check_rows.csv, found %s" % shift_x)
    sx = shift_x[0]
    quant = {}
    for pw in PW4:
        for r in csv.DictReader(open(rev(pw, "ecoli_holdout_quantities"), newline="")):
            if r["direction"] == "forward":
                quant[(pw, r["Q_row"])] = r
    header = ["pathway", "reaction_id", "reaction_name", "Q_row", "ec_code", "direction",
              "davidi_reaction_id", "match_quality", "kmax_1s", "kmax_source", "mw_used_kDa",
              "isozymes_in_iJO1366", "protein_complex", "has_kcat_records", "in_direct_1s",
              "in_all_matched", "n_records_train", "n_organisms_train", "n_ecoli_records",
              "n_ecoli_mutant_records", "model", "model_label", "pred_lo_1s", "pred_median_1s",
              "pred_hi_1s", "width95_decades", "inside95", "above95", "below95",
              "log10_kmax_over_median", "lpd_gauss", "crps", "pit",
              "inside95_kmax_x1.9", "above95_kmax_x1.9", "notes"]
    out = [header]
    for d in davidi:
        if d["direction"] != "forward" or not d["kmax_1s"]:
            continue
        q = quant[(d["pathway"], d["Q_row"])]
        for m in ("fe_org", "fe_rec", "pd_org", "pd_rec"):
            a = iv[("kmax", d["pathway"], d["Q_row"], m)]
            b = iv[(sx, d["pathway"], d["Q_row"], m)]
            out.append([d["pathway"], d["reaction_id"], d["reaction_name"], d["Q_row"], d["ec_code"],
                        d["direction"], d["davidi_reaction_id"], d["match_quality"], d["kmax_1s"],
                        d["kmax_source"], g4(float(d["mw_used_kDa"])) if d["mw_used_kDa"] else "",
                        d["isozymes_in_iJO1366"],
                        d["protein_complex"], d["has_kcat_records"], a["in_direct_1s"],
                        a["in_all_matched"], q["n_records_train"], q["n_organisms_train"],
                        q["n_ecoli"], q["n_ecoli_mutant"], m, MODEL_LABELS[m],
                        g4(math.exp(float(a["pred_lo"]))), g4(math.exp(float(a["pred_median"]))),
                        g4(math.exp(float(a["pred_hi"]))), g4(float(a["width95_decades"])),
                        a["inside95"], a["above95"], a["below95"],
                        g4(float(a["log10_target_over_median"])), g4(float(a["lpd_gauss"])),
                        g4(float(a["crps"])), g4(float(a["pit"])), b["inside95"], b["above95"],
                        d["notes"]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder that holds the supporting tables")
    ap.add_argument("--only", default="", help="comma-separated table numbers (default all)")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x.strip()}
    for k, spec in TABLES.items():
        if only and k not in only:
            continue
        dest = os.path.join(a.out, table_name(k))
        kind = spec[0]
        if kind == "copy":
            shutil.copyfile(os.path.join(REPO, spec[1]), dest)
        elif kind == "stack":
            write(dest, stack(spec[1], spec[2]), spec[3])
        elif kind == "stack_with_pathway":
            write(dest, stack(spec[1], spec[2], with_pathway=True), spec[3])
        elif kind == "invivo_join":
            write(dest, invivo_join(), LF)
        print("wrote %s (%s)" % (table_name(k), kind if kind == "invivo_join" else
                                 "%s %s" % (kind, spec[1] if kind != "copy" else os.path.basename(spec[1]))))


if __name__ == "__main__":
    main()
