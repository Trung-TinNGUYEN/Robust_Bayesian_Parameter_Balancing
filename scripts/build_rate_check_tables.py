"""Assemble the supporting tables of the rate checks from the outputs of their drivers.

Writes three comma-separated tables, numbered as in the shorter manuscript, into the folder
given by --out:

  S29 Table  coverage of predicted rates for held-out organisms, a copy of
             output/rate_checks/rate_holdout_summary.csv (run_revision_rate_holdout.jl)
  S30 Table  coverage of predicted rates in simulation with known parameters: the main run
             and the two sensitivity runs of run_revision_rate_simulation.jl stacked, with a
             first column `run` (main, em_budget_x4, truth_drawn), and with the medians over
             replicates of the fitted record scale, between-organism variances and prior
             scale, and the mean parameter-level coverage, of each model joined from the
             files rate_simulation_fits*.csv
  S31 Table  predicted rates at measured E. coli concentrations against in vivo apparent
             rates: the main run and the two concentration-error runs of
             run_revision_rate_invivo.jl stacked, with a first column
             `concentration_error_sd` (0, 0.2, 0.5)

Values are copied as the drivers wrote them, apart from the statistics joined into S30 Table.

Usage
  python3 -I scripts/build_rate_check_tables.py --out <folder that holds the supporting tables>
Only the standard library is needed.
"""
import argparse
import csv
import os
import shutil
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKS = os.path.join(REPO, "output", "rate_checks")

# table number -> (first column, [(value, output file)])
STACKED = {
    30: ("run", [("main", "rate_simulation_summary.csv"),
                 ("em_budget_x4", "rate_simulation_summary_em4.csv"),
                 ("truth_drawn", "rate_simulation_summary_truthdraw.csv")]),
    31: ("concentration_error_sd", [("0", "rate_invivo_rows.csv"),
                                    ("0.2", "rate_invivo_rows_concsd02.csv"),
                                    ("0.5", "rate_invivo_rows_concsd05.csv")]),
}
COPIED = {29: "rate_holdout_summary.csv"}
# Fitted hyperparameters joined into S30 Table: (output column, column of the fits file, statistic)
FITS = {"main": "rate_simulation_fits.csv", "em_budget_x4": "rate_simulation_fits_em4.csv",
        "truth_drawn": "rate_simulation_fits_truthdraw.csv"}
FIT_COLUMNS = [("median_c", "c", statistics.median),
               ("median_tau2_pooled", "tau2_pooled", statistics.median),
               ("median_tau2_kcat", "tau2_kcat", statistics.median),
               ("median_tau2_KM", "tau2_KM", statistics.median),
               ("median_lambda", "lambda", statistics.median),
               ("mean_param_cov95_curated", "param_cov95_curated_rows", statistics.mean)]


def fit_summaries():
    """(run, pathway, scenario, model) -> the statistics of FIT_COLUMNS over replicates."""
    out = {}
    for run, f in FITS.items():
        groups = {}
        with open(os.path.join(CHECKS, f), newline="") as fh:
            for r in csv.DictReader(fh):
                groups.setdefault((run, r["pathway"], r["scenario"], r["model"]), []).append(r)
        for key, rows in groups.items():
            out[key] = [repr(stat([float(r[col]) for r in rows])) for _, col, stat in FIT_COLUMNS]
    return out


def table_name(k):
    return "S%d_Table" % k + ".csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder that holds the supporting tables")
    a = ap.parse_args()
    for k, f in COPIED.items():
        shutil.copyfile(os.path.join(CHECKS, f), os.path.join(a.out, table_name(k)))
        print("wrote %s (copy of %s)" % (table_name(k), f))
    for k, (col, parts) in STACKED.items():
        header, rows = None, []
        for value, f in parts:
            with open(os.path.join(CHECKS, f), newline="") as fh:
                r = csv.reader(fh)
                h = next(r)
                if header is None:
                    header = h
                elif h != header:
                    raise SystemExit("%s has other columns than %s" % (f, parts[0][1]))
                rows += [[value] + row for row in r]
        if k == 30:
            fits = fit_summaries()
            hi = {c: header.index(c) for c in ("pathway", "scenario", "model")}
            header = header + [c for c, _, _ in FIT_COLUMNS]
            rows = [row + fits[(row[0], row[1 + hi["pathway"]], row[1 + hi["scenario"]],
                                row[1 + hi["model"]])] for row in rows]
        with open(os.path.join(a.out, table_name(k)), "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow([col] + header)
            w.writerows(rows)
        print("wrote %s (%d rows from %d files)" % (table_name(k), len(rows), len(parts)))


if __name__ == "__main__":
    main()
