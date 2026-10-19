"""Assemble the supporting tables of the propagation from the outputs of its drivers.

Writes three comma-separated tables, numbered as in the shorter manuscript, into the folder
given by --out:

  S22 Table  downstream propagation results by reaction, the files
             downstream_reactions_<pathway><suffix>.csv of run_revision_downstream.jl
  S23 Table  dynamic propagation and rate-law sensitivity, the files
             dynamic_summary_<pathway><suffix>.csv of run_revision_dynamic.jl
  S24 Table  downstream substrate elasticities, the files
             downstream_elasticity_<pathway><suffix>.csv of run_revision_downstream.jl

Each table stacks the files of glycolysis and the TCA cycle in the order of the published
tables: the rows of the fits under the half-decade convention for glycolysis, then for the TCA
cycle, then the rows of the pre-specified default (cell gaussian_cest_RE_orgtype) for
glycolysis and for the TCA cycle, each file keeping its own row order.  Values are copied as
the drivers wrote them.  With the default suffix the three tables are byte-identical to the
tables of the concise manuscript (S38, S36 and S37 there); the shorter manuscript uses the
suffix _priorheld, the runs with the prior mean held (PB_EB_KAPPA0=1e-9).

Usage
  python3 -I scripts/build_propagation_tables.py --out <folder that holds the supporting tables> \\
      [--suffix _priorheld]
Only the standard library is needed.
"""
import argparse
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHWAYS = ["glycolysis", "tca"]
DEFAULT_CELL = "gaussian_cest_RE_orgtype"
# table number -> (output-file stem of the driver, line ending of the published table)
TABLES = {22: ("downstream_reactions", "\r\n"), 23: ("dynamic_summary", "\n"),
          24: ("downstream_elasticity", "\r\n")}


def table_name(k):
    return "S%d_Table" % k + ".csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder that holds the supporting tables")
    ap.add_argument("--suffix", default="", help="suffix of the driver outputs (default none)")
    a = ap.parse_args()
    for k, (stem, eol) in TABLES.items():
        header, parts = None, {}
        for pw in PATHWAYS:
            f = os.path.join(REPO, "output", pw, "results", "revision", "%s_%s%s.csv" % (stem, pw, a.suffix))
            with open(f, newline="") as fh:
                r = csv.reader(fh)
                h = next(r)
                if header is None:
                    header = h
                elif h != header:
                    raise SystemExit("%s has other columns than the glycolysis file" % f)
                parts[pw] = list(r)
        ci = header.index("cell")
        rows = ([row for pw in PATHWAYS for row in parts[pw] if row[ci] != DEFAULT_CELL] +
                [row for pw in PATHWAYS for row in parts[pw] if row[ci] == DEFAULT_CELL])
        with open(os.path.join(a.out, table_name(k)), "w", newline="") as fh:
            w = csv.writer(fh, lineterminator=eol)
            w.writerow(header)
            w.writerows(rows)
        print("wrote %s (%d rows from %s%s)" % (table_name(k), len(rows), stem, a.suffix))


if __name__ == "__main__":
    main()
