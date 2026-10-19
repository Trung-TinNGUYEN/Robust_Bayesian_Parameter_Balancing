#!/usr/bin/env bash
# Reproduce the paper's released specification for one pathway from the shipped inputs:
# replication error scale, organism-by-type random intercept, per-type between-source
# variance with the pooled fallback, prior mean held at the thermodynamic prior.
# Usage: bash examples/released_specification.sh [pathway] [output directory]
#   pathway: glycolysis, tca, pentose_phosphate or shikimate (default shikimate)
#   set JULIA to the julia executable if it is not on PATH
set -euo pipefail
PW="${1:-shikimate}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${2:-$PWD/pb_example_$PW}"
JULIA="${JULIA:-julia}"
"$JULIA" --project="$ROOT" "$ROOT/examples/export_cli_inputs.jl" "$ROOT" "$PW" "$OUT"
"$JULIA" --project="$ROOT" "$ROOT/bin/parameter-balance.jl" balance-hierarchical \
  --Q "$OUT/Q.csv" --x "$OUT/x.csv" --Cx "$OUT/Cx.csv" \
  --q-prior "$OUT/q_prior.csv" --C-prior "$OUT/C_prior.csv" \
  --group "$OUT/groups.csv" --replicate-group "$OUT/replicates.csv" --type "$OUT/types.csv" \
  --error-scale replication \
  --output "$OUT/basis_posterior.csv" --output-cov "$OUT/basis_covariance.csv" \
  --predict "$OUT/new_source_intervals.csv"
echo "Outputs in $OUT; labels.csv maps each row of Q to its organism, type, EC number and substrate."
