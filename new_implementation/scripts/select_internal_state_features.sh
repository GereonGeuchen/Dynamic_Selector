#!/bin/bash
# Submit tsfresh/Random-Forest feature-selection jobs for every collected
# dimension and budget: 5, 10, 20, and 40 dimensions; budgets 50 through 950.
#
# Usage: bash scripts/select_internal_state_features.sh

set -euo pipefail

ENV_PATH="/rwthfs/rz/cluster/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPUS=30
DIMENSIONS=(5 10 20 40)
BUDGETS=$(seq 50 50 950)

mkdir -p "$WORKDIR/logs"

for DIMENSION in "${DIMENSIONS[@]}"; do
    INPUT_CSV="data/dim_${DIMENSION}/run_data/internal_state/internal_state_Non-elitist_B1000_${DIMENSION}D.csv"
    if [[ ! -f "$WORKDIR/$INPUT_CSV" ]]; then
        echo "Skipping ${DIMENSION}D: missing $INPUT_CSV" >&2
        continue
    fi

    for BUDGET in $BUDGETS; do
        sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=tsfresh_fid_B${BUDGET}_${DIMENSION}D
#SBATCH --output=${WORKDIR}/logs/tsfresh_fid_B${BUDGET}_${DIMENSION}D_%j.out
#SBATCH --error=${WORKDIR}/logs/tsfresh_fid_B${BUDGET}_${DIMENSION}D_%j.err
#SBATCH --time=13:00:00
#SBATCH --mem=12G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=${CPUS}

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

python src/select_internal_state_features.py "$INPUT_CSV" \\
    --dimension "$DIMENSION" \\
    --budget "$BUDGET" \\
    --n-jobs "$CPUS"
EOF
    done
done
