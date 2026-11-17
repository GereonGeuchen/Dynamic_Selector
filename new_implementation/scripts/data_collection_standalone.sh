#!/bin/bash
# Submit one standalone-baseline collection job per algorithm.
#
# Usage:
#   ./data_collection_standalone.sh [dimension] [function_ids]
#
# Examples:
#   ./data_collection_standalone.sh 40
#   ./data_collection_standalone.sh 5 1,2,3
#
# The Python defaults are used: 20 repetitions, instances 6 and 7, and a
# total budget of 1000 evaluations.  Results are written below
# data/dim_<dimension>_standalone.
set -euo pipefail

ENV_PATH="/rwthfs/rz/cluster/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
DIMENSION="${1:-40}"
FUNCTION_IDS="${2:-$(seq -s, 1 24)}"
ALGORITHMS=("DE" "MLSL" "PSO" "BFGS" "Non-elitist" "Elitist")

mkdir -p "$WORKDIR/logs"

for algorithm in "${ALGORITHMS[@]}"; do
  sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=standalone_${algorithm}_D${DIMENSION}
#SBATCH --output=${WORKDIR}/logs/standalone_${algorithm}_D${DIMENSION}_%j.out
#SBATCH --error=${WORKDIR}/logs/standalone_${algorithm}_D${DIMENSION}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

# Arguments: A1 budget, algorithms, dimension, FIDs, suffix, standalone mode,
# and IIDs. B0 is recorded automatically in standalone mode.
PYTHONPATH="$WORKDIR/src" python -m data_collection.data_collection 0 "$algorithm" "$DIMENSION" "$FUNCTION_IDS" "" true "6,7"
EOF
done
