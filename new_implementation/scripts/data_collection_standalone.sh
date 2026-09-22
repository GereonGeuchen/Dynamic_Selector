#!/bin/bash
# Submit one standalone-baseline collection job per algorithm.
#
# Usage:
#   ./data_collection_standalone.sh [true|false] [dimension] [function_ids]
#
# Examples:
#   ./data_collection_standalone.sh false 40
#   ./data_collection_standalone.sh true 5 1,2,3
#
# The Python defaults are used: 20 repetitions, instances 6 and 7, and a
# total budget of 1000 evaluations.  Results are written below
# data/dim_<dimension>_standalone or data/dim_<dimension>_ma_standalone.
set -euo pipefail

ENV_PATH="/rwthfs/rz/cluster/home/p0027894/Dynamic_Selector/envs/dynamic_selector_venv"
WORKDIR="/hpcwork/p0027894/Dynamic_Selector/new_implementation"
USE_MA="${1:-false}"
DIMENSION="${2:-40}"
FUNCTION_IDS="${3:-$(seq -s, 1 24)}"
ALGORITHMS=("DE" "MLSL" "PSO" "BFGS" "Non-elitist" "Elitist")

case "$USE_MA" in
  true|false) ;;
  *)
    echo "Usage: $0 [true|false] [dimension] [function_ids]" >&2
    exit 1
    ;;
esac

mkdir -p "$WORKDIR/logs"

for algorithm in "${ALGORITHMS[@]}"; do
  sbatch <<EOF
#!/bin/bash
#SBATCH -A p0027894
#SBATCH --job-name=standalone_${algorithm}_D${DIMENSION}_ma${USE_MA}
#SBATCH --output=${WORKDIR}/logs/standalone_${algorithm}_D${DIMENSION}_ma${USE_MA}_%j.out
#SBATCH --error=${WORKDIR}/logs/standalone_${algorithm}_D${DIMENSION}_ma${USE_MA}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=c25ms
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$WORKDIR"
module load GCCcore/12.2.0
module load Python/3.10.8
source "$ENV_PATH/bin/activate"

# Arguments: A1 budget, algorithms, dimension, FIDs, suffix, ManyAffine,
# standalone mode, and IIDs.  B0 is recorded automatically in standalone mode.
python data_collection.py 0 "$algorithm" "$DIMENSION" "$FUNCTION_IDS" "" "$USE_MA" true "1,2,3,4,5,6,7"
EOF
done
