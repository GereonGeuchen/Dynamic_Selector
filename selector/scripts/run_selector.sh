#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory (adjust as needed) ===
WORKDIR="$HOME/Dokumente/Dynamic_Selector/selector/scripts"

# === Python script name ===
PY_SCRIPT="selector.py"

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

for i in $(seq 0 19); do
  EPMS+=($((i)))
done

EPMS+=(-1)

for NUM_LOOKAHEAD_EPMS in "${EPMS[@]}"; do

sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=selector_lookahead_${NUM_LOOKAHEAD_EPMS}_auc
#SBATCH --output=${WORKDIR}/logs/selector_lookahead_${NUM_LOOKAHEAD_EPMS}_auc.out
#SBATCH --error=${WORKDIR}/logs/selector_lookahead_${NUM_LOOKAHEAD_EPMS}_auc.err
#SBATCH --time=10:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your ELA calculation script with BUDGET
python $PY_SCRIPT $NUM_LOOKAHEAD_EPMS

EOF

done



