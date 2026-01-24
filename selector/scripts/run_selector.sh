#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory (adjust as needed) ===
WORKDIR="$HOME/Dokumente/Dynamic_Selector/selector/scripts"

# === Python script name ===
PY_SCRIPT="selector.py"

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=selector_with_lookahead
#SBATCH --output=${WORKDIR}/logs/selector_with_lookahead.out
#SBATCH --error=${WORKDIR}/logs/selector_with_lookahead.err
#SBATCH --time=10:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your ELA calculation script with BUDGET
python $PY_SCRIPT 

EOF

