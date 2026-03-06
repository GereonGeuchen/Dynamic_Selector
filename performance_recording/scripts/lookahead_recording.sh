#!/bin/bash

# Path to your unpacked conda-pack environment
ENV_PATH="$HOME/general-env"

# Project directory (where your Python code and data are)
WORKDIR="$HOME/Dokumente/Dynamic_Selector/performance_recording/scripts"

# Path to your Python script
PY_SCRIPT="record_lookahead_performance.py"

# Create logs directory
mkdir -p "$WORKDIR/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=performance_lookahead
#SBATCH --output=${WORKDIR}/logs/performance_lookahead.out
#SBATCH --error=${WORKDIR}/logs/performance_lookahead.err
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate your packed conda environment
source $ENV_PATH/bin/activate

# Run the Python tuning script
python $PY_SCRIPT 
EOF
