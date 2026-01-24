#!/bin/bash

# Path to your unpacked conda-pack environment
ENV_PATH="$HOME/general-env"

# Project directory (where your Python code and data are)
WORKDIR="$HOME/Dokumente/Dynamic_Selector/model_optimisation/scripts"

# Path to your Python script
PY_SCRIPT="switch_model_optimisation.py"
sbatch <<EOF
#!/bin/bash
#SBATCH -A p0026688
#SBATCH --job-name=switch_optimisation_algo_features_lookahead
#SBATCH --output=${WORKDIR}/logs/switch_optimisation_algo_features_lookahead.out
#SBATCH --error=${WORKDIR}/logs/switch_optimisation_algo_features_lookahead.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=5

# Go to the working directory
cd $WORKDIR

# Activate your packed conda environment
source $ENV_PATH/bin/activate

# Run the Python tuning script
python $PY_SCRIPT $BUDGET
EOF