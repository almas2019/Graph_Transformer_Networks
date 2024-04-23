#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=63500       # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH -J extract_scen5dat
#SBATCH --output=R-%x.%j.out #customize the output name %j gives job id and 
#SBATCH --error=R-%x.%j.err
#SBATCH -N 1
# ---------------------------------------------------------------------
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# activate your virtual environment
module load python/3.10
module load scipy-stack
source /project/def-gregorys/almas/spgraph_env/bin/activate


cd /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/run_model
python extract_data_for_plot.py /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/run_model/slurm-29692450.out /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib5/
#python extract_log_data.py /home/almas/projects/def-gregorys/almas/OpenHGNN/openhgnn/output/fastGTN/fastGTN-Apr-10-2024_14-08-36.log /home/almas/projects/def-gregorys/almas/OpenHGNN/try_hgnn/data/ 


