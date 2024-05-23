#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=31500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH -J plot_loss
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
python plot_loss.py /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/IMDB/04_25_2024_13_26_FastGTN_IMDB_500_1_4.csv "Epoch" "Train Loss" 0 "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/IMDB/" "train_loss_imdb_run0_epochs500"
python plot_loss.py /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/ACM/04_25_2024_13_26_FastGTN_ACM_500_2_3.csv "Epoch" "Train Loss" 0 "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/ACM/" "train_loss_acm_run0_epochs500"
python plot_loss.py /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/DBLP/04_25_2024_13_26_FastGTN_DBLP_500_2_4.csv "Epoch" "Train Loss" 0 "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/DBLP/" "train_loss_dblp_run0_epochs500"
python plot_loss.py /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib5/04_25_2024_13_29_FastGTN_graph_het_subsampled_tcell_fib5_500_2_2.csv "Epoch" "Train Loss" 0 "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib5/" "train_loss_tcell_fib5_run0_epochs500"
python plot_loss.py //home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib4/05_02_2024_11_50_FastGTN_graph_het_subsampled_tcell_fib4_600_2_2.csv "Epoch" "Train Loss" 0 "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib4/" "train_loss_tcell_fib4_run0_epochs600"
python plot_loss.py /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib3/04_25_2024_13_29_FastGTN_graph_het_subsampled_tcell_fib3_500_2_2.csv "Epoch" "Train Loss" 0 "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib3/" "train_loss_tcell_fib3_run0_epochs500"
python plot_loss.py /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib2/04_25_2024_13_29_FastGTN_graph_het_subsampled_tcell_fib2_500_2_2.csv "Epoch" "Train Loss" 0 "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib2/" "train_loss_tcell_fib2_run0_epochs500"
python plot_loss.py /home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib/04_25_2024_13_29_FastGTN_graph_het_subsampled_tcell_fib_500_2_2.csv "Epoch" "Train Loss" 0 "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib/" "train_loss_tcell_fib_run0_epochs500"

#python plot_log_vals.py "/home/almas/projects/def-gregorys/almas/OpenHGNN/try_hgnn/data/2024-Apr-10_1408_fastGTN_node_classification_tcell_fib4.csv" "Epoch" "Train Loss" "/project/def-gregorys/almas/OpenHGNN/try_hgnn/img" "train_loss_tcell_fib"


