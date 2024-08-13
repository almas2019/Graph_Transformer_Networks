import csv
import pandas as pd 
import numpy as np 
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os  

def read_data(input_file): 
    return pd.read_csv(input_file)  

def line_loss_alt(data, X, Y, output_folder, output_file, run):
    filtered_data = data[data['Run'] == run]
    print(filtered_data.head())
    chart = alt.Chart(filtered_data).mark_line().encode(
        x=X + ":Q",
        y=Y + ":Q",
    ).properties(title=output_file)
    chart.save(os.path.join(output_folder, output_file + ".html"))  

def line_loss_sns(data, X, Y, output_folder, output_file, run):
    filtered_data = data[data['Run'] == run]
    sns_plot = sns.lineplot(data=filtered_data, x=X, y=Y)
    plt.savefig(os.path.join(output_folder, output_file + ".svg"))  
    plt.savefig(os.path.join(output_folder, output_file + ".png"))  
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse log file and write to CSV')
    parser.add_argument('input_file', help='Input log file path')
    parser.add_argument('X', help="X axis")
    parser.add_argument('Y', help="Y axis")
    parser.add_argument('run', help="Run column",type=int)
    parser.add_argument('output_folder', help='Output folder path')
    parser.add_argument('output_file', help='Output file path and name')

    args = parser.parse_args()

    data = read_data(args.input_file)  

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    line_loss_alt(data=data, X=args.X, Y=args.Y, output_folder=args.output_folder, output_file=args.output_file, run=args.run)
    line_loss_sns(data=data, X=args.X, Y=args.Y, output_folder=args.output_folder, output_file=args.output_file, run=args.run)


# data= read_data("/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/IMDB/04_25_2024_08_52_FastGTN_IMDB_50_1_4.csv")
# line_loss_alt(data=data, X="Epoch", Y="Train Loss", run=9, output_folder="/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/IMDB/", output_file="train_loss_tcell_imdb_run10")
# line_loss_alt(data=data, X="Epoch", Y="Train Loss", 9 output_folder="/project/def-gregorys/almas/OpenHGNN/try_hgnn/img/", output_file="tcell_1_train_loss_curve")
# line_loss_sns(data=data, X="Epoch", Y="Valid Loss", output_folder="/project/def-gregorys/almas/OpenHGNN/try_hgnn/img/", output_file="tcell_1_valid_loss_curve")
# line_loss_sns(data=data, X="Epoch", Y="Train Loss", output_folder="/project/def-gregorys/almas/OpenHGNN/try_hgnn/img/", output_file="tcell_1_train_loss_curve")
#line_loss_sns(data=data, X="Epoch", Y="Train Loss", run=9, output_folder="/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/IMDB/", output_file="train_loss_tcell_imdb_run10")