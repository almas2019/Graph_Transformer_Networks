import torch
import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def load_layer_tensors(output_folder, num_layers, split_type, run_number):
    layer_tensors = []

    for x in range(num_layers):
        file_path = output_folder + f"layer_{x}_{split_type}_{run_number}.pt"
        tensor = torch.load(file_path, map_location=torch.device('cpu'))
        layer_tensors.append(tensor)

    return layer_tensors

def compute_layer_averages(layer_tensors):
    layer_averages = []

    # Compute the average across tensors for each layer
    for layer in layer_tensors:
        layer_average = torch.mean(torch.stack(tuple(layer)), dim=0)  # Convert list to tuple
        layer_averages.append(layer_average)

    return layer_averages

def create_heatmap(layer_averages, labels, output_folder, split_type, run_number):
    combined_matrix = torch.stack(layer_averages)

    data = []
    for i, layer in enumerate(combined_matrix):
        for j, value in enumerate(layer):
            data.append({'Layer': i, 'Label': labels[j], 'Value': value.item()})

    df = pd.DataFrame(data)
    # Convert the "Label" column to a categorical data type with the specified order
    df['Label'] = pd.Categorical(df['Label'], categories=labels, ordered=True)

    # Altair heatmap
    heatmap = alt.Chart(df).mark_rect().encode(
        x='Layer:O',
        y=alt.Y('Label:O', sort=None),  # Disable alphabetical sorting
        color=alt.Color('Value:Q').scale(scheme="blues"),
    ).properties(
        title='Heatmap of Filters',
        width=200,
        height=150
    )
    heatmap.save(output_folder + f'{split_type}_{run_number}_heatmap_500_epochs.html') # add epoch parameter after

    # Seaborn heatmap
    plt.figure(figsize=(8, 6))
    df_pivot = df.pivot(index='Label', columns='Layer', values='Value')  # Pivot the DataFrame
    sns.heatmap(data=df_pivot,
                cmap='Blues',
                cbar_kws={'label': 'Value'},
                annot=True,
                fmt=".2f")
    
    # Rotate y-axis labels to make them horizontal
    plt.yticks(rotation=0)
    plt.title('Heatmap of Filters')
    plt.xlabel('Layer')
    plt.ylabel('Label')

    # Save as PNG
    plt.savefig(output_folder + f'heatmap_epoch_500_{split_type}_{run_number}.png')

    # Save as SVG
    plt.savefig(output_folder + f'heatmap_epoch_500_{split_type}_{run_number}.svg', format='svg')

    # Clear the plot for the next iteration
    plt.clf()
    df.to_csv(output_folder + f'{split_type}_{run_number}_layer_avg_channel_500_epochs.csv')

data_folder = "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/"

def run_for_each_model(dataset_nm, labels, num_layers, split_type, run_number):
    output_folder = data_folder + dataset_nm + "/"
    layer_tensors = load_layer_tensors(output_folder, num_layers, split_type, run_number)
    layer_averages = compute_layer_averages(layer_tensors)
    create_heatmap(layer_averages, labels, output_folder, split_type=split_type, run_number=run_number)

labels_imdb = ['MD', 'DM', 'MA', 'AM', 'I']
labels_dblp = ['PA', 'AP', 'PC', 'CP', 'I']
labels_acm = ['PA', 'AP', 'PS', 'SP', 'I']
labels_graph_het_subsampled_tcell_fib4 = ['t-t', 'f-f', 'bridge', 't-f', 'I']
labels_graph_het_subsampled_tcell_fib5 = ['t-t', 'f-f', 'bridge', 't-f', 'I']
labels_graph_het_subsampled_tcell_fib3 = ['t-t', 'f-f', 't-f','I']
labels_graph_het_subsampled_tcell_fib2 = ['t-t', 'f-f', 't-f','f-t','I']
labels_graph_het_subsampled_tcell_fib = ['t-t', 'f-f', 't-f','I']
run_for_each_model("IMDB", labels_imdb, 4, "train", 1)
run_for_each_model("DBLP", labels_dblp, 4, "train", 1)
run_for_each_model("ACM", labels_acm, 3, "train", 1)
run_for_each_model("graph_het_subsampled_tcell_fib4", labels_graph_het_subsampled_tcell_fib4, 2, "train", 1)
run_for_each_model("graph_het_subsampled_tcell_fib5", labels_graph_het_subsampled_tcell_fib5, 2, "train", 1)
run_for_each_model("graph_het_subsampled_tcell_fib3", labels_graph_het_subsampled_tcell_fib3, 2, "train", 1)
run_for_each_model("graph_het_subsampled_tcell_fib2", labels_graph_het_subsampled_tcell_fib2, 2, "train", 1)
run_for_each_model("graph_het_subsampled_tcell_fib", labels_graph_het_subsampled_tcell_fib, 2, "train", 1)

