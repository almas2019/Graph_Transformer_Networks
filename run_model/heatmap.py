import torch
import altair as alt
import pandas as pd

def load_layer_tensors(output_folder, num_layers, split_type, run_number):
    layer_tensors = []

    for x in range(num_layers):
        file_path = output_folder + f"layer_{x}_{split_type}_{run_number}.pt"
        tensor = torch.load(file_path,map_location=torch.device('cpu'))
        layer_tensors.extend(tensor)

    return layer_tensors

def create_heatmap(layer_tensors, labels, output_folder, split_type,run_number):
    combined_matrix = torch.stack(layer_tensors)

    data = []
    for i, layer in enumerate(combined_matrix):
        for j, value in enumerate(layer):
            data.append({'Layer': i, 'Label': labels[j], 'Value': value.item()})

    df = pd.DataFrame(data)

    heatmap = alt.Chart(df).mark_rect().encode(
        x='Layer:O',
        y='Label:O',
        color=alt.Color('Value:Q').scale(scheme="blues"),
    ).properties(
        title='Heatmap of Filters',
        width=200,
        height=150
    )

    heatmap.save(output_folder + f'{split_type}_{run_number}_heatmap.html')
    df.to_csv(output_folder + f'{split_type}_{run_number}_layer_avg_channel.csv')

data_folder = "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/"
def run_for_each_model(dataset_nm,labels,num_layers,split_type,run_number):
    output_folder = data_folder+dataset_nm+"/"
    layer_tensors = load_layer_tensors(output_folder, num_layers, split_type, run_number)
    create_heatmap(layer_tensors, labels, output_folder,split_type=split_type,run_number=run_number)

labels_imdb = ['MD', 'DM', 'MA', 'AM', 'I']
run_for_each_model("IMDB",labels_imdb,3,"train",10)

