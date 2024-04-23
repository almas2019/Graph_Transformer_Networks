import re
import csv
import argparse
import os
from datetime import datetime

def extract_filename_info(text):
    """Extract filename information from the provided text."""
    # Define the regex pattern to extract the required information
    pattern = r'Starting run at: (.*?)(?:\n.*?){25}Namespace\(model=\'(.*?)\',\s*dataset=\'(.*?)\',.*?non_local=(.*?),\s*num_layers=(.*?),\s*num_channels=(.*?)\)'
    #(.*?): This part of the pattern captures any characters (except newline) lazily, captures date time info
    # (?:\n.*?): newline and don't capture This part of the pattern matches a newline character followed by any characters (except newline) lazily. (?: ... ) is a non-capturing group used here.
    # {25} skip 25 lines (i.e. match the non capture 25 times)
    # Namespace\(model=\'(.*?)\',\s*dataset=\'(.*?)\',.*?non_local=(.*?),\s*num_layers=(.*?),\s*num_channels=(.*?)\): This part of the pattern captures the model, dataset, non_local, num_layers, and num_channels information. It matches the literal string "Namespace(model='" followed by capturing the model name, "dataset='" followed by capturing the dataset name, and so on until the closing parenthesis.
    match = re.search(pattern, text, re.DOTALL) # search for regular function, The re.DOTALL flag is used to make the dot (.) in the pattern match any character, including newline.
    if match:
        # Extracting individual components
        run_time = match.group(1)
        model = match.group(2)
        dataset = match.group(3)
        non_local = match.group(4)
        num_layers = match.group(5)
        num_channels = match.group(6)

        # Parse run time to get a file-friendly datetime format
        datetime_obj = datetime.strptime(run_time, '%a %d %b %Y %I:%M:%S %p %Z')
        datetime_str_formatted = datetime_obj.strftime('%Y-%b-%d_%H%M')

        # Construct filename info
        filename_info = f"{datetime_str_formatted}_{model}_{dataset}_nonlocal{non_local}_layers{num_layers}_channels{num_channels}.csv"
        print("extracted file name info!")
        return filename_info
    else:
        print("Filename information could not be extracted.")
        return None


def parse_log(log_file):
    """Parse log file and extract relevant data."""
    # Define the regex patterns for data extraction
    regex_epoch = re.compile(r'Epoch\s+(\d+)')
    regex_train_loss = re.compile(r'Train - Loss: ([\d.]+)')
    regex_valid_loss = re.compile(r'Valid - Loss: ([\d.]+)')
    regex_test_loss = re.compile(r'Test - Loss: ([\d.]+)')
    regex_f1_scores = re.compile(r'Mode:(\w+), Macro_F1: ([\d.]+), Micro_F1: ([\d.]+)')

    # Initialize lists to store extracted data
    epochs = []
    train_losses = []
    valid_losses = []
    test_losses = []
    train_macro_f1s = []
    train_micro_f1s = []
    valid_macro_f1s = []
    valid_micro_f1s = []
    test_macro_f1s = []
    test_micro_f1s = []

    # Read the log file
    with open(log_file, 'r') as file:
        log_data = file.readlines()

        # Extract data
        for line in log_data:
            epoch_match = regex_epoch.search(line)
            if epoch_match:
                epoch = epoch_match.group(1)
                train_loss_match = regex_train_loss.search(line)
                valid_loss_match = regex_valid_loss.search(line)
                test_loss_match = regex_test_loss.search(line)
                f1_scores_matches = regex_f1_scores.findall(line)
                if train_loss_match and valid_loss_match and test_loss_match and f1_scores_matches:
                    train_loss = train_loss_match.group(1)
                    valid_loss = valid_loss_match.group(1)
                    test_loss = test_loss_match.group(1)
                    for match in f1_scores_matches:
                        mode, macro_f1, micro_f1 = match
                        if mode == 'train':
                            train_macro_f1s.append(macro_f1)
                            train_micro_f1s.append(micro_f1)
                        elif mode == 'valid':
                            valid_macro_f1s.append(macro_f1)
                            valid_micro_f1s.append(micro_f1)
                        elif mode == 'test':
                            test_macro_f1s.append(macro_f1)
                            test_micro_f1s.append(micro_f1)
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    test_losses.append(test_loss)
                else:
                    print("Could not extract necessary information from the line:", line)
    print("extracted necessary info!")
    return epochs, train_losses, valid_losses, test_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s, test_macro_f1s, test_micro_f1s


def write_csv(output_folder, filename_info, epochs, train_losses, valid_losses, test_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s, test_macro_f1s, test_micro_f1s):
    os.makedirs(output_folder, exist_ok=True)
    if epochs:
        output_file = os.path.join(output_folder, filename_info)
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train Loss', 'Valid Loss', 'Test Loss', 'Train Macro F1', 'Train Micro F1', 'Valid Macro F1', 'Valid Micro F1', 'Test Macro F1', 'Test Micro F1'])
            writer.writerows(zip(epochs, train_losses, valid_losses, test_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s, test_macro_f1s, test_micro_f1s))
        print(f"Data has been extracted and saved to {output_file}")
    else:
        print("No data found in the log file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse log file and write to CSV')
    parser.add_argument('input_file', help='Input log file path')
    parser.add_argument('output_folder', help='Output folder path')
    args = parser.parse_args()

    with open(args.input_file, 'r') as file:
        log_text = file.read()

    filename_info = extract_filename_info(log_text)
    if filename_info:
        epochs, train_losses, valid_losses, test_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s, test_macro_f1s, test_micro_f1s = parse_log(args.input_file)
        write_csv(args.output_folder, filename_info, epochs, train_losses, valid_losses, test_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s, test_macro_f1s, test_micro_f1s)
    else:
        print("Filename information extraction failed.")
input_file="/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/run_model/slurm-29692450.out"
output_folder= "/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/graph_het_subsampled_tcell_fib5/"
with open(input_file, 'r') as file:
        log_text = file.read()
filename_info = extract_filename_info(log_text)
if filename_info:
    epochs, train_losses, valid_losses, test_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s, test_macro_f1s, test_micro_f1s = parse_log(input_file)
    write_csv(output_folder, filename_info, epochs, train_losses, valid_losses, test_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s, test_macro_f1s, test_micro_f1s)