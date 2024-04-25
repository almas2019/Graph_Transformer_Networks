import torch
import numpy as np
import torch.nn as nn
from model_gtn import GTN
from model_fastgtn import FastGTNs
import pickle
import argparse
from torch_geometric.utils import f1_score, add_self_loops
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, _norm
import copy
import os
from datetime import datetime
import csv	

def write_metric_csv(output_folder, date_time, args, all_epochs_list, all_runs_list, all_train_losses, all_valid_losses, all_test_losses,
                     all_train_macro_f1s, all_train_micro_f1s, all_valid_macro_f1s, all_valid_micro_f1s, all_test_macro_f1s, all_test_micro_f1s):
    os.makedirs(output_folder, exist_ok=True)
    if args.non_local:
        metric_out_file = f'{date_time}_{args.model}_{args.dataset}_{args.epoch}_{args.num_channels}_{args.num_layers}_non_local.csv'
    else:
        metric_out_file = f'{date_time}_{args.model}_{args.dataset}_{args.epoch}_{args.num_channels}_{args.num_layers}.csv'
    output_file = os.path.join(output_folder, metric_out_file)
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Run', 'Epoch', 'Train Loss', 'Valid Loss', 'Test Loss', 'Train Macro F1', 'Train Micro F1',
                         'Valid Macro F1', 'Valid Micro F1', 'Test Macro F1', 'Test Micro F1'])
        for run_idx, epochs_list in enumerate(all_epochs_list):
            for epoch_idx, epoch in enumerate(epochs_list):
                writer.writerow([run_idx, epoch, all_train_losses[run_idx][epoch_idx], all_valid_losses[run_idx][epoch_idx],
                                 all_test_losses[run_idx][epoch_idx], all_train_macro_f1s[run_idx][epoch_idx], all_train_micro_f1s[run_idx][epoch_idx],
                                 all_valid_macro_f1s[run_idx][epoch_idx], all_valid_micro_f1s[run_idx][epoch_idx],
                                 all_test_macro_f1s[run_idx][epoch_idx], all_test_micro_f1s[run_idx][epoch_idx]])
    print(f"Metrics data has been saved to {output_file}")


# Main function
if __name__ == '__main__':
    # Initialize random seed
    init_seed(seed=777)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GTN', help='Model')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--epoch', type=int, default=200, help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64, help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2, help='number of channels')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=1, help='number of GT/FastGT layers')
    parser.add_argument('--runs', type=int, default=10, help='number of runs')
    parser.add_argument("--channel_agg", type=str, default='concat')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")
    
    # Configurations for FastGTNs
    parser.add_argument("--non_local", action='store_true', help="use non local operations")
    parser.add_argument("--non_local_weight", type=float, default=0, help="weight initialization for non local operations")
    parser.add_argument("--beta", type=float, default=0, help="beta (Identity matrix)")
    parser.add_argument('--K', type=int, default=1, help='number of non-local negibors')
    parser.add_argument("--pre_train", action='store_true', help="pre-training FastGT layers")
    parser.add_argument('--num_FastGTN_layers', type=int, default=1, help='number of FastGTN layers')
    parser.add_argument('--save_metrics', action='store_true', help="save metrics?")
    parser.add_argument('--layer_split', type=str, default='train', help="which layer to save weights")

    # Get current date and time
    now = datetime.now() 
    date_time = now.strftime("%m_%d_%Y_%H_%M")
    print("date and time:", date_time)
    args = parser.parse_args()
    print(args)
    # Initialize lists to store metrics for all epochs and runs
    all_epochs_list = []
    all_runs_list = []
    all_train_losses = []
    all_valid_losses = []
    all_test_losses = []
    all_train_macro_f1s = []
    all_train_micro_f1s = []
    all_valid_macro_f1s = []
    all_valid_micro_f1s = []
    all_test_macro_f1s = []
    all_test_micro_f1s = []
    
    # Determine output folder based on whether non-local operations are used
    if args.non_local:
        output_folder = os.path.join('../Graph_Transformer_Networks/data/non_local/', args.dataset)
    else:
        output_folder = os.path.join('../Graph_Transformer_Networks/data/', args.dataset)

    # Load data
    with open('../Graph_Transformer_Networks/data/%s/node_features.pkl' % args.dataset,'rb') as f:
        node_features = pickle.load(f)
    with open('../Graph_Transformer_Networks/data/%s/edges.pkl' % args.dataset,'rb') as f:
        edges = pickle.load(f)
    with open('../Graph_Transformer_Networks/data/%s/labels.pkl' % args.dataset,'rb') as f:
        labels = pickle.load(f)
    
    # Process data based on dataset
    if args.dataset == 'PPI':
        with open('../data/%s/ppi_tvt_nids.pkl' % args.dataset, 'rb') as fp:
            nids = pickle.load(fp)
    num_nodes = edges[0].shape[0]
    args.num_nodes = num_nodes
    
    # Build adjacency matrices for each edge type
    A = []
    for i, edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        if args.model == 'FastGTN' and args.dataset != 'AIRPORT':
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp, value_tmp))
    edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp, value_tmp))
    num_edge_type = len(A)
    
    # Convert node features to tensor
    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    
    # Process data based on dataset
    if args.dataset == 'PPI':
        train_node = torch.from_numpy(nids[0]).type(torch.cuda.LongTensor)
        train_target = torch.from_numpy(labels[nids[0]]).type(torch.cuda.FloatTensor)
        valid_node = torch.from_numpy(nids[1]).type(torch.cuda.LongTensor)
        valid_target = torch.from_numpy(labels[nids[1]]).type(torch.cuda.FloatTensor)
        test_node = torch.from_numpy(nids[2]).type(torch.cuda.LongTensor)
        test_target = torch.from_numpy(labels[nids[2]]).type(torch.cuda.FloatTensor)
        num_classes = 121
        is_ppi = True
    else:
        train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.cuda.LongTensor)
        train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.cuda.LongTensor)
        valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.cuda.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.cuda.LongTensor)
        test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.cuda.LongTensor)
        test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.cuda.LongTensor)
        num_classes = np.max([torch.max(train_target).item(), torch.max(valid_target).item(), torch.max(test_target).item()]) + 1
        is_ppi = False
    
    final_f1, final_micro_f1 = [], []
    tmp = None
    runs = args.runs
    if args.pre_train:
        runs += 1
        pre_trained_fastGTNs = None
    
    # # Set layer_split outside the epoch loop
    # if args.model == 'FastGTN':
    #     if args.layer_split in ["train", "train_test", "all", "train_valid"]:
    #         layer_split = "train"
    #     elif args.layer_split in ["valid", "valid_test", "all", "train_valid"]:
    #         layer_split = "valid"
    #     elif args.layer_split in ["test", "valid_test", "all", "train_test"]:
    #         layer_split = "test"
    #     else:
    #         layer_split = None

    # Iterate over runs
    for l in range(runs):
        # Initialize lists to store metrics for each run
        epochs_list = []
        train_losses = []
        train_macro_f1s = []
        train_micro_f1s = []
        valid_losses = []
        valid_macro_f1s = []
        valid_micro_f1s = []
        test_losses = []
        test_macro_f1s = []
        test_micro_f1s = [] 
        runs_list = []
        runs_list.append(l)
        # Initialize model
        if args.model == 'GTN':
            model = GTN(num_edge=len(A),
                        num_channels=args.num_channels,
                        w_in=node_features.shape[1],
                        w_out=args.node_dim,
                        num_class=num_classes,
                        num_layers=args.num_layers,
                        num_nodes=num_nodes,
                        args=args)      
        elif args.model == 'FastGTN':
            if args.pre_train and l == 1:
                pre_trained_fastGTNs = []
                for layer in range(args.num_FastGTN_layers):
                    pre_trained_fastGTNs.append(copy.deepcopy(model.fastGTNs[layer].layers))
            while len(A) > num_edge_type:
                del A[-1]
            model = FastGTNs(num_edge_type=len(A),
                            w_in=node_features.shape[1],
                            num_class=num_classes,
                            num_nodes=node_features.shape[0], output_folder=output_folder, args=args)
            if args.pre_train and l > 0:
                for layer in range(args.num_FastGTN_layers):
                    model.fastGTNs[layer].layers = pre_trained_fastGTNs[layer]
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.cuda()
        
        if args.dataset == 'PPI':
            loss = nn.BCELoss()
        else:
            loss = nn.CrossEntropyLoss()
        Ws = []
        
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1, best_micro_train_f1 = 0, 0
        best_val_f1, best_micro_val_f1 = 0, 0
        best_test_f1, best_micro_test_f1 = 0, 0
        
        # Iterate over epochs
        for i in range(args.epoch):
            print('Epoch ', i)
            epochs_list.append(i)
            model.zero_grad()
            model.train()
            
            if args.model == 'FastGTN':
                if args.layer_split in ["train", "train_test", "all", "train_valid"]:
                    loss, y_train, W = model(A, node_features, train_node, train_target, epoch=i, layer_split="train")
                else:
                    loss, y_train, W = model(A, node_features, train_node, train_target, epoch=i, layer_split=None)
            else:
                loss, y_train, W = model(A, node_features, train_node, train_target)
            
            if args.dataset == 'PPI':
                y_train = (y_train > 0).detach().float().cpu()
                train_f1 = 0.0
                sk_train_f1 = sk_f1_score(train_target.detach().cpu().numpy(), y_train.numpy(), average='micro')
            else:
                train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(), dim=1), train_target, num_classes=num_classes)).cpu().numpy()
                sk_train_f1 = sk_f1_score(train_target.detach().cpu(), np.argmax(y_train.detach().cpu(), axis=1), average='micro')
            
            train_losses.append(loss.detach().cpu().numpy())
            train_macro_f1s.append(train_f1)
            train_micro_f1s.append(sk_train_f1)
            print(W)
            print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1, sk_train_f1))
            
            loss.backward()
            optimizer.step()
            model.eval()
            
            # Validation
            with torch.no_grad():
                # Use the determined layer_split variable here instead of args.layer_split
                if args.model == 'FastGTN':
                    if args.layer_split in ["valid", "valid_test", "all", "train_valid"]:
                        val_loss, y_valid, _ = model.forward(A, node_features, valid_node, valid_target, epoch=i, layer_split="valid")
                    else:
                        val_loss, y_valid, _ = model.forward(A, node_features, valid_node, valid_target, epoch=i, layer_split=None)

                else:
                    val_loss, y_valid, _ = model.forward(A, node_features, valid_node, valid_target)
                
                if args.dataset == 'PPI':
                    val_f1 = 0.0
                    y_valid = (y_valid > 0).detach().float().cpu()
                    sk_val_f1 = sk_f1_score(valid_target.detach().cpu().numpy(), y_valid.numpy(), average='micro')
                else:
                    val_f1 = torch.mean(f1_score(torch.argmax(y_valid, dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                    sk_val_f1 = sk_f1_score(valid_target.detach().cpu(), np.argmax(y_valid.detach().cpu(), axis=1), average='micro')
                
                valid_losses.append(val_loss.detach().cpu().numpy())
                valid_macro_f1s.append(val_f1)
                valid_micro_f1s.append(sk_val_f1)
                print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1, sk_val_f1))
            
            # Test
            with torch.no_grad():
                # Use the determined layer_split variable here instead of args.layer_split
                if args.model == 'FastGTN':
                    if args.layer_split in ["test", "valid_test", "all", "train_test"]:
                        test_loss, y_test, _ = model.forward(A, node_features, test_node, test_target, epoch=i, layer_split="test")
                    else:
                        test_loss, y_test, _ = model.forward(A, node_features, test_node, test_target, epoch=i, layer_split=None)

                else:
                    test_loss, y_test, _ = model.forward(A, node_features, test_node, test_target)
                
                if args.dataset == 'PPI':
                    y_test = (y_test > 0).detach().float().cpu()
                    test_f1 = 0.0
                    sk_test_f1 = sk_f1_score(test_target.detach().cpu().numpy(), y_test.numpy(), average='micro')
                else:
                    test_f1 = torch.mean(f1_score(torch.argmax(y_test.detach(), dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                    sk_test_f1 = sk_f1_score(test_target.detach().cpu(), np.argmax(y_test.detach().cpu(), axis=1), average='micro')
                
                test_losses.append(test_loss.detach().cpu().numpy())
                test_macro_f1s.append(test_f1)
                test_micro_f1s.append(sk_test_f1)
                print('Test - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(test_loss.detach().cpu().numpy(), test_f1, sk_test_f1))
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = loss
                best_train_f1 = val_f1
                best_micro_train_f1 = sk_val_f1
                best_val_f1 = val_f1
                best_micro_val_f1 = sk_val_f1
                best_test_f1 = test_f1
                best_micro_test_f1 = sk_test_f1
                tmp = model.state_dict()
        
        # Print metrics for the best model
        print('Best val loss:', best_val_loss)
        print('Best train loss:', best_train_loss)
        print('Best test loss:', best_test_loss)
        print('Best val Macro F1:', best_val_f1)
        print('Best val Micro F1:', best_micro_val_f1)
        print('Best test Macro F1:', best_test_f1)
        print('Best test Micro F1:', best_micro_test_f1)
        
        # Append metrics for the best model to lists
        final_f1.append(best_test_f1)
        final_micro_f1.append(best_micro_test_f1)
        all_epochs_list.append(epochs_list)
        all_runs_list.append(runs_list)
        all_train_losses.append(train_losses)
        all_valid_losses.append(valid_losses)
        all_test_losses.append(test_losses)
        all_train_macro_f1s.append(train_macro_f1s)
        all_train_micro_f1s.append(train_micro_f1s)
        all_valid_macro_f1s.append(valid_macro_f1s)
        all_valid_micro_f1s.append(valid_micro_f1s)
        all_test_macro_f1s.append(test_macro_f1s)
        all_test_micro_f1s.append(test_micro_f1s)
        
        # Save the model with the best validation loss
        if args.save_metrics:
            if args.non_local:
                torch.save(tmp, os.path.join(output_folder, date_time + '_' + args.model + '_non_local_' + args.dataset + '_best.pt'))
            else:
                torch.save(tmp, os.path.join(output_folder, date_time + '_' + args.model + '_' + args.dataset + '_best.pt'))
            write_metric_csv(output_folder, date_time, args, all_epochs_list, all_runs_list, all_train_losses, all_valid_losses, all_test_losses,
                             all_train_macro_f1s, all_train_micro_f1s, all_valid_macro_f1s, all_valid_micro_f1s, all_test_macro_f1s, all_test_micro_f1s)


# model_save_path= os.path.join(output_folder,f'{args.model}_{num_layers}_{num_channels}_{args.runs}.pt')
# torch.save(model,model_save_path) 
# print(f" Model saved at {model_save_path}") 
