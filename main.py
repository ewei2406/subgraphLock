import torch
import numpy as np
import argparse

from getPokec import MyDataset
from deeprobust.graph.defense import GCN

################################################
# Configuration
################################################

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')

parser.add_argument('--model_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden_layers', type=int, default=32, help='Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')

parser.add_argument('--protect_size', type=float, default=0.14, help='Number of randomly chosen protected nodes')
parser.add_argument('--random_select', type=str, default="Y", help='Choose a class instead')
parser.add_argument('--ptb_rate', type=float, default=0.5, help='Perturbation rate (percentage of available edges)')

parser.add_argument('--reg_epochs', type=int, default=50, help='Epochs to train models')
parser.add_argument('--ptb_epochs', type=int, default=20, help='Epochs to perturb adj matrix')
parser.add_argument('--surrogate_epochs', type=int, default=20, help='Epochs to train surrogate')

parser.add_argument('--csv', type=str, default='', help='save the outputs to csv')
args = parser.parse_args()

################################################
# Environment
################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

print('==== Environment ====')
print(f'  torch version: {torch.__version__}')
print(f'  device: {device}')
print(f'  torch seed: {args.seed}')

################################################
# Load data
################################################

print('==== Dataset: Pokec ====')

Pokec = MyDataset("pokec2")
data = Pokec.getComponents()

data["features"] = data["features"].t()

task2classes = data["features"][0]

data["features"] = data["features"][1:]
data["features"] = data["features"].t()

task1classes = data["labels"]

print(data["features"])
print(task1classes)
print(task2classes)

################################################
# Baseline model
################################################

print("1")
# Classifier for labels
baseline_task1 = GCN(
    nfeat=data["features"].shape[1],
    nclass=2,
    nhid=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay)
print("2")

baseline_task1.fit(
    features=data["features"], 
    adj=data["adj"], 
    labels=task1classes, 
    idx_train=data["idx_train"], 
    idx_val=data["idx_test"], 
    train_iters=5,
    verbose=True
)