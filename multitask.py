import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn.functional as F

from AAGNN.dataset import Dataset
from AAGNN.loadGraph import loadGraph
from AAGNN.GCN import GCN
from AAGNN import metrics
from AAGNN import utils

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
parser.add_argument('--ptb_epochs', type=int, default=10, help='Epochs to perturb adj matrix')
parser.add_argument('--surrogate_epochs', type=int, default=10, help='Epochs to train surrogate before perturb')

parser.add_argument('--csv', type=str, default='', help='save the outputs to csv')
parser.add_argument('--dataset', type=str, default='blogcatalog', help='dataset')
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

print(f'==== Dataset: {args.dataset} ====')

adj, labels, features, idx_train, idx_val, idx_test = loadGraph('./datasets', args.dataset, 'gcn', args.seed, device)

task2idx = 1

task1labels = labels

task2labels = features.t()[task2idx].long()

print(task1labels)
print(task2labels.sum())

features = features.t()
features = torch.cat([features[0:task2idx], features[task2idx:]]).t()

# print(adj.sum())

################################################
# Protected set
################################################

if args.random_select == "Y":
    g0 = torch.rand(features.shape[0]) <= args.protect_size
else:
    g0 = labels == 1

g_g0 = ~g0

print(f"Number of protected nodes: {g0.sum():.0f}")
print(f"Protected Size: {g0.sum() / features.shape[0]:.2%}")

################################################
# Baseline model
################################################

print(f'==== Training baselines ====')

# Task 1

baseline_task1 = GCN(
    input_features=features.shape[1],
    output_classes=task1labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name="baseline_task1"
)

baseline_task1.fit(
    features=features, 
    adj=adj, 
    labels=task1labels, 
    idx_train=idx_train, 
    idx_test=idx_test, 
    epochs=args.reg_epochs
)

pred = baseline_task1(features, adj)
base1_acc = metrics.partial_acc(pred, task1labels, g0, g_g0)

# Task 2

baseline_task2 = GCN(
    input_features=features.shape[1],
    output_classes=task2labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name="baseline_task2"
)

baseline_task2.fit(
    features=features, 
    adj=adj, 
    labels=task2labels, 
    idx_train=idx_train, 
    idx_test=idx_test, 
    epochs=args.reg_epochs,
)

pred = baseline_task2(features, adj)
base2_acc = metrics.partial_acc(pred, task2labels, g0, g_g0)

################################################
# Perturbing
################################################

surrogate_task1 = GCN(
    input_features=features.shape[1],
    output_classes=task1labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name="surrogate_task1"
)

surrogate_task1.fit(
    features=features, 
    adj=adj, 
    labels=task1labels, 
    idx_train=idx_train, 
    idx_test=idx_test, 
    epochs=args.surrogate_epochs
)

surrogate_task2 = GCN(
    input_features=features.shape[1],
    output_classes=task2labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name="surrogate_task2"
)

surrogate_task2.fit(
    features=features, 
    adj=adj, 
    labels=task2labels, 
    idx_train=idx_train, 
    idx_test=idx_test, 
    epochs=args.surrogate_epochs,
)

perturbations = torch.zeros_like(adj).float()
perturbations.requires_grad = True
num_perturbations = int(args.ptb_rate * (adj.sum() / 2))

t = tqdm(range(args.ptb_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
t.set_description("Perturbing")

def loss_func(p1, l1, p2, l2):
    loss = F.cross_entropy(p1[g0], l1[g0]) \
        + F.cross_entropy(p2[g0], l2[g0]) \
        - F.cross_entropy(p1[g_g0], l1[g_g0]) \
        - F.cross_entropy(p2[g_g0], l2[g_g0])

    return loss

for epoch in t:
    # Perturb
    surrogate_task1.eval()
    surrogate_task2.eval()

    modified_adj = utils.get_modified_adj(adj, perturbations)

    pred1 = surrogate_task1(features, modified_adj) 
    pred2 = surrogate_task2(features, modified_adj) 

    loss = loss_func(pred1, task1labels, pred2, task2labels)
    adj_grad = torch.autograd.grad(loss, perturbations)[0]

    lr = (num_perturbations * 0.02) / (epoch + 1)
            
    perturbations = perturbations + (lr * adj_grad)

    pre_projection = int(perturbations.sum() / 2)

    perturbations = utils.projection(perturbations, num_perturbations)

    # Train
    modified_adj = utils.get_modified_adj(adj, perturbations)
    l1 = surrogate_task1.train1epoch(features, modified_adj, task1labels, idx_train, idx_test)
    modified_adj = utils.get_modified_adj(adj, perturbations)
    l2 = surrogate_task2.train1epoch(features, modified_adj, task2labels, idx_train, idx_test)

    t.set_postfix({"adj_l": loss.item(),
                    "adj_g": int(adj_grad.sum()),
                    "pre-p": pre_projection,
                    "target": int(num_perturbations / 2),
                    "l1": l1,
                    "l2": l2})

with torch.no_grad():

    max_loss = -1000

    for k in range(0,3):
        sample = torch.bernoulli(perturbations)
        modified_adj = utils.get_modified_adj(adj, perturbations)

        modified_adj = utils.make_symmetric(modified_adj) # Removing this creates "impossible" adj, but works well

        pred1 = surrogate_task1(features, modified_adj) 
        pred2 = surrogate_task2(features, modified_adj) 

        loss = loss_func(pred1, task1labels, pred2, task2labels)

        if loss > max_loss:
            max_loss = loss
            best = sample
    
    print(f"Best sample loss: {loss:.2f}\t Edges: {best.abs().sum() / 2:.0f}")

################################################
# Train on locked graph
################################################

print(f'==== Training baselines ====')

locked_adj = utils.get_modified_adj(adj, best)

# Task 1

locked_task1 = GCN(
    input_features=features.shape[1],
    output_classes=task1labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name="locked_task1"
)

locked_task1.fit(
    features=features, 
    adj=locked_adj, 
    labels=task1labels, 
    idx_train=idx_train, 
    idx_test=idx_test, 
    epochs=args.reg_epochs
)

pred = locked_task1(features, locked_adj)
lock1_acc = metrics.partial_acc(pred, task1labels, g0, g_g0)

# Task 2

locked_task2 = GCN(
    input_features=features.shape[1],
    output_classes=task2labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name="locked_task2"
)

locked_task2.fit(
    features=features, 
    adj=locked_adj, 
    labels=task2labels, 
    idx_train=idx_train, 
    idx_test=idx_test, 
    epochs=args.reg_epochs
)

pred = locked_task2(features, locked_adj)
lock2_acc = metrics.partial_acc(pred, task2labels, g0, g_g0)

################################################
# Evaluation
################################################

locked_adj = utils.get_modified_adj(adj, best)
change = locked_adj - adj

dg0_1 = lock1_acc["g0"] - base1_acc["g0"]
dgX_1 = lock1_acc["gX"] - base1_acc["gX"]
dg0_2 = lock2_acc["g0"] - base2_acc["g0"]
dgX_2 = lock2_acc["gX"] - base2_acc["gX"]

print("==== Accuracies ====")
print(f"         ΔG0\tΔGX")
print(f"task1 | {dg0_1:.1%}\t{dgX_1:.1%}")
print(f"task2 | {dg0_2:.1%}\t{dgX_2:.1%}")

print()
print("=====TASK 1=====")
metrics.show_metrics(change, task1labels, g0)

print()
print("=====TASK 2=====")
metrics.show_metrics(change, task2labels, g0)
