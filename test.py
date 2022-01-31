from AAGNN.loadGraph import loadPokec
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj, labels, features, idx_train, idx_val, idx_test = loadPokec(device)

print(adj.shape)
print(labels.shape)
print(features.shape)
