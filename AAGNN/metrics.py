import torch
from . import utils

def acc(predictions, labels):
    correct = (predictions.argmax(1) == labels).sum()
    acc = correct / predictions.size(dim=0)
    return acc.item()


def calc_acc(model, features, adj, labels, idx=False):
    if not idx:
        idx = torch.ones_like(labels) > 0

    pred = model(features, adj)

    correct = (pred.argmax(1)[idx] == labels[idx]).sum()
    acc = correct / idx.sum()
    return acc.item()

def partial_acc(predictions, labels, g0, g_g0):
    g0_acc = acc(predictions[g0], labels[g0])
    gX_acc = acc(predictions[g_g0], labels[g_g0])
    print(f"G0: {g0_acc:.2%}")
    print(f"GX: {gX_acc:.2%}")

    return {
        "g0": g0_acc,
        "gX": gX_acc
    }

def mask_adj(adj, bool_list, device):
    idx = utils.bool_to_idx(bool_list).squeeze().to(device)

    temp_adj = adj.clone().to(device)
    temp_adj.index_fill_(dim=0, index=idx, value=0)
    diff = adj - temp_adj

    temp_adj = diff.clone().to(device)
    temp_adj.index_fill_(dim=1, index=idx, value=0)
    diff = diff - temp_adj

    # add = int(diff.clamp(0,1).sum() / 2)
    # remove = int(diff.clamp(-1,0).abs().sum() / 2)

    return diff

def show_metrics(changes, labels, g0, device):
    def print_same_diff(type, adj):
        edges = utils.to_edges(adj)
        same = 0
        for edge in edges.t():
            same += int(labels[edge[0]].item() == labels[edge[1]].item())
        
        diff = edges.shape[1] - same

        print(f"     {type}   {int(same)}  \t{int(diff)}  \t{int(same+diff)}")


    def print_add_remove(adj):
        add = adj.clamp(0,1)
        remove = adj.clamp(-1,0).abs()
        print("                A-A\tA-B\tTOTAL")
        print_same_diff("     (+)", add)
        print_same_diff("     (-)", remove)
    # print_add_remove(changes)

    print("     Within G0 ====")
    g0_adj = mask_adj(changes, g0, device)
    print_add_remove(g0_adj)

    print("     Within GX ====")
    gX_adj = mask_adj(changes, ~g0, device)
    print_add_remove(gX_adj)

    print("     Between G0-GX ====")
    g0gX_adj = (changes - g0_adj - gX_adj)
    print_add_remove(g0gX_adj)

    print()
    print_same_diff("   TOTAL", changes)