"""
Make our own datasets
We store three components for each dataset
-- node_feature.csv: store node feature
-- node_label.csv: store node label
-- edge.csv: store the edges

Look for pokec in ./pokec
"""

import dgl
from dgl.data import DGLDataset
import torch
from dgl import backend as F
import os
# import urllib.request
import pandas as pd
import numpy as np
import networkx as nx


class MyDataset(DGLDataset):
    def __init__(self, data_name):
        self.data_name = data_name
        super().__init__(name='customized_dataset')

    def process(self):
        # Load the data as DataFrame
        node_features = pd.read_csv(
            './processedData/{}_node_feature.csv'.format(self.data_name.split('_')[0]))
        node_labels = pd.read_csv(
            './processedData/{}_node_label.csv'.format(self.data_name.split('_')[0]))
        edges = pd.read_csv('./processedData/{}_edge.csv'.format(self.data_name))

        node_attributes = None
        if os.path.isfile('./processedData/{}_node_attribute.csv'.format(self.data_name.split('_')[0])):
            node_attributes = pd.read_csv(
                './processedData/{}_node_attribute.csv'.format(self.data_name.split('_')[0]))

        c = node_labels['Label'].astype('category')
        classes = dict(enumerate(c.cat.categories))
        self.num_classes = len(classes)

        # Transform from DataFrame to torch tensor
        node_features = torch.from_numpy(node_features.to_numpy()).float()
        node_labels = torch.from_numpy(node_labels['Label'].to_numpy()).long()
        edge_features = torch.from_numpy(edges['Weight'].to_numpy()).float()
        edges_src = torch.from_numpy(edges['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges['Dst'].to_numpy())

        # construct DGL graph
        # !!! note to turn a directed graph into a undirected one
        # otherwise the graph embedding performance might be compromised
        g = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        g.ndata['feat'] = node_features
        g.ndata['label'] = node_labels
        g.edata['weight'] = edge_features

        nx_g = dgl.to_networkx(g)
        # print(nx.classes.function.density(nx_g))
        # print(nx_g.number_of_nodes())
        # print(nx_g.number_of_edges())

        if node_attributes is not None:
            for l in list(node_attributes):
                g.ndata[l] = torch.from_numpy(
                    node_attributes[l].to_numpy()).long()

        # rewrite the to_bidirected function to support edge weights on bidirected graph (aggregated)
        # self.graph = dgl.to_bidirected(g)
        g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
        g = dgl.to_simple(g, return_counts=None,
                          copy_ndata=True, copy_edata=True)

        # zero in-degree nodes will lead to invalid output value
        # a common practice to avoid this is to add a self-loop
        self.graph = dgl.add_self_loop(g)
        # self.graph = g

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        # print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.graph.ndata['feat'].shape[1]))
        print('  NumClasses: {}'.format(self.num_classes))
        print('  NumTrainingSamples: {}'.format(
            F.nonzero_1d(self.graph.ndata['train_mask']).shape[0]))
        print('  NumValidationSamples: {}'.format(
            F.nonzero_1d(self.graph.ndata['val_mask']).shape[0]))
        print('  NumTestSamples: {}'.format(
            F.nonzero_1d(self.graph.ndata['test_mask']).shape[0]))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    def getComponents(self):
        return {
            "adj": self.graph.adj().to_dense(),
            "labels": self.graph.ndata['label'],
            "features": self.graph.ndata['feat'],
            "idx_train": self.graph.ndata['train_mask'],
            "idx_val": self.graph.ndata['val_mask'],
            "idx_test": self.graph.ndata['test_mask'],
        }

def process_raw_pokec():

    os.makedirs('./processedData', exist_ok=True)

    edge_file = './pokec/region_job_2_relationship.txt'
    node_file = './pokec/region_job_2.csv'

    edges = pd.read_csv(edge_file, sep='\t', names=['Src', 'Dst'])
    nodes = pd.read_csv(node_file, sep=',', header=0)

    node_ids = list(nodes['user_id'])
    new_ids = list(range(len(node_ids)))

    id_map = dict(zip(node_ids, new_ids))
    nodes['Label'] = nodes['public']
    node_labels = nodes.filter(['Label'])

    edges['Weight'] = np.ones(edges.shape[0])
    edges['Src'].replace(id_map, inplace=True)
    edges['Dst'].replace(id_map, inplace=True)

    node_attributes = nodes.filter(['gender', 'region', 'AGE'])
    node_features = nodes.drop(columns=['Label', 'user_id', 'public',
                                        'completion_percentage', 'gender', 'region', 'AGE'])

    node_attributes.to_csv(
        './processedData/pokec2_node_attribute.csv', sep=',', index=False)
    node_features.to_csv(
        './processedData/pokec2_node_feature.csv', sep=',', index=False)
    node_labels.to_csv('./processedData/pokec2_node_label.csv', sep=',', index=False)
    edges.to_csv('./processedData/pokec2_edge.csv', sep=',', index=False)

# First process data into the unified csv format
# process_raw_karate()
# process_raw_movielens()

if __name__ == "__main__":
    process_raw_pokec()

# d = MyDataset("pokec2")
# print(d.graph.adj().to_dense())
