"""
attention_curvature.py

Created on Tue Mar 12 2024

@author: Lukas

This file contains functionality to compute the curvature 
of the computation graph of an attention-based GNN.
"""

import os.path as osp
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, HeterophilousGraphDataset
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj

from GraphRicciCurvature.OllivierRicci import OllivierRicci

import networkx as nx


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, num_heads):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, 8, heads=num_heads, dropout=0.6))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(8 * num_heads, 8, heads=num_heads, dropout=0.6))
        self.convs.append(GATConv(8 * num_heads, num_classes, heads=num_heads, concat=False, dropout=0.6))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = F.elu(self.convs[i](x, edge_index))
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def attention(self, data):
        x, edge_index = data.x, data.edge_index
        attention_scores = []
        for i in range(self.num_layers - 1):
            x, att = self.convs[i](x, edge_index, return_attention=True)
            attention_scores.append(att)
        x, att = self.convs[-1](x, edge_index, return_attention=True)
        attention_scores.append(att)
        return attention_scores


class Experiment:
    def __init__(self, dataset, num_layers, num_heads=1):
        self.dataset = dataset
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model()
        self.epochs = 0

    def create_model(self):
        """
        Create a GAT model for the given dataset with the given number of layers and heads.
        The dataset is one of Mutag, Enzymes, or Proteins, and has type list.

        Each element of the list is of the form Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1])
        """
        dataset = self.dataset
        num_features = dataset[0].num_features
        # we only run binary classification experiments
        num_classes = 2
        model = GAT(num_features, num_classes, self.num_layers, self.num_heads).to(self.device)
        return model
    
    def train(self, epochs=100):
        """
        Train the GAT model on the given binary classification dataset
        for the given number of epochs.
        """
        model = self.model
        device = self.device
        dataset = self.dataset
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for data in dataset:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
        self.epochs += epochs
        print(f"Model trained for {self.epochs} total epochs.")
        return model
    
    def create_attention_digraph(self, input_graph, layer):
        """
        Create a directed graph from the attention scores of the given layer.
        """
        attention_scores = self.model.attention(input_graph)[layer]
        G = nx.DiGraph()
        num_edges = attention_scores.shape[1]
        G.add_nodes_from(range(num_edges))
        for i in range(num_edges):
            G.add_edge(attention_scores[0, i].item(), attention_scores[1, i].item(), weight=i)
        # remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    
def compute_edge_curvatures(graph):
    """
    Compute the Ollivier-Ricci curvature of a given directed and weighted graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph.

    Returns
    -------
    curvatures : dict
        A dictionary of edge curvatures.
    """
    curvatures = {}
    for u, v in graph.edges:
        curvatures[(u, v)] = OllivierRicci(graph, u, v).compute_ricci_curvature()
    return curvatures